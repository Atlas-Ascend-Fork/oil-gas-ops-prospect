/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * FusedSoftmaxGrad kernel 实现（两条路径，同一算子 tiling key 分发）：
 *
 * - FusedSoftmaxGradMix：全部 bf16。MIX_AIC_1_2 两阶段：未对齐时 AIV 先把
 *   go/V pad 到 padS/padD，再 24 AIC 高阶 MatmulImpl 逐 plane 算 g(half) 写
 *   user workspace -> SyncAll -> 48 AIV 读回做 JVP 原地写回。
 * - FusedSoftmaxGradVec：fp32/fp16。48 AIV 纯向量：per plane V 驻留 UB，逐
 *   (q,s) 点积得 g 行，同核 JVP 写回。
 *
 * 约束（实证）：队列 depth=1；行和用 Sum 高阶 api（禁 Level-2 ReduceSum 标量
 * 管道）；裸 TBuf 路径必须显式 MTE2_V / V_MTE3 / MTE3_V 管道事件；g 缓冲一律
 * GetUserWorkspace。
 */
#ifndef FUSED_SOFTMAX_GRAD_IMPL_H
#define FUSED_SOFTMAX_GRAD_IMPL_H

/*!
 * \file fused_softmax_grad.h
 * \brief FusedSoftmaxGrad kernel: g = go @ V^T (plane aligned);
 *        probs <- probs ⊙ (g − rowsum(probs⊙g)), fp32 domain, in-place.
 *
 * 两条路径 (同一算子, tiling key 分发):
 * - FusedSoftmaxGradMix: 全部 bf16。未对齐先 pad 再 Cube，再 AIV JVP 原地写回。
 * - FusedSoftmaxGradVec: fp32/fp16。48 AIV 纯向量点积 + 同核 JVP。
 *
 * 约束 (交接文档实证): 队列 depth=1; 关闭 --cce-auto-sync; 行和用 Sum 高阶 api
 * (禁 Level-2 ReduceSum 标量管道); g 缓冲一律 GetUserWorkspace。
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/reduce/sum.h"

namespace AscendC {

constexpr uint64_t FSG_SUM_TMP_BYTES = 4096; // Sum 高阶 api 自备 tmp

template <typename T1, typename T2>
__aicore__ inline T1 FsgCeilDiv(T1 a, T2 b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

template <typename T1, typename T2>
__aicore__ inline T1 FsgMin(T1 a, T2 b)
{
    return (a < b) ? a : b;
}

// 共享辅助: 多行 DataCopyPad 载入, GM 行主序 [rows, cols] -> UB [rows, padCols] (rightPad 补齐)
template <typename UT>
__aicore__ inline void FsgCopyRowsIn(LocalTensor<UT>& dst, GlobalTensor<UT> src, uint64_t rows, uint64_t cols,
                                     uint64_t padCols)
{
    DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(UT)), 0, 0, 0};
    DataCopyPadExtParams<UT> extParams{true, 0, static_cast<uint8_t>(padCols - cols), 0.0};
    DataCopyPad(dst, src, params, extParams);
}

// ---------------------------------------------------------------------------
// 纯向量路径: fp32 / fp16 / 非对齐 bf16
// ---------------------------------------------------------------------------
template <typename T>
class FusedSoftmaxGradVec {
public:
    __aicore__ inline FusedSoftmaxGradVec() {}

    __aicore__ inline void Init(GM_ADDR probs, GM_ADDR gradOutput, GM_ADDR values, GM_ADDR out,
                                const FusedSoftmaxGradTilingData& tilingData, TPipe* pipe)
    {
        this->blockIdx = GetBlockIdx();
        this->aivNum = GetBlockNum() * 2; // MIX_AIC_1_2: AIV = 2 * blockDim
        this->planes = tilingData.planes;
        this->qLen = tilingData.height;
        this->sLen = tilingData.width;
        this->dLen = tilingData.dLen;
        this->padS = tilingData.padS;
        this->padD = tilingData.padD;

        gmP.SetGlobalBuffer((__gm__ T*)probs);
        gmGo.SetGlobalBuffer((__gm__ T*)gradOutput);
        gmV.SetGlobalBuffer((__gm__ T*)values);
        gmOut.SetGlobalBuffer((__gm__ T*)out);

        if ASCEND_IS_AIV {
            // V plane fp32 驻留 + 行缓冲 (tiling 侧已校验 UB 预算)
            pipe->InitBuffer(bufV32, sLen * padD * sizeof(float));
            pipe->InitBuffer(bufGoq32, padD * sizeof(float));
            pipe->InitBuffer(bufGrow, padS * sizeof(float));
            pipe->InitBuffer(bufP32, padS * sizeof(float));
            pipe->InitBuffer(bufT, FsgMax(padS, padD) * sizeof(float));
            pipe->InitBuffer(bufRs, FSG_RS_ELEMS * sizeof(float));
            pipe->InitBuffer(bufSumTmp, FSG_SUM_TMP_BYTES);
            if constexpr (!std::is_same<T, float>::value) {
                pipe->InitBuffer(bufV16, sLen * padD * sizeof(T));
                pipe->InitBuffer(bufGoq16, padD * sizeof(T));
                pipe->InitBuffer(bufP16, padS * sizeof(T));
                pipe->InitBuffer(bufOut16, padS * sizeof(T));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            return; // 向量路径: AIC 空转
        }
        for (uint64_t p = blockIdx; p < planes; p += aivNum) {
            ProcessPlane(p);
        }
    }

private:
    __aicore__ inline uint64_t FsgMax(uint64_t a, uint64_t b) { return (a > b) ? a : b; }

    // 载入 V plane [S, D] -> fp32 [S, padD]（裸 TBuf 路径需显式 MTE2_V 同步）
    __aicore__ inline void LoadVPlane(uint64_t vBase, LocalTensor<float>& v32)
    {
        if constexpr (std::is_same<T, float>::value) {
            FsgCopyRowsIn(v32, gmV[vBase], sLen, dLen, padD);
        } else {
            LocalTensor<T> v16 = bufV16.Get<T>();
            FsgCopyRowsIn(v16, gmV[vBase], sLen, dLen, padD);
            SyncMte2ToV();
            ChunkedCast(v32, v16, RoundMode::CAST_NONE, sLen * padD);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (std::is_same<T, float>::value) {
            SyncMte2ToV();
        }
    }

    // 单 plane: V 驻留; 逐 q: S 个点积 -> g 行 -> JVP -> 原地写回
    // 载入 go 行 q [D] -> fp32 [padD]，并计算该行与 V 驻留块的全部点积写入 grow
    __aicore__ inline void LoadGoRowAndDot(LocalTensor<float>& v32, LocalTensor<float>& goq32, GlobalTensor<T>& gmGoSrc,
                                           uint64_t goRowOff, LocalTensor<float>& t, LocalTensor<float>& rs,
                                           LocalTensor<uint8_t>& sumTmp, LocalTensor<float>& growOut)
    {
        if constexpr (std::is_same<T, float>::value) {
            FsgCopyRowsIn(goq32, gmGoSrc[goRowOff], 1, dLen, padD);
        } else {
            LocalTensor<T> goq16 = bufGoq16.Get<T>();
            FsgCopyRowsIn(goq16, gmGoSrc[goRowOff], 1, dLen, padD);
            SyncMte2ToV();
            ChunkedCast(goq32, goq16, RoundMode::CAST_NONE, padD);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (std::is_same<T, float>::value) {
            SyncMte2ToV();
        }
        for (uint64_t s = 0; s < sLen; ++s) {
            Mul(t, v32[s * padD], goq32, padD);
            PipeBarrier<PIPE_V>();
            SumParams dotParams{1u, static_cast<uint32_t>(padD), static_cast<uint32_t>(dLen)};
            Sum<float>(rs, t, sumTmp, dotParams);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            float val = rs.GetValue(0);
            growOut.SetValue(s, val);
        }
    }

    __aicore__ inline void LoadPRow(LocalTensor<float>& p32, uint64_t rowOff)
    {
        if constexpr (std::is_same<T, float>::value) {
            FsgCopyRowsIn(p32, gmP[rowOff], 1, sLen, padS);
        } else {
            LocalTensor<T> p16 = bufP16.Get<T>();
            FsgCopyRowsIn(p16, gmP[rowOff], 1, sLen, padS);
            SyncMte2ToV();
            ChunkedCast(p32, p16, RoundMode::CAST_NONE, padS);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (std::is_same<T, float>::value) {
            SyncMte2ToV();
        }
    }

    // JVP: out = p ⊙ (g − rowsum(p⊙g))，然后原地写回行 q。
    // V 管道结果须经 V_MTE3 后方可用 MTE3 搬出；store 完成前禁止下一迭代
    // 的 V 管道覆写 t/out16，需 MTE3_V 回边。
    __aicore__ inline void ApplyJvpAndStore(LocalTensor<float>& p32, LocalTensor<float>& grow, LocalTensor<float>& t,
                                            LocalTensor<float>& rs, LocalTensor<uint8_t>& sumTmp, uint64_t rowOff)
    {
        Mul(t, p32, grow, padS);
        PipeBarrier<PIPE_V>();
        SumParams rowParams{1u, static_cast<uint32_t>(padS), static_cast<uint32_t>(sLen)};
        Sum<float>(rs, t, sumTmp, rowParams);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float sc = rs.GetValue(0);
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
        Adds(grow, grow, -sc, sLen);
        PipeBarrier<PIPE_V>();
        Mul(t, p32, grow, padS);

        SyncVToMte3();
        if constexpr (std::is_same<T, float>::value) {
            DataCopyPad(gmOut[rowOff], t, DataCopyExtParams{1u, static_cast<uint32_t>(sLen * sizeof(T)), 0, 0, 0});
        } else {
            LocalTensor<T> out16 = bufOut16.Get<T>();
            ChunkedCast(out16, t, RoundMode::CAST_RINT, padS);
            DataCopyPad(gmOut[rowOff], out16, DataCopyExtParams{1u, static_cast<uint32_t>(sLen * sizeof(T)), 0, 0, 0});
        }
        SyncMte3ToV();
    }

    __aicore__ inline void ProcessPlane(uint64_t p)
    {
        uint64_t vBase = p * sLen * dLen;   // values [S, D] per plane
        uint64_t goBase = p * qLen * dLen;  // grad_output [Q, D] per plane
        uint64_t rowBase = p * qLen * sLen; // probs/out [Q, S] per plane

        LocalTensor<float> v32 = bufV32.Get<float>();
        LocalTensor<float> goq32 = bufGoq32.Get<float>();
        LocalTensor<float> grow = bufGrow.Get<float>();
        LocalTensor<float> p32 = bufP32.Get<float>();
        LocalTensor<float> t = bufT.Get<float>();
        LocalTensor<float> rs = bufRs.Get<float>();
        LocalTensor<uint8_t> sumTmp = bufSumTmp.Get<uint8_t>();

        LoadVPlane(vBase, v32);

        for (uint64_t q = 0; q < qLen; ++q) {
            LoadGoRowAndDot(v32, goq32, gmGo, goBase + q * dLen, t, rs, sumTmp, grow);
            SetFlag<HardEvent::S_V>(EVENT_ID1);
            WaitFlag<HardEvent::S_V>(EVENT_ID1);
            LoadPRow(p32, rowBase + q * sLen);
            ApplyJvpAndStore(p32, grow, t, rs, sumTmp, rowBase + q * sLen);
        }
    }

    // 多行 DataCopyPad 载入: GM 行主序 [rows, cols] -> UB [rows, padCols] (rightPad 补齐)
    // 裸 TBuf 路径的管道同步 (队列的 EnQue/DeQue 等价物)
    __aicore__ inline void SyncMte2ToV()
    {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    }

    __aicore__ inline void SyncVToMte3()
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
    }

    template <typename DST, typename SRC>
    __aicore__ inline void ChunkedCast(DST& dst, SRC& src, RoundMode mode, uint64_t count)
    {
        constexpr uint64_t CHUNK = 16384; // Level-2 repeatTimes 上限保护
        for (uint64_t off = 0u; off < count; off += CHUNK) {
            uint64_t n = FsgMin(count - off, CHUNK);
            Cast(dst[off], src[off], mode, n);
        }
    }

private:
    GlobalTensor<T> gmP;
    GlobalTensor<T> gmGo;
    GlobalTensor<T> gmV;
    GlobalTensor<T> gmOut;

    TBuf<TPosition::VECCALC> bufV32;
    TBuf<TPosition::VECCALC> bufGoq32;
    TBuf<TPosition::VECCALC> bufGrow;
    TBuf<TPosition::VECCALC> bufP32;
    TBuf<TPosition::VECCALC> bufT;
    TBuf<TPosition::VECCALC> bufRs;
    TBuf<TPosition::VECCALC> bufSumTmp;
    TBuf<TPosition::VECCALC> bufV16;
    TBuf<TPosition::VECCALC> bufGoq16;
    TBuf<TPosition::VECCALC> bufP16;
    TBuf<TPosition::VECCALC> bufOut16;

    static constexpr uint64_t FSG_RS_ELEMS = 8; // 单行 Sum 结果缓冲 (32B 对齐)

    uint64_t blockIdx;
    uint64_t aivNum;
    uint64_t planes;
    uint64_t qLen;
    uint64_t sLen;
    uint64_t dLen;
    uint64_t padS;
    uint64_t padD;
};

// ---------------------------------------------------------------------------
// MIX 路径: 全部 bf16（未对齐 AIV pad 后再 Cube）
// ---------------------------------------------------------------------------
template <typename T, typename MM_TYPE>
class FusedSoftmaxGradMix {
public:
    __aicore__ inline FusedSoftmaxGradMix(MM_TYPE& mm_) : mm(mm_) {}

    __aicore__ inline void Init(GM_ADDR probs, GM_ADDR gradOutput, GM_ADDR values, GM_ADDR out, GM_ADDR gWs,
                                const FusedSoftmaxGradTilingData& tilingData, TPipe* pipe)
    {
        this->blockIdx = GetBlockIdx();
        this->aicNum = GetBlockNum(); // = 24
        this->aivNum = aicNum * 2;    // = 48
        this->planes = tilingData.planes;
        this->qLen = tilingData.height;
        this->sLen = tilingData.width;
        this->dLen = tilingData.dLen;
        this->padS = tilingData.padS;
        this->padD = tilingData.padD;
        this->rowsPerIter = tilingData.rowsPerIter;
        this->needPad = (this->dLen != this->padD || this->sLen != this->padS) ? 1 : 0;
        InitGm(probs, gradOutput, values, out, gWs);
        if ASCEND_IS_AIC {
            // Cube 用对齐后的 padS/padD；未对齐时 A/B 来自 pad workspace
            mm.SetOrgShape(qLen, padS, padD);
            mm.SetSingleShape(qLen, padS, padD);
        }
        if ASCEND_IS_AIV {
            InitAivPipe(pipe);
        }
    }

    __aicore__ inline void Process()
    {
        if (needPad == 1) {
            if ASCEND_IS_AIV {
                PadPlanes();
            }
            SyncAll<false>();
        }
        if ASCEND_IS_AIC {
            for (uint64_t p = blockIdx; p < planes; p += aicNum) {
                if (needPad == 1) {
                    mm.SetTensorA(gmAPad[p * qLen * padD], false);
                    mm.SetTensorB(gmBPad[p * padS * padD], true);
                } else {
                    mm.SetTensorA(gmGo[p * qLen * dLen], false);
                    mm.SetTensorB(gmV[p * sLen * dLen], true);
                }
                mm.template IterateAll<true>(gmWs[p * qLen * padS], 0);
            }
            SyncAll<false>();
        } else {
            SyncAll<false>();
            ProcessJvp();
        }
    }

private:
    __aicore__ inline void InitGm(GM_ADDR probs, GM_ADDR gradOutput, GM_ADDR values, GM_ADDR out, GM_ADDR gWs)
    {
        gmP.SetGlobalBuffer((__gm__ T*)probs);
        gmGo.SetGlobalBuffer((__gm__ bfloat16_t*)gradOutput);
        gmV.SetGlobalBuffer((__gm__ bfloat16_t*)values);
        const uint64_t gElems = planes * qLen * padS;
        gmWs.SetGlobalBuffer((__gm__ half*)gWs, gElems);
        gmOut.SetGlobalBuffer((__gm__ T*)out);
        if (this->needPad == 1) {
            const uint64_t aElems = planes * qLen * padD;
            const uint64_t bElems = planes * padS * padD;
            gmAPad.SetGlobalBuffer((__gm__ bfloat16_t*)(gWs + gElems * 2), aElems);
            gmBPad.SetGlobalBuffer((__gm__ bfloat16_t*)(gWs + (gElems + aElems) * 2), bElems);
        }
    }

    __aicore__ inline void InitAivPipe(TPipe* pipe)
    {
        uint64_t totalRows = planes * qLen;
        uint64_t base = totalRows / aivNum;
        uint64_t rem = totalRows % aivNum;
        this->myRows = base + (blockIdx < rem ? 1u : 0u);
        this->myStart = blockIdx * base + FsgMin(blockIdx, rem);

        if (this->needPad == 1) {
            uint64_t pr = 8192 / (padD * sizeof(T));
            if (pr == 0) {
                pr = 1;
            }
            if (pr > 32) {
                pr = 32;
            }
            this->padRows = pr;
            pipe->InitBuffer(bufPad, padRows * padD * sizeof(T));
        }

        uint64_t bufElems = rowsPerIter * padS;
        pipe->InitBuffer(inQueueP, 1, bufElems * sizeof(T));
        pipe->InitBuffer(inQueueG, 1, bufElems * sizeof(half));
        pipe->InitBuffer(outQueue, 1, bufElems * sizeof(T));
        pipe->InitBuffer(bufP32, bufElems * sizeof(float));
        pipe->InitBuffer(bufG32, bufElems * sizeof(float));
        pipe->InitBuffer(bufT, bufElems * sizeof(float));
        pipe->InitBuffer(bufRs, FsgCeilDiv(rowsPerIter, 8) * 8 * sizeof(float));
        pipe->InitBuffer(bufSumTmp, FSG_SUM_TMP_BYTES);
    }

    __aicore__ inline void PadCopy(GlobalTensor<bfloat16_t> dst, GlobalTensor<bfloat16_t> src, uint64_t rows,
                                   uint64_t cols)
    {
        LocalTensor<T> ub = bufPad.Get<T>();
        for (uint64_t r = 0; r < rows; r += padRows) {
            uint64_t n = FsgMin(padRows, rows - r);
            FsgCopyRowsIn(ub, src[r * cols], n, cols, padD);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID5);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID5);
            DataCopyPad(dst[r * padD], ub,
                        DataCopyExtParams{static_cast<uint16_t>(n), static_cast<uint32_t>(padD * sizeof(T)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        }
    }

    __aicore__ inline void ZeroPadRows(GlobalTensor<bfloat16_t> dst, uint64_t rows)
    {
        LocalTensor<T> ub = bufPad.Get<T>();
        Duplicate(ub, static_cast<T>(0), padRows * padD);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
        for (uint64_t r = 0; r < rows; r += padRows) {
            uint64_t n = FsgMin(padRows, rows - r);
            DataCopyPad(dst[r * padD], ub,
                        DataCopyExtParams{static_cast<uint16_t>(n), static_cast<uint32_t>(padD * sizeof(T)), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
        }
    }

    __aicore__ inline void PadPlanes()
    {
        for (uint64_t p = blockIdx; p < planes; p += aivNum) {
            PadCopy(gmAPad[p * qLen * padD], gmGo[p * qLen * dLen], qLen, dLen);
            PadCopy(gmBPad[p * padS * padD], gmV[p * sLen * dLen], sLen, dLen);
            if (sLen < padS) {
                ZeroPadRows(gmBPad[p * padS * padD + sLen * padD], padS - sLen);
            }
        }
    }

    // 标量域读取 rs 并逐行 Adds 回写（V_S/S_V 事件同步）
    __aicore__ inline void ApplyRowBias(LocalTensor<float>& rs, LocalTensor<float>& g32, uint64_t rows, uint64_t padS,
                                        uint64_t sLen)
    {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        for (uint64_t r = 0u; r < rows; ++r) {
            WaitFlag<HardEvent::S_V>(EVENT_ID1);
            float val = rs.GetValue(r);
            SetFlag<HardEvent::S_V>(EVENT_ID1);
            WaitFlag<HardEvent::S_V>(EVENT_ID1);
            Adds(g32[r * padS], g32[r * padS], -val, sLen);
            SetFlag<HardEvent::S_V>(EVENT_ID1);
        }
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }

    __aicore__ inline void ProcessJvp()
    {
        LocalTensor<float> p32 = bufP32.Get<float>();
        LocalTensor<float> g32 = bufG32.Get<float>();
        LocalTensor<float> t = bufT.Get<float>();
        LocalTensor<float> rs = bufRs.Get<float>();
        LocalTensor<uint8_t> sumTmp = bufSumTmp.Get<uint8_t>();

        uint64_t done = 0;
        while (done < myRows) {
            uint64_t rows = FsgMin(rowsPerIter, myRows - done);
            uint64_t row0 = myStart + done;
            uint64_t elems = rows * padS;

            // 载入 p (bf16) 与 g (half) 行段
            LocalTensor<T> p16 = inQueueP.AllocTensor<T>();
            FsgCopyRowsIn(p16, gmP[row0 * sLen], rows, sLen, padS);
            inQueueP.EnQue(p16);
            LocalTensor<half> g16 = inQueueG.AllocTensor<half>();
            FsgCopyRowsIn(g16, gmWs[row0 * padS], rows, sLen, padS);
            inQueueG.EnQue(g16);

            p16 = inQueueP.DeQue<T>();
            g16 = inQueueG.DeQue<half>();
            Cast(p32, p16, RoundMode::CAST_NONE, elems);
            Cast(g32, g16, RoundMode::CAST_NONE, elems);
            inQueueP.FreeTensor(p16);
            inQueueG.FreeTensor(g16);

            // t = p⊙g; 按行求和
            Mul(t, p32, g32, elems);
            SumParams rowParams{static_cast<uint32_t>(rows), static_cast<uint32_t>(padS), static_cast<uint32_t>(sLen)};
            Sum<float>(rs, t, sumTmp, rowParams);

            // 每行: g[r] -= rs[r] (标量域读取 + S_V 同步回向量域)
            ApplyRowBias(rs, g32, rows, padS, sLen);

            // out = p⊙(g − rs), 原地写回
            Mul(t, p32, g32, elems);
            LocalTensor<T> out16 = outQueue.AllocTensor<T>();
            Cast(out16, t, RoundMode::CAST_RINT, elems);
            outQueue.EnQue(out16);
            out16 = outQueue.DeQue<T>();
            // 写回必须走 out 指针（框架缓存路径下 in/out 指针可能不同，
            // 写 probs 侧指针会导致结果不可见）
            DataCopyPad(
                gmOut[row0 * sLen], out16,
                DataCopyExtParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(sLen * sizeof(T)), 0, 0, 0});
            outQueue.FreeTensor(out16);

            done += rows;
        }
    }

private:
    MM_TYPE& mm;
    TPipe* pipe;

    TQue<QuePosition::VECIN, 1> inQueueP;
    TQue<QuePosition::VECIN, 1> inQueueG;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> bufP32;
    TBuf<TPosition::VECCALC> bufG32;
    TBuf<TPosition::VECCALC> bufT;
    TBuf<TPosition::VECCALC> bufRs;
    TBuf<TPosition::VECCALC> bufSumTmp;

    GlobalTensor<T> gmP;
    GlobalTensor<bfloat16_t> gmGo;
    GlobalTensor<bfloat16_t> gmV;
    GlobalTensor<half> gmWs;
    GlobalTensor<T> gmOut;
    GlobalTensor<bfloat16_t> gmAPad;
    GlobalTensor<bfloat16_t> gmBPad;
    TBuf<TPosition::VECCALC> bufPad;

    uint64_t blockIdx;
    uint64_t aicNum;
    uint64_t aivNum;
    uint64_t planes;
    uint64_t qLen;
    uint64_t sLen;
    uint64_t dLen;
    uint64_t padS;
    uint64_t padD;
    uint64_t needPad;
    uint64_t padRows;
    uint64_t rowsPerIter;
    uint64_t myRows;
    uint64_t myStart;
};

} // namespace AscendC

#endif // FUSED_SOFTMAX_GRAD_IMPL_H
