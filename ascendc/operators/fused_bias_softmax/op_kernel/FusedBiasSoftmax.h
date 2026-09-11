/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * FusedBiasSoftmaxOg kernel 实现：bias*2 + in-place 行 softmax 融合（AIV）。
 *
 * 源自 CANN scaled_masked_softmax_v2 生产模板改造（tiling/pipeline/DataCopyPad/
 * SoftMax 高阶 api 保留），验证与性能数据见 PR 测试报告。
 */
#ifndef FUSED_BIAS_SOFTMAX_IMPL_H
#define FUSED_BIAS_SOFTMAX_IMPL_H


#include "kernel_operator.h"

namespace AscendC {

template<typename T1, typename T2>
__aicore__ inline T1 FbsCeilDiv(T1 a, T2 b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

template<typename T>
class FusedBiasSoftmax {
public:
    __aicore__ inline FusedBiasSoftmax()
    {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias1, GM_ADDR bias2, GM_ADDR y,
                                const FusedBiasSoftmaxTilingData& tilingData)
                                {
        this->blockIdx = GetBlockIdx();
        this->coreOffset = (this->blockIdx >= tilingData.headCoreNum ? tilingData.headCoreNum : this->blockIdx)
                        * (tilingData.lineHeadCore * tilingData.width)
                        + (this->blockIdx >= tilingData.headCoreNum ? this->blockIdx - tilingData.headCoreNum : 0)
                        * (tilingData.lineTailCore * tilingData.width);
        this->startLineOffset = this->coreOffset / tilingData.width;
        gmX.SetGlobalBuffer((__gm__ T*)x + this->coreOffset);
        gmY.SetGlobalBuffer((__gm__ T*)y + this->coreOffset);

        uint64_t bufSize = tilingData.padLineNum * tilingData.lineHeadIter;
        pipe.InitBuffer(inQueueX, 1, bufSize * sizeof(T));
        pipe.InitBuffer(outQueueY, 1, bufSize * sizeof(T));

        if constexpr (!std::is_same<T, float>::value) {
            pipe.InitBuffer(bufX, bufSize * sizeof(float));
        }
        const uint64_t tmpBufSize = 32 * 1024; // 高阶api复用空间大小 32K
        pipe.InitBuffer(bufShare, tmpBufSize);

        this->hasB1 = tilingData.hasB1;
        this->hasB2 = tilingData.hasB2;
        this->b1Full = tilingData.b1Full;
        this->qPerPlane = tilingData.qPerPlane;
        this->tilingData = tilingData;

        if (this->hasB2) {
            pipe.InitBuffer(inQueueB2, 1, bufSize * sizeof(T));
            if constexpr (!std::is_same<T, float>::value) {
                pipe.InitBuffer(bufB2F32, bufSize * sizeof(float));
            }
            gmB2.SetGlobalBuffer((__gm__ T*)bias2);
        }
        if (this->hasB1) {
            if (this->b1Full) {
                pipe.InitBuffer(inQueueB1, 1, bufSize * sizeof(T));
                if constexpr (!std::is_same<T, float>::value) {
                    pipe.InitBuffer(bufB1F32, bufSize * sizeof(float));
                }
            } else {
                // 广播形态: 每次迭代只需 ceil(lineHeadIter/qPerPlane)+1 行
                this->b1RowsMax = FbsCeilDiv(tilingData.lineHeadIter, this->qPerPlane) + 1;
                pipe.InitBuffer(inQueueB1, 1, this->b1RowsMax * tilingData.padLineNum * sizeof(T));
                if constexpr (!std::is_same<T, float>::value) {
                    pipe.InitBuffer(bufBRow, tilingData.padLineNum * sizeof(float));
                }
            }
            gmB1.SetGlobalBuffer((__gm__ T*)bias1);
        }

        this->linesPerIter = tilingData.lineHeadIter;
        this->linesLastIter = this->blockIdx >= tilingData.headCoreNum ? tilingData.lineLastTailIter
                              : tilingData.lineLastHeadIter;
        this->width = tilingData.width;
        this->padLineNum = tilingData.padLineNum;
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= tilingData.coreNum) {
            return;
        }

        uint64_t loop = this->blockIdx >= tilingData.headCoreNum ? tilingData.iterTailCore : tilingData.iterHeadCore;
        uint64_t linePerIter = this->linesPerIter;
        for (uint64_t idx = 0u; idx < loop; ++idx) {
            // tail iter
            if (idx == loop - 1) {
                linePerIter = this->linesLastIter;
            }
            this->elePerIter = tilingData.padLineNum * linePerIter;
            CopyIn(idx, linePerIter);
            Compute(idx, linePerIter);
            CopyOut(idx, linePerIter);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t idx, uint64_t linePerIter)
    {
        uint64_t curLine = this->startLineOffset + idx * this->linesPerIter;
        // gmX/gmY 基址已含 coreOffset, x 按迭代号偏移; bias 基址为原始 GM, 按绝对行偏移
        CopyXIn(idx, linePerIter);
        if (this->hasB2) {
            CopyBiasRows(inQueueB2, gmB2, curLine, linePerIter);
        }
        if (this->hasB1) {
            if (this->b1Full) {
                CopyBiasRows(inQueueB1, gmB1, curLine, linePerIter);
            } else {
                // 广播: 载入本迭代覆盖的 distinct plane 行 [p0, p1]
                uint64_t p0 = curLine / this->qPerPlane;
                uint64_t p1 = (curLine + linePerIter - 1) / this->qPerPlane;
                CopyBiasRows(inQueueB1, gmB1, p0, p1 - p0 + 1);
            }
        }
    }

    __aicore__ inline void CopyXIn(uint64_t idx, uint64_t linePerIter)
    {
        LocalTensor<T> xTensor = inQueueX.AllocTensor<T>();
        DataCopyExtParams params = {
            static_cast<uint16_t>(linePerIter),
            static_cast<uint32_t>(this->width * sizeof(T)),
            0, 0, 0
        };
        DataCopyPadExtParams<T> extParams = {
            true, 0, static_cast<uint8_t>(tilingData.paddingNum), 0.0
        };
        DataCopyPad(xTensor, gmX[idx * this->linesPerIter * this->width], params, extParams);
        inQueueX.EnQue(xTensor);
    }

    // 多行批量载入 bias 行, 每行落位 padLineNum 对齐(rightPadding 补齐), 与 x 布局一致
    __aicore__ inline void CopyBiasRows(TQue<QuePosition::VECIN, 1>& queue, GlobalTensor<T>& gm,
                                        uint64_t firstRow, uint64_t rows)
                                        {
        LocalTensor<T> bTensor = queue.AllocTensor<T>();
        DataCopyExtParams params = {
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(this->width * sizeof(T)),
            0, 0, 0
        };
        DataCopyPadExtParams<T> extParams = {
            true, 0, static_cast<uint8_t>(tilingData.paddingNum), 0.0
        };
        DataCopyPad(bTensor, gm[firstRow * this->width], params, extParams);
        queue.EnQue(bTensor);
    }

    __aicore__ inline void Compute(uint64_t idx, uint64_t linePerIter)
    {
        uint64_t curLine = this->startLineOffset + idx * this->linesPerIter;
        LocalTensor<T> xTensor = inQueueX.DeQue<T>();
        sharedBuffer = bufShare.Get<uint8_t>();

        if constexpr (!std::is_same<T, float>::value) {
            xF32 = bufX.Get<float>();
            ChunkedCast(xF32, xTensor, RoundMode::CAST_NONE, this->elePerIter);
        }

        if (this->hasB2) {
            LocalTensor<T> b2Tensor = inQueueB2.DeQue<T>();
            AddBiasFullLine(xTensor, b2Tensor, bufB2F32, linePerIter);
            inQueueB2.FreeTensor(b2Tensor);
        }
        if (this->hasB1) {
            LocalTensor<T> b1Tensor = inQueueB1.DeQue<T>();
            if (this->b1Full) {
                AddBiasFullLine(xTensor, b1Tensor, bufB1F32, linePerIter);
            } else {
                AddBiasBcast(xTensor, b1Tensor, curLine, linePerIter);
            }
            inQueueB1.FreeTensor(b1Tensor);
        }

        LocalTensor<T> yTensor = outQueueY.AllocTensor<T>();
        if constexpr (!std::is_same<T, float>::value) {
            SoftmaxX(xF32, xF32, sharedBuffer, linePerIter);
            ChunkedCast(yTensor, xF32, RoundMode::CAST_RINT, this->elePerIter);
        } else {
            SoftmaxX(yTensor, xTensor, sharedBuffer, linePerIter);
        }

        inQueueX.FreeTensor(xTensor);
        outQueueY.EnQue(yTensor);
    }

    // stride-1 全尺寸 bias: 布局与 x 一致, 整块 Cast(fp32) 后分块 Add
    __aicore__ inline void AddBiasFullLine(LocalTensor<T>& xT, LocalTensor<T>& bTensor,
                                           TBuf<TPosition::VECCALC>& bF32Buf, uint64_t linePerIter)
    {
        if constexpr (std::is_same<T, float>::value) {
            ChunkedAdd(xT, bTensor, this->elePerIter);
        } else {
            LocalTensor<float> bF32 = bF32Buf.Get<float>();
            ChunkedCast(bF32, bTensor, RoundMode::CAST_NONE, this->elePerIter);
            ChunkedAdd(xF32, bF32, this->elePerIter);
        }
    }

    // 广播 bias: 每行叠加所属 plane 的 bias 行 (fp32 域)
    __aicore__ inline void AddBiasBcast(LocalTensor<T>& xT, LocalTensor<T>& b1Tensor,
                                        uint64_t curLine, uint64_t linePerIter)
    {
        uint64_t p0 = curLine / this->qPerPlane;
        if constexpr (std::is_same<T, float>::value) {
            for (uint64_t l = 0u; l < linePerIter; ++l) {
                uint64_t row = (curLine + l) / this->qPerPlane - p0;
                uint64_t off = l * this->padLineNum;
                Add(xT[off], xT[off], b1Tensor[row * this->padLineNum], this->padLineNum);
            }
        } else {
            LocalTensor<float> bRow = bufBRow.Get<float>();
            for (uint64_t l = 0u; l < linePerIter; ++l) {
                uint64_t row = (curLine + l) / this->qPerPlane - p0;
                uint64_t off = l * this->padLineNum;
                Cast(bRow, b1Tensor[row * this->padLineNum], RoundMode::CAST_NONE, this->padLineNum);
                Add(xF32[off], xF32[off], bRow, this->padLineNum);
            }
        }
    }

    template<typename DST, typename SRC>
    __aicore__ inline void ChunkedCast(DST& dst, SRC& src, RoundMode mode, uint64_t count)
    {
        constexpr uint64_t CHUNK = 16384; // Level-2 算子 repeatTimes 上限保护
        for (uint64_t off = 0u; off < count; off += CHUNK) {
            uint64_t n = (count - off) < CHUNK ? (count - off) : CHUNK;
            Cast(dst[off], src[off], mode, n);
        }
    }

    template<typename D1, typename D2>
    __aicore__ inline void ChunkedAdd(D1& dst, D2& src2, uint64_t count)
    {
        constexpr uint64_t CHUNK = 16384;
        for (uint64_t off = 0u; off < count; off += CHUNK) {
            uint64_t n = (count - off) < CHUNK ? (count - off) : CHUNK;
            Add(dst[off], dst[off], src2[off], n);
        }
    }

    __aicore__ inline void SoftmaxX(LocalTensor<float>& dstTensor, LocalTensor<float>& srcTensor,
                                    LocalTensor<uint8_t> sharedBuffer, uint64_t lines)
                                    {
        SoftMaxTiling softmaxTilingData = tilingData.softmaxTilingData;
        SoftMaxShapeInfo softmaxShapeInfoData = {
            static_cast<uint32_t>(lines),
            static_cast<uint32_t>(tilingData.padLineNum),
            static_cast<uint32_t>(lines),
            static_cast<uint32_t>(tilingData.width),
        };

        SoftMax<float, false, false>(dstTensor, srcTensor, sharedBuffer, softmaxTilingData, softmaxShapeInfoData);
    }

    __aicore__ inline void CopyOut(uint64_t idx, uint64_t linePerIter)
    {
        LocalTensor<T> yTensor = outQueueY.DeQue<T>();
        DataCopyExtParams params = {
            static_cast<uint16_t>(linePerIter),
            static_cast<uint32_t>(this->width * sizeof(T)),
            0, 0, 0
        };
        DataCopyPad(gmY[idx * this->linesPerIter * this->width], yTensor, params);
        outQueueY.FreeTensor(yTensor);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueB1;
    TQue<QuePosition::VECIN, 1> inQueueB2;
    TQue<QuePosition::VECOUT, 1> outQueueY;

    GlobalTensor<T> gmX;
    GlobalTensor<T> gmY;
    GlobalTensor<T> gmB1;
    GlobalTensor<T> gmB2;

    LocalTensor<float> xF32;
    LocalTensor<uint8_t> sharedBuffer;

    TBuf<TPosition::VECCALC> bufX;
    TBuf<TPosition::VECCALC> bufB1F32;
    TBuf<TPosition::VECCALC> bufB2F32;
    TBuf<TPosition::VECCALC> bufBRow;
    TBuf<TPosition::VECCALC> bufShare;

    FusedBiasSoftmaxTilingData tilingData;

    uint64_t hasB1;
    uint64_t hasB2;
    uint64_t b1Full;
    uint64_t qPerPlane;
    uint64_t b1RowsMax;
    uint64_t blockIdx;
    uint64_t coreOffset;
    uint64_t startLineOffset;
    uint64_t linesPerIter;
    uint64_t linesLastIter;
    uint64_t elePerIter;
    uint64_t width;
    uint64_t padLineNum;
};
} // namespace AscendC

#endif // FUSED_BIAS_SOFTMAX_IMPL_H
