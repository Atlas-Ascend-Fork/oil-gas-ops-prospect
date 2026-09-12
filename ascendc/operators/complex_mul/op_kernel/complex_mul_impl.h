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
 * ComplexMul AIV kernel（方案 C）。
 *
 * inQue 每槽：ur|ui|vr|vi（各 tileFloats）
 * outQue 每槽：wr|wi
 * tmpBuf：一路中间乘积
 *
 * 核间切分 / tile 循环 / DataCopyPad 样板见 og_elementwise_kernel.h，
 * 这里只描述"搬哪几路、算什么"。
 */

#ifndef COMPLEX_MUL_IMPL_H
#define COMPLEX_MUL_IMPL_H

#include "og_elementwise_kernel.h"

#include "complex_mul_tiling.h"

namespace ComplexMul {

using namespace AscendC;

constexpr int32_t IN_SLOTS = 4;  // ur, ui, vr, vi
constexpr int32_t OUT_SLOTS = 2; // wr, wi

class ComplexMulKernel : public OgOpsKernel::ElementwiseTileKernel<ComplexMulKernel> {
public:
    __aicore__ inline ComplexMulKernel() {}

    __aicore__ inline void Init(GM_ADDR ur, GM_ADDR ui, GM_ADDR vr, GM_ADDR vi, GM_ADDR wr, GM_ADDR wi,
                                const ComplexMulTilingData& tiling, TPipe* pipe)
    {
        const int64_t numel = tiling.totalElems;
        urGm_.SetGlobalBuffer((__gm__ float*)ur, numel);
        uiGm_.SetGlobalBuffer((__gm__ float*)ui, numel);
        vrGm_.SetGlobalBuffer((__gm__ float*)vr, numel);
        viGm_.SetGlobalBuffer((__gm__ float*)vi, numel);
        wrGm_.SetGlobalBuffer((__gm__ float*)wr, numel);
        wiGm_.SetGlobalBuffer((__gm__ float*)wi, numel);

        InitTiles(numel, tiling.tileFloats, tiling.coreNum, IN_SLOTS, OUT_SLOTS, pipe);
    }

    __aicore__ inline LocalTensor<float> LoadTile(int64_t gmOff, int32_t cur)
    {
        const uint32_t t = SlotStride();
        LocalTensor<float> in = inQue_.AllocTensor<float>();
        CopyInPad(in[0], urGm_[gmOff], cur);
        CopyInPad(in[t], uiGm_[gmOff], cur);
        CopyInPad(in[2 * t], vrGm_[gmOff], cur);
        CopyInPad(in[3 * t], viGm_[gmOff], cur);
        return EnqueueIn(in);
    }

    // 每路单独走完 EnQue/DeQue/CopyOut：同一拍里算两路或中途 CopyOut
    // 会在 910B2 + TBE 上丢掉第二路（wi 变成 ur*vi）。
    __aicore__ inline void ProcessTile(int64_t gmOff, int32_t cur, LocalTensor<float> tmp)
    {
        const int32_t n = static_cast<int32_t>((static_cast<uint32_t>(cur) + 7u) & ~7u);
        LocalTensor<float> in = LoadTile(gmOff, cur);
        const uint32_t t = SlotStride();
        LocalTensor<float> ur = in[0];
        LocalTensor<float> ui = in[t];
        LocalTensor<float> vr = in[2 * t];
        LocalTensor<float> vi = in[3 * t];

        LocalTensor<float> out = outQue_.AllocTensor<float>();
        Mul(out, ur, vr, n);
        Mul(tmp, ui, vi, n);
        PipeBarrier<PIPE_V>();
        Sub(out, out, tmp, n);
        outQue_.EnQue(out);
        out = outQue_.DeQue<float>();
        CopyOutPad(wrGm_[gmOff], out, cur);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        outQue_.FreeTensor(out);

        out = outQue_.AllocTensor<float>();
        Mul(out, ur, vi, n);
        Mul(tmp, ui, vr, n);
        PipeBarrier<PIPE_V>();
        Add(out, out, tmp, n);
        inQue_.FreeTensor(in);
        outQue_.EnQue(out);
        out = outQue_.DeQue<float>();
        CopyOutPad(wiGm_[gmOff], out, cur);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        outQue_.FreeTensor(out);
    }

private:
    GlobalTensor<float> urGm_, uiGm_, vrGm_, viGm_, wrGm_, wiGm_;
};

} // namespace ComplexMul

#endif // COMPLEX_MUL_IMPL_H
