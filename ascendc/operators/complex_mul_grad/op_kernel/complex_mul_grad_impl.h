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
 * ComplexMulGrad：
 *   gur =  gwr*vr + gwi*vi
 *   gui = -gwr*vi + gwi*vr
 *   gvr =  gwr*ur + gwi*ui
 *   gvi = -gwr*ui + gwi*ur
 *
 * inQue：gwr|gwi|ur|ui|vr|vi
 * outQue：gur|gui|gvr|gvi
 *
 * 核间切分 / tile 循环 / DataCopyPad 样板见 og_elementwise_kernel.h。
 */

#ifndef COMPLEX_MUL_GRAD_IMPL_H
#define COMPLEX_MUL_GRAD_IMPL_H

#include "og_elementwise_kernel.h"

#include "complex_mul_grad_tiling.h"

namespace ComplexMulGrad {

using namespace AscendC;

constexpr int32_t IN_SLOTS = 6;  // gwr, gwi, ur, ui, vr, vi
constexpr int32_t OUT_SLOTS = 4; // gur, gui, gvr, gvi

class ComplexMulGradKernel : public OgOpsKernel::ElementwiseTileKernel<ComplexMulGradKernel> {
public:
    __aicore__ inline ComplexMulGradKernel() {}

    __aicore__ inline void Init(GM_ADDR gwr, GM_ADDR gwi, GM_ADDR ur, GM_ADDR ui, GM_ADDR vr, GM_ADDR vi, GM_ADDR gur,
                                GM_ADDR gui, GM_ADDR gvr, GM_ADDR gvi, const ComplexMulGradTilingData& tiling,
                                TPipe* pipe)
    {
        const int64_t numel = tiling.totalElems;
        gwrGm_.SetGlobalBuffer((__gm__ float*)gwr, numel);
        gwiGm_.SetGlobalBuffer((__gm__ float*)gwi, numel);
        urGm_.SetGlobalBuffer((__gm__ float*)ur, numel);
        uiGm_.SetGlobalBuffer((__gm__ float*)ui, numel);
        vrGm_.SetGlobalBuffer((__gm__ float*)vr, numel);
        viGm_.SetGlobalBuffer((__gm__ float*)vi, numel);
        // 与前向相同：TBE 会丢掉后几路输出指针，四路写进 gur 起的连续缓冲。
        (void)gui;
        (void)gvr;
        (void)gvi;
        gurGm_.SetGlobalBuffer((__gm__ float*)gur, numel * 4);
        guiGm_.SetGlobalBuffer((__gm__ float*)gur + numel, numel);
        gvrGm_.SetGlobalBuffer((__gm__ float*)gur + numel * 2, numel);
        gviGm_.SetGlobalBuffer((__gm__ float*)gur + numel * 3, numel);

        InitTiles(numel, tiling.tileFloats, tiling.coreNum, IN_SLOTS, OUT_SLOTS, pipe);
    }

    __aicore__ inline LocalTensor<float> LoadTile(int64_t gmOff, int32_t cur)
    {
        const uint32_t t = SlotStride();
        LocalTensor<float> in = inQue_.AllocTensor<float>();
        CopyInPad(in[0], gwrGm_[gmOff], cur);
        CopyInPad(in[t], gwiGm_[gmOff], cur);
        CopyInPad(in[2 * t], urGm_[gmOff], cur);
        CopyInPad(in[3 * t], uiGm_[gmOff], cur);
        CopyInPad(in[4 * t], vrGm_[gmOff], cur);
        CopyInPad(in[5 * t], viGm_[gmOff], cur);
        return EnqueueIn(in);
    }

    __aicore__ inline void ProcessTile(int64_t gmOff, int32_t cur, LocalTensor<float> tmp)
    {
        LocalTensor<float> in = LoadTile(gmOff, cur);
        const uint32_t t = SlotStride();
        LocalTensor<float> gwr = in[0];
        LocalTensor<float> gwi = in[t];
        LocalTensor<float> ur = in[2 * t];
        LocalTensor<float> ui = in[3 * t];
        LocalTensor<float> vr = in[4 * t];
        LocalTensor<float> vi = in[5 * t];

        StoreOne(gurGm_[gmOff], tmp, gwr, vr, gwi, vi, cur, false);
        StoreOne(guiGm_[gmOff], tmp, gwr, vi, gwi, vr, cur, true);
        StoreOne(gvrGm_[gmOff], tmp, gwr, ur, gwi, ui, cur, false);
        StoreOne(gviGm_[gmOff], tmp, gwr, ui, gwi, ur, cur, true);
        inQue_.FreeTensor(in);
    }

private:
    /** dstGm = ±(a0*a1) + (b0*b1)；每路单独 EnQue/DeQue/CopyOut。 */
    __aicore__ inline void StoreOne(GlobalTensor<float> dstGm, LocalTensor<float> tmp, LocalTensor<float> a0,
                                    LocalTensor<float> a1, LocalTensor<float> b0, LocalTensor<float> b1, int32_t cur,
                                    bool negA)
    {
        const int32_t n = static_cast<int32_t>((static_cast<uint32_t>(cur) + 7u) & ~7u);
        LocalTensor<float> out = outQue_.AllocTensor<float>();
        Mul(out, a0, a1, n);
        if (negA) {
            PipeBarrier<PIPE_V>();
            Muls(out, out, -1.0f, n);
        }
        Mul(tmp, b0, b1, n);
        PipeBarrier<PIPE_V>();
        Add(out, out, tmp, n);
        outQue_.EnQue(out);
        out = outQue_.DeQue<float>();
        CopyOutPad(dstGm, out, cur);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        outQue_.FreeTensor(out);
    }

    GlobalTensor<float> gwrGm_, gwiGm_, urGm_, uiGm_, vrGm_, viGm_;
    GlobalTensor<float> gurGm_, guiGm_, gvrGm_, gviGm_;
};

} // namespace ComplexMulGrad

#endif // COMPLEX_MUL_GRAD_IMPL_H
