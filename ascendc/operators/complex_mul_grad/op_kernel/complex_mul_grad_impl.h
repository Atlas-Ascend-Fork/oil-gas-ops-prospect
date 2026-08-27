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
 * ComplexMulGradOg：
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
                                GM_ADDR gui, GM_ADDR gvr, GM_ADDR gvi, const ComplexMulGradTilingData &tiling,
                                TPipe *pipe)
    {
        const int64_t numel = tiling.totalElems;
        gwrGm_.SetGlobalBuffer((__gm__ float *)gwr, numel);
        gwiGm_.SetGlobalBuffer((__gm__ float *)gwi, numel);
        urGm_.SetGlobalBuffer((__gm__ float *)ur, numel);
        uiGm_.SetGlobalBuffer((__gm__ float *)ui, numel);
        vrGm_.SetGlobalBuffer((__gm__ float *)vr, numel);
        viGm_.SetGlobalBuffer((__gm__ float *)vi, numel);
        gurGm_.SetGlobalBuffer((__gm__ float *)gur, numel);
        guiGm_.SetGlobalBuffer((__gm__ float *)gui, numel);
        gvrGm_.SetGlobalBuffer((__gm__ float *)gvr, numel);
        gviGm_.SetGlobalBuffer((__gm__ float *)gvi, numel);

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
        inQue_.EnQue(in);
        return inQue_.DeQue<float>();
    }

    __aicore__ inline void ComputeTile(LocalTensor<float> in, LocalTensor<float> out, LocalTensor<float> tmp,
                                       int32_t cur)
    {
        const uint32_t t = SlotStride();
        LocalTensor<float> gwr = in[0];
        LocalTensor<float> gwi = in[t];
        LocalTensor<float> ur = in[2 * t];
        LocalTensor<float> ui = in[3 * t];
        LocalTensor<float> vr = in[4 * t];
        LocalTensor<float> vi = in[5 * t];

        // gur = gwr*vr + gwi*vi
        FusedMulAdd(out[0], tmp, gwr, vr, gwi, vi, cur);
        // gui = -gwr*vi + gwi*vr
        FusedMulSub(out[t], tmp, gwr, vi, gwi, vr, cur);
        // gvr = gwr*ur + gwi*ui
        FusedMulAdd(out[2 * t], tmp, gwr, ur, gwi, ui, cur);
        // gvi = -gwr*ui + gwi*ur
        FusedMulSub(out[3 * t], tmp, gwr, ui, gwi, ur, cur);
    }

    __aicore__ inline void StoreTile(int64_t gmOff, LocalTensor<float> out, int32_t cur)
    {
        const uint32_t t = SlotStride();
        CopyOutPad(gurGm_[gmOff], out[0], cur);
        CopyOutPad(guiGm_[gmOff], out[t], cur);
        CopyOutPad(gvrGm_[gmOff], out[2 * t], cur);
        CopyOutPad(gviGm_[gmOff], out[3 * t], cur);
    }

private:
    /** dst = a0*a1 + b0*b1 */
    __aicore__ inline void FusedMulAdd(LocalTensor<float> dst, LocalTensor<float> tmp, LocalTensor<float> a0,
                                       LocalTensor<float> a1, LocalTensor<float> b0, LocalTensor<float> b1, int32_t cur)
    {
        Mul(dst, a0, a1, cur);
        Mul(tmp, b0, b1, cur);
        Add(dst, dst, tmp, cur);
    }

    /** dst = -a0*a1 + b0*b1 */
    __aicore__ inline void FusedMulSub(LocalTensor<float> dst, LocalTensor<float> tmp, LocalTensor<float> a0,
                                       LocalTensor<float> a1, LocalTensor<float> b0, LocalTensor<float> b1, int32_t cur)
    {
        Mul(dst, a0, a1, cur);
        Muls(dst, dst, -1.0f, cur);
        Mul(tmp, b0, b1, cur);
        Add(dst, dst, tmp, cur);
    }

    GlobalTensor<float> gwrGm_, gwiGm_, urGm_, uiGm_, vrGm_, viGm_;
    GlobalTensor<float> gurGm_, guiGm_, gvrGm_, gviGm_;
};

} // namespace ComplexMulGrad

#endif // COMPLEX_MUL_GRAD_IMPL_H
