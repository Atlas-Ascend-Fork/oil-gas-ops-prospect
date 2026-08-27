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
 * ComplexMulOg AIV kernel（方案 C）。
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
                                const ComplexMulTilingData &tiling, TPipe *pipe)
    {
        const int64_t numel = tiling.totalElems;
        urGm_.SetGlobalBuffer((__gm__ float *)ur, numel);
        uiGm_.SetGlobalBuffer((__gm__ float *)ui, numel);
        vrGm_.SetGlobalBuffer((__gm__ float *)vr, numel);
        viGm_.SetGlobalBuffer((__gm__ float *)vi, numel);
        wrGm_.SetGlobalBuffer((__gm__ float *)wr, numel);
        wiGm_.SetGlobalBuffer((__gm__ float *)wi, numel);

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
        inQue_.EnQue(in);
        return inQue_.DeQue<float>();
    }

    __aicore__ inline void ComputeTile(LocalTensor<float> in, LocalTensor<float> out, LocalTensor<float> tmp,
                                       int32_t cur)
    {
        const uint32_t t = SlotStride();
        LocalTensor<float> ur = in[0];
        LocalTensor<float> ui = in[t];
        LocalTensor<float> vr = in[2 * t];
        LocalTensor<float> vi = in[3 * t];
        LocalTensor<float> wr = out[0];
        LocalTensor<float> wi = out[t];

        // wr = ur*vr - ui*vi
        Mul(wr, ur, vr, cur);
        Mul(tmp, ui, vi, cur);
        Sub(wr, wr, tmp, cur);
        // wi = ur*vi + ui*vr
        Mul(wi, ur, vi, cur);
        Mul(tmp, ui, vr, cur);
        Add(wi, wi, tmp, cur);
    }

    __aicore__ inline void StoreTile(int64_t gmOff, LocalTensor<float> out, int32_t cur)
    {
        const uint32_t t = SlotStride();
        CopyOutPad(wrGm_[gmOff], out[0], cur);
        CopyOutPad(wiGm_[gmOff], out[t], cur);
    }

private:
    GlobalTensor<float> urGm_, uiGm_, vrGm_, viGm_, wrGm_, wiGm_;
};

} // namespace ComplexMul

#endif // COMPLEX_MUL_IMPL_H
