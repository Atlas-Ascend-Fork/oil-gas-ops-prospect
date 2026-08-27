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
 * ComplexMulGradOg — ComplexMul 反向（AIV-only，方案 C double buffer）。
 */

#include "kernel_operator.h"
#include "og_elementwise_kernel.h"
#include "complex_mul_grad_tiling.h"
#include "complex_mul_grad_impl.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void complex_mul_grad(GM_ADDR gwr, GM_ADDR gwi, GM_ADDR ur, GM_ADDR ui, GM_ADDR vr,
                                                       GM_ADDR vi, GM_ADDR gur, GM_ADDR gui, GM_ADDR gvr, GM_ADDR gvi,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;

    ComplexMulGradTilingData tilingData;
    OgOpsKernel::CopyTilingData(&tilingData, tiling);

    if (TILING_KEY_IS(COMPLEX_MUL_GRAD_TILING_KEY)) {
        TPipe pipe;
        ComplexMulGrad::ComplexMulGradKernel op;
        op.Init(gwr, gwi, ur, ui, vr, vi, gur, gui, gvr, gvi, tilingData, &pipe);
        op.Process();
    }
}
