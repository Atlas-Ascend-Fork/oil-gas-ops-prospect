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
 * ComplexMulGrad — ComplexMul 反向（AIV-only，方案 C double buffer）。
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

    // TBE 预编译只接受数值字面量；用命名常量会跳过本分支，kernel 空跑导致精度崩。
    if (TILING_KEY_IS(0)) {
        TPipe pipe;
        ComplexMulGrad::ComplexMulGradKernel op;
        op.Init(gwr, gwi, ur, ui, vr, vi, gur, gui, gvr, gvi, tilingData, &pipe);
        op.Process();
    }
}
