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
 * FusedSoftmaxGrad kernel 入口（fused_softmax_grad）：tiling key 分发
 * MIX（bf16 对齐）/ VEC（fp32/fp16/非对齐 bf16）。
 *
 * 直接实例化 matmul::MatmulImpl（vllm-ascend 同款），不经 Matmul 别名选择逻辑，
 * 规避 SPLIT_CORE_CUBE 分支下 MatmulServiceAux CFG_NORM IterateAll 空体陷阱。
 */

#define ASCENDC_CUBE_ONLY
#include "fused_softmax_grad_impl.h"

extern "C" __global__ __aicore__ void fused_softmax_grad(GM_ADDR probs, GM_ADDR gradOutput, GM_ADDR values, GM_ADDR out,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe tPipe;
    AscendC::AscendCUtils::SetOverflow(1);
    GM_ADDR gWs = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        // fp32: 纯向量 (fixpipe C 16bit 上限, cube 路径不满足 fp32 契约)
        AscendC::FusedSoftmaxGradVec<float> op;
        op.Init(probs, gradOutput, values, out, tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        // fp16: 纯向量 (契约 rtol=5e-4 与 half-g 量化误差过近)
        AscendC::FusedSoftmaxGradVec<half> op;
        op.Init(probs, gradOutput, values, out, tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        // bf16 向量（含对齐形状：精度优先，能进 UB 就走 VEC）
        AscendC::FusedSoftmaxGradVec<bfloat16_t> op;
        op.Init(probs, gradOutput, values, out, tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        // bf16 对齐: MIX_AIC_1_2 两阶段
        // (CubeFormat/CFG_NORM 为全局作用域; MatmulType/MatmulImpl 在 AscendC, matmul 为其别名)
        using AType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
        using BType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
        using CType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
        using BiasT = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
        using MM_TYPE = matmul::MatmulImpl<AType, BType, CType, BiasT, CFG_NORM>;

        MM_TYPE mm;
        if ASCEND_IS_AIC {
            mm.SetSubBlockIdx(0);
            mm.Init(&tilingData.matmulTilingData, &tPipe);
        }
        AscendC::FusedSoftmaxGradMix<bfloat16_t, MM_TYPE> op(mm);
        op.Init(probs, gradOutput, values, out, gWs, tilingData, &tPipe);
        op.Process();
    }
}
