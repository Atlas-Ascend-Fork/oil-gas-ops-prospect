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
 * FusedSoftmaxGrad 切分：platform 读取与高阶 Matmul tiling（MIX 路径）在此桥接，
 * 平台无关的切分算法在 fused_softmax_grad_tiling_core.h（ophost UT 覆盖）。
 */

#ifndef FUSED_SOFTMAX_GRAD_TILING_IMPL_H
#define FUSED_SOFTMAX_GRAD_TILING_IMPL_H

#include <cstdint>

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "fused_softmax_grad_tiling_core.h"
#include "fused_softmax_grad_tiling_data_def.h"

namespace FusedSoftmaxGradHost {

/** MIX 路径 MatmulApiTiling：g_p = go_p @ V_p^T，A=[Q,D] bf16 ND，B=[S,D] bf16 ND(transB)，C=[Q,S] ND half。 */
inline bool ComputeMatmulTiling(platform_ascendc::PlatformAscendC& platform, uint64_t q, uint64_t s, uint64_t dLen,
                                optiling::TCubeTiling& mmTilingOut)
{
    matmul_tiling::MatmulApiTiling mmApi(platform);
    mmApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16,
                   false);
    mmApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, true);
    mmApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    mmApi.SetBias(false);
    mmApi.SetOrgShape(q, s, dLen);
    mmApi.SetShape(q, s, dLen);
    mmApi.SetBufferSpace(-1, -1, -1);
    return mmApi.GetTiling(mmTilingOut) != -1;
}

/** platform 读取并委托 core；结果写入 tiling 数据结构。 */
inline bool ComputeFusedSoftmaxGradTiling(uint64_t batch, uint64_t channel, uint64_t q, uint64_t s, uint64_t dLen,
                                          uint64_t dtypeKey, platform_ascendc::PlatformAscendC& platform,
                                          optiling::FusedSoftmaxGradTilingData& tilingData, FusedSoftmaxGradPlan& plan)
{
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        return false;
    }

    FusedSoftmaxGradCoreParams params;
    if (!ComputeFusedSoftmaxGradTilingCore(batch, channel, q, s, dLen, dtypeKey, ubSize, params, plan)) {
        return false;
    }
    plan.blockDim = 24; // 统一 MIX_AIC_1_2：24 AIC + 48 AIV（向量 key 下 AIC 空转）

    if (params.useMix == 1) {
        if (!ComputeMatmulTiling(platform, q, s, dLen, tilingData.matmulTilingData)) {
            return false;
        }
    }

    tilingData.set_coreNum(24); // MIX_AIC_1_2：blockDim = AIC 数，AIV = 2×blockDim
    tilingData.set_batch(batch);
    tilingData.set_channel(channel);
    tilingData.set_height(q);
    tilingData.set_width(s);
    tilingData.set_dLen(dLen);
    tilingData.set_planes(batch * channel);
    tilingData.set_useMix(params.useMix);
    tilingData.set_padS(params.padS);
    tilingData.set_padD(params.padD);
    tilingData.set_rowsPerIter(params.rowsPerIter);
    return true;
}

} // namespace FusedSoftmaxGradHost

#endif // FUSED_SOFTMAX_GRAD_TILING_IMPL_H
