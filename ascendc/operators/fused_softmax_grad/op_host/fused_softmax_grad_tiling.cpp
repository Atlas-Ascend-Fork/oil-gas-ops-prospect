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
 * FusedSoftmaxGrad tiling 入口（gert）：形状/dtype 校验后委托
 * fused_softmax_grad_tiling_impl.h；切分算法本体在 *_tiling_core.h（ophost UT 覆盖）。
 */

#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "og_dtype_utils.h"
#include "fused_softmax_grad_tiling_data_def.h"
#include "fused_softmax_grad_tiling_impl.h"

namespace {

constexpr uint32_t kShapeProbs = 0;
constexpr uint32_t kShapeGo = 1;
constexpr uint32_t kShapeV = 2;

/** 形状结构校验：4D、go/v 与 probs 的 B/H 对齐、Q≠S 合法、D 一致。 */
bool ValidateFusedSoftmaxGradShapes(const gert::Shape& pShapeVal, const gert::Shape& goShapeVal,
                                    const gert::Shape& vShapeVal, uint64_t& batch, uint64_t& channel, uint64_t& q,
                                    uint64_t& s, uint64_t& dLen)
{
    if (pShapeVal.GetDimNum() != OgOpsCommon::kOgDimNum || goShapeVal.GetDimNum() != OgOpsCommon::kOgDimNum ||
        vShapeVal.GetDimNum() != OgOpsCommon::kOgDimNum) {
        return false;
    }
    batch = static_cast<uint64_t>(pShapeVal.GetDim(OgOpsCommon::kOgDim0));
    channel = static_cast<uint64_t>(pShapeVal.GetDim(OgOpsCommon::kOgDim1));
    q = static_cast<uint64_t>(pShapeVal.GetDim(OgOpsCommon::kOgDim2));
    s = static_cast<uint64_t>(pShapeVal.GetDim(OgOpsCommon::kOgDim3));
    // Q≠S 合法（openfold MSA/三角注意力形态）；4D 下 kernel 内 plane 对齐退化为标准 matmul
    if (q == 0 || s == 0) {
        return false;
    }
    if (goShapeVal.GetDim(OgOpsCommon::kOgDim0) != batch || goShapeVal.GetDim(OgOpsCommon::kOgDim1) != channel ||
        goShapeVal.GetDim(OgOpsCommon::kOgDim2) != q) {
        return false;
    }
    if (vShapeVal.GetDim(OgOpsCommon::kOgDim0) != batch || vShapeVal.GetDim(OgOpsCommon::kOgDim1) != channel ||
        vShapeVal.GetDim(OgOpsCommon::kOgDim2) != s) {
        return false;
    }
    dLen = static_cast<uint64_t>(goShapeVal.GetDim(OgOpsCommon::kOgDim3));
    return dLen != 0 && vShapeVal.GetDim(OgOpsCommon::kOgDim3) == dLen;
}

/** dtype 校验：三输入一致且为支持的类型。 */
bool ValidateFusedSoftmaxGradDtypes(gert::TilingContext* context, uint64_t& dtypeKey)
{
    const auto pDesc = context->GetInputDesc(kShapeProbs);
    const auto goDesc = context->GetInputDesc(kShapeGo);
    const auto vDesc = context->GetInputDesc(kShapeV);
    if (pDesc == nullptr || goDesc == nullptr || vDesc == nullptr) {
        return false;
    }
    if (goDesc->GetDataType() != pDesc->GetDataType() || vDesc->GetDataType() != pDesc->GetDataType()) {
        return false;
    }
    if (OgOpsCommon::GetOgDtypeSize(pDesc->GetDataType()) == 0) {
        return false;
    }
    dtypeKey = OgOpsCommon::GetOgDtypeKey(pDesc->GetDataType());
    return true;
}

/** 从 gert 上下文收集形状/dtype 并完成全部校验（tiling func 的前置段）。 */
bool CollectFusedSoftmaxGradArgs(gert::TilingContext* context, uint64_t& batch, uint64_t& channel, uint64_t& q,
                                 uint64_t& s, uint64_t& dLen, uint64_t& dtypeKey)
{
    const auto pShape = context->GetInputShape(kShapeProbs);
    const auto goShape = context->GetInputShape(kShapeGo);
    const auto vShape = context->GetInputShape(kShapeV);
    if (pShape == nullptr || goShape == nullptr || vShape == nullptr) {
        return false;
    }
    if (!ValidateFusedSoftmaxGradShapes(pShape->GetOriginShape(), goShape->GetOriginShape(), vShape->GetOriginShape(),
                                        batch, channel, q, s, dLen)) {
        return false;
    }
    return ValidateFusedSoftmaxGradDtypes(context, dtypeKey);
}
} // namespace

namespace optiling {

ge::graphStatus FusedSoftmaxGradTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t batch = 0;
    uint64_t channel = 0;
    uint64_t q = 0;
    uint64_t s = 0;
    uint64_t dLen = 0;
    uint64_t dtypeKey = 0;
    if (!CollectFusedSoftmaxGradArgs(context, batch, channel, q, s, dLen, dtypeKey)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    FusedSoftmaxGradTilingData tilingData;
    if (!FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTiling(batch, channel, q, s, dLen, dtypeKey, platform, tilingData,
                                                             plan)) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(plan.blockDim);
    context->SetTilingKey(plan.tilingKey);

    // workspace：系统预留（含 SyncAll 计数器区）+ MIX 路径 g 缓冲（half）
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = 16ULL * 1024 * 1024 + plan.gBytes;

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
