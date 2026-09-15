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
 * FusedBiasSoftmax tiling 入口（gert）：形状/dtype/bias 校验后委托
 * fused_bias_softmax_tiling_impl.h；切分算法本体在 *_tiling_core.h（ophost UT 覆盖）。
 */

#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "og_dtype_utils.h"
#include "fused_bias_softmax_tiling_data_def.h"
#include "fused_bias_softmax_tiling_impl.h"

namespace {

constexpr uint32_t kShapeInput = 0;
constexpr uint32_t kBias1Index = 1;
constexpr uint32_t kBias2Index = 2;

/** bias 形状校验：4D、与 x 的 B/H/S 对齐；bias1 的 dim2 为 1（广播）或 Q（全尺寸）。 */
bool CheckBiasValid(gert::TilingContext* context, uint32_t idx, const gert::Shape& xShape, uint64_t b1FullRequired)
{
    const auto desc = context->GetOptionalInputDesc(idx);
    const auto shape = context->GetOptionalInputShape(idx);
    if (desc == nullptr || shape == nullptr) {
        return true; // 不存在，合法
    }
    const auto& shapeVal = shape->GetOriginShape();
    if (shapeVal.GetDimNum() != OgOpsCommon::kOgDimNum || shapeVal.GetShapeSize() == 0) {
        return false;
    }
    if (!FusedBiasSoftmaxHost::BiasShapeValid(static_cast<int64_t>(xShape.GetDim(OgOpsCommon::kOgDim0)),
                                              static_cast<int64_t>(xShape.GetDim(OgOpsCommon::kOgDim1)),
                                              static_cast<int64_t>(xShape.GetDim(OgOpsCommon::kOgDim2)),
                                              static_cast<int64_t>(xShape.GetDim(OgOpsCommon::kOgDim3)),
                                              static_cast<int64_t>(shapeVal.GetDim(OgOpsCommon::kOgDim0)),
                                              static_cast<int64_t>(shapeVal.GetDim(OgOpsCommon::kOgDim1)),
                                              static_cast<int64_t>(shapeVal.GetDim(OgOpsCommon::kOgDim2)),
                                              static_cast<int64_t>(shapeVal.GetDim(OgOpsCommon::kOgDim3)),
                                              b1FullRequired == 1)) {
        return false;
    }
    return desc->GetDataType() == context->GetInputDesc(kShapeInput)->GetDataType();
}

/** 从 gert 上下文收集形状/dtype/bias 信息并完成全部校验（tiling func 的前置段）。 */
bool CollectFusedBiasSoftmaxShape(gert::TilingContext* context, FusedBiasSoftmaxHost::FusedBiasSoftmaxShape& shape)
{
    const auto xShape = context->GetInputShape(kShapeInput);
    const auto xDesc = context->GetInputDesc(kShapeInput);
    if (xShape == nullptr || xDesc == nullptr) {
        return false;
    }
    const auto& xShapeVal = xShape->GetOriginShape();
    if (xShapeVal.GetDimNum() != OgOpsCommon::kOgDimNum) {
        return false;
    }
    const uint64_t width = static_cast<uint64_t>(xShapeVal.GetDim(OgOpsCommon::kOgDim3));
    if (width == 0 || width > 4096) {
        return false;
    }

    const uint64_t dtypeSize = OgOpsCommon::GetOgDtypeSize(xDesc->GetDataType());
    if (dtypeSize == 0) {
        return false;
    }

    // bias 校验：bias2 若存在必须全尺寸；bias1 的 dtype 必须与 x 一致
    if (!CheckBiasValid(context, kBias2Index, xShapeVal, 1)) {
        return false;
    }
    if (!CheckBiasValid(context, kBias1Index, xShapeVal, 0)) {
        return false;
    }

    shape.batch = static_cast<uint64_t>(xShapeVal.GetDim(OgOpsCommon::kOgDim0));
    shape.channel = static_cast<uint64_t>(xShapeVal.GetDim(OgOpsCommon::kOgDim1));
    shape.height = static_cast<uint64_t>(xShapeVal.GetDim(OgOpsCommon::kOgDim2));
    shape.width = width;
    shape.dtypeSize = dtypeSize;
    shape.dtypeKey = OgOpsCommon::GetOgDtypeKey(xDesc->GetDataType());
    shape.hasB1 = (context->GetOptionalInputDesc(kBias1Index) != nullptr &&
                   context->GetOptionalInputShape(kBias1Index) != nullptr) ?
                      1 :
                      0;
    shape.hasB2 = (context->GetOptionalInputDesc(kBias2Index) != nullptr &&
                   context->GetOptionalInputShape(kBias2Index) != nullptr) ?
                      1 :
                      0;
    shape.b1Full = 0;
    if (shape.hasB1 == 1) {
        shape.b1Full = (context->GetOptionalInputShape(kBias1Index)->GetOriginShape().GetDim(OgOpsCommon::kOgDim2) ==
                        xShapeVal.GetDim(OgOpsCommon::kOgDim2)) ?
                           1 :
                           0;
    }
    return true;
}
} // namespace

namespace optiling {

ge::graphStatus FusedBiasSoftmaxTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    FusedBiasSoftmaxHost::FusedBiasSoftmaxShape shape;
    if (!CollectFusedBiasSoftmaxShape(context, shape)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    FusedBiasSoftmaxHost::FusedBiasSoftmaxPlan plan;
    FusedBiasSoftmaxTilingData tilingData;
    if (!FusedBiasSoftmaxHost::ComputeFusedBiasSoftmaxTiling(shape, platform, tilingData, plan)) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(plan.blockDim);
    context->SetTilingKey(plan.tilingKey);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
