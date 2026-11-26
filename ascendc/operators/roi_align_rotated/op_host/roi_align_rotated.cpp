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
 * ROIAlignRotated host：OpDef / InferShape / InferDataType / Tiling。
 *
 *   input [N,C,H,W] + rois [K,6] -> out [K,C,PH,PW]
 *   rois 每行: (batch_idx, x_ctr, y_ctr, width, height, angle_deg)
 *
 * 切分算法在 roi_align_rotated_tiling_core.h（ophost UT 覆盖）。
 */

#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "og_tiling_context.h"
#include "roi_align_rotated_tiling_data_def.h"
#include "roi_align_rotated_tiling_core.h"

namespace {

constexpr uint32_t kInput = 0;
constexpr uint32_t kRois = 1;
constexpr uint32_t kOutput = 0;

constexpr int64_t kAttrSpatialScale = 0;
constexpr int64_t kAttrPooledHeight = 1;
constexpr int64_t kAttrPooledWidth = 2;
constexpr int64_t kAttrSamplingRatio = 3;

int64_t ReadIntAttr(const gert::RuntimeAttrs* attrs, int64_t index, int64_t defaultValue)
{
    if (attrs == nullptr) {
        return defaultValue;
    }
    const int64_t* value = attrs->GetInt(index);
    return value == nullptr ? defaultValue : *value;
}

float ReadFloatAttr(const gert::RuntimeAttrs* attrs, int64_t index, float defaultValue)
{
    if (attrs == nullptr) {
        return defaultValue;
    }
    const float* value = attrs->GetFloat(index);
    return value == nullptr ? defaultValue : *value;
}

struct RoiAlignRotatedShape {
    int64_t n = 0;
    int64_t c = 0;
    int64_t h = 0;
    int64_t w = 0;
    int64_t k = 0;
    int64_t pooledH = 0;
    int64_t pooledW = 0;
    int64_t samplingRatio = 0;
    float spatialScale = 1.0F;
};

bool CollectShape(gert::TilingContext* context, RoiAlignRotatedShape& shape)
{
    const gert::StorageShape* inputStorage = context->GetInputShape(kInput);
    const gert::StorageShape* roisStorage = context->GetInputShape(kRois);
    const auto inputDesc = context->GetInputDesc(kInput);
    const auto roisDesc = context->GetInputDesc(kRois);
    if (inputStorage == nullptr || roisStorage == nullptr || inputDesc == nullptr || roisDesc == nullptr) {
        return false;
    }
    const gert::Shape& input = inputStorage->GetOriginShape();
    const gert::Shape& rois = roisStorage->GetOriginShape();
    if (input.GetDimNum() != 4 || rois.GetDimNum() != 2 || rois.GetDim(1) != 6) {
        return false;
    }
    if (inputDesc->GetDataType() != ge::DT_FLOAT || roisDesc->GetDataType() != ge::DT_FLOAT) {
        return false;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    shape.n = input.GetDim(0);
    shape.c = input.GetDim(1);
    shape.h = input.GetDim(2);
    shape.w = input.GetDim(3);
    shape.k = rois.GetDim(0);
    shape.pooledH = ReadIntAttr(attrs, kAttrPooledHeight, 7);
    shape.pooledW = ReadIntAttr(attrs, kAttrPooledWidth, 7);
    shape.samplingRatio = ReadIntAttr(attrs, kAttrSamplingRatio, 0);
    shape.spatialScale = ReadFloatAttr(attrs, kAttrSpatialScale, 1.0F);
    return true;
}

} // namespace

namespace optiling {

static ge::graphStatus RoiAlignRotatedTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    RoiAlignRotatedShape shape;
    if (!CollectShape(context, shape)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    ROIAlignRotatedTilingData tiling{};
    RoiAlignRotatedHost::RoiAlignRotatedPlan plan;
    if (!RoiAlignRotatedHost::ComputeRoiAlignRotatedTilingCore(
            shape.n, shape.c, shape.h, shape.w, shape.k, shape.pooledH, shape.pooledW, shape.samplingRatio,
            shape.spatialScale, platform.GetCoreNumAiv(), tiling, plan)) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(plan.blockDim);
    context->SetTilingKey(plan.tilingKey);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(plan.totalWorkspace);

    if (!OgOpsCommon::StoreTilingData(context->GetRawTilingData(), tiling)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus RoiAlignRotatedInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* input = context->GetInputShape(kInput);
    const gert::Shape* rois = context->GetInputShape(kRois);
    gert::Shape* output = context->GetOutputShape(kOutput);
    if (input == nullptr || rois == nullptr || output == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (input->GetDimNum() != 4 || rois->GetDimNum() != 2 || rois->GetDim(1) != 6) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t pooledH = ReadIntAttr(attrs, kAttrPooledHeight, 7);
    const int64_t pooledW = ReadIntAttr(attrs, kAttrPooledWidth, 7);
    if (pooledH <= 0 || pooledW <= 0) {
        return ge::GRAPH_FAILED;
    }

    output->SetDimNum(4);
    output->SetDim(0, rois->GetDim(0));
    output->SetDim(1, input->GetDim(1));
    output->SetDim(2, pooledH);
    output->SetDim(3, pooledW);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RoiAlignRotatedInferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kOutput, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class ROIAlignRotated : public OpDef {
public:
    explicit ROIAlignRotated(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rois")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("spatial_scale").AttrType(OPTIONAL).Float(1.0F);
        this->Attr("pooled_height").AttrType(OPTIONAL).Int(7);
        this->Attr("pooled_width").AttrType(OPTIONAL).Int(7);
        this->Attr("sampling_ratio").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::RoiAlignRotatedInferShape).SetInferDataType(ge::RoiAlignRotatedInferDataType);
        this->AICore().SetTiling(optiling::RoiAlignRotatedTilingFunc);
        AddAscend910bElementwiseConfig(*this, "roi_align_rotated");
    }
};

OP_ADD(ROIAlignRotated);
} // namespace ops
