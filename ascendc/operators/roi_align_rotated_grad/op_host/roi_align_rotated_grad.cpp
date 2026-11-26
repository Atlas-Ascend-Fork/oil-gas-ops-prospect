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
 * ROIAlignRotatedGrad host：OpDef / InferShape / InferDataType / Tiling。
 *
 *   grad_output [K,C,PH,PW] + rois [K,6] -> grad_input [N,C,H,W]
 *
 * 反向按 grad_input 元素空间做 64B slab 独占切分，kernel 内 UB 累加后整片回写。
 */

#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "og_tiling_context.h"
#include "roi_align_rotated_grad_tiling_data_def.h"
#include "roi_align_rotated_grad_tiling_core.h"

namespace {

constexpr uint32_t kGradOutput = 0;
constexpr uint32_t kRois = 1;
constexpr uint32_t kGradInput = 0;

constexpr int64_t kAttrSpatialScale = 0;
constexpr int64_t kAttrPooledHeight = 1;
constexpr int64_t kAttrPooledWidth = 2;
constexpr int64_t kAttrSamplingRatio = 3;
constexpr int64_t kAttrBatchSize = 4;
constexpr int64_t kAttrChannels = 5;
constexpr int64_t kAttrHeight = 6;
constexpr int64_t kAttrWidth = 7;

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

struct RoiAlignRotatedGradShape {
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

bool CollectShape(gert::TilingContext* context, RoiAlignRotatedGradShape& shape)
{
    const gert::StorageShape* gradStorage = context->GetInputShape(kGradOutput);
    const gert::StorageShape* roisStorage = context->GetInputShape(kRois);
    const auto gradDesc = context->GetInputDesc(kGradOutput);
    const auto roisDesc = context->GetInputDesc(kRois);
    if (gradStorage == nullptr || roisStorage == nullptr || gradDesc == nullptr || roisDesc == nullptr) {
        return false;
    }
    const gert::Shape& grad = gradStorage->GetOriginShape();
    const gert::Shape& rois = roisStorage->GetOriginShape();
    if (grad.GetDimNum() != 4 || rois.GetDimNum() != 2 || rois.GetDim(1) != 6) {
        return false;
    }
    if (gradDesc->GetDataType() != ge::DT_FLOAT || roisDesc->GetDataType() != ge::DT_FLOAT) {
        return false;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    shape.k = rois.GetDim(0);
    shape.n = ReadIntAttr(attrs, kAttrBatchSize, 1);
    shape.c = ReadIntAttr(attrs, kAttrChannels, 1);
    shape.h = ReadIntAttr(attrs, kAttrHeight, 1);
    shape.w = ReadIntAttr(attrs, kAttrWidth, 1);
    shape.pooledH = ReadIntAttr(attrs, kAttrPooledHeight, 7);
    shape.pooledW = ReadIntAttr(attrs, kAttrPooledWidth, 7);
    shape.samplingRatio = ReadIntAttr(attrs, kAttrSamplingRatio, 0);
    shape.spatialScale = ReadFloatAttr(attrs, kAttrSpatialScale, 1.0F);

    // grad_output 的 K/C/PH/PW 必须与 rois 行数和属性一致。
    if (grad.GetDim(0) != shape.k || grad.GetDim(1) != shape.c || grad.GetDim(2) != shape.pooledH ||
        grad.GetDim(3) != shape.pooledW) {
        return false;
    }
    return true;
}

} // namespace

namespace optiling {

static ge::graphStatus RoiAlignRotatedGradTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    RoiAlignRotatedGradShape shape;
    if (!CollectShape(context, shape)) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    ROIAlignRotatedGradTilingData tiling{};
    RoiAlignRotatedGradHost::RoiAlignRotatedGradPlan plan;
    if (!RoiAlignRotatedGradHost::ComputeRoiAlignRotatedGradTilingCore(
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

static ge::graphStatus RoiAlignRotatedGradInferShape(gert::InferShapeContext* context)
{
    gert::Shape* output = context->GetOutputShape(kGradInput);
    if (output == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t n = ReadIntAttr(attrs, kAttrBatchSize, 1);
    const int64_t c = ReadIntAttr(attrs, kAttrChannels, 1);
    const int64_t h = ReadIntAttr(attrs, kAttrHeight, 1);
    const int64_t w = ReadIntAttr(attrs, kAttrWidth, 1);
    if (n < 0 || c < 0 || h < 0 || w < 0) {
        return ge::GRAPH_FAILED;
    }
    output->SetDimNum(4);
    output->SetDim(0, n);
    output->SetDim(1, c);
    output->SetDim(2, h);
    output->SetDim(3, w);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RoiAlignRotatedGradInferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kGradInput, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class ROIAlignRotatedGrad : public OpDef {
public:
    explicit ROIAlignRotatedGrad(const char* name) : OpDef(name)
    {
        this->Input("grad_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rois")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("spatial_scale").AttrType(OPTIONAL).Float(1.0F);
        this->Attr("pooled_height").AttrType(OPTIONAL).Int(7);
        this->Attr("pooled_width").AttrType(OPTIONAL).Int(7);
        this->Attr("sampling_ratio").AttrType(OPTIONAL).Int(0);
        this->Attr("batch_size").AttrType(OPTIONAL).Int(1);
        this->Attr("channels").AttrType(OPTIONAL).Int(1);
        this->Attr("height").AttrType(OPTIONAL).Int(1);
        this->Attr("width").AttrType(OPTIONAL).Int(1);

        this->SetInferShape(ge::RoiAlignRotatedGradInferShape).SetInferDataType(ge::RoiAlignRotatedGradInferDataType);
        this->AICore().SetTiling(optiling::RoiAlignRotatedGradTilingFunc);
        AddAscend910bElementwiseConfig(*this, "roi_align_rotated_grad");
    }
};

OP_ADD(ROIAlignRotatedGrad);
} // namespace ops
