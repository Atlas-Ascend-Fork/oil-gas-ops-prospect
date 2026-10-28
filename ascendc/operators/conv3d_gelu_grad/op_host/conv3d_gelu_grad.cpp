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
 * Conv3DGeluGrad：dz = dy * gelu'(z)，float32 / ND。
 * 卷积的重算和 conv3d 反向由 L2 aclnnConv3dGeluGrad 串在同一个 executor 里。
 */

#include <cstdint>
#include <cstring>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/conv3d_gelu_grad_tiling.h"

namespace {

constexpr uint32_t kInputZ = 0;
constexpr int64_t kUbFactor = 1024;

} // namespace

namespace optiling {

static ge::graphStatus Conv3DGeluGradTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* zShape = context->GetInputShape(kInputZ);
    if (zShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t total = 1;
    const gert::Shape& shape = zShape->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        total *= shape.GetDim(i);
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    int64_t aivNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    if (aivNum <= 0) {
        aivNum = 1;
    }
    int64_t blockFactor = (total + aivNum - 1) / aivNum;
    blockFactor = (blockFactor + 7) / 8 * 8;

    Conv3DGeluGradTilingData tilingData{};
    tilingData.totalNum = total;
    tilingData.blockFactor = blockFactor;
    tilingData.ubFactor = kUbFactor;

    gert::TilingData* rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < sizeof(tilingData)) {
        return ge::GRAPH_FAILED;
    }
    if (memcpy_s(rawTiling->GetData(), rawTiling->GetCapacity(), &tilingData, sizeof(tilingData)) != 0) {
        return ge::GRAPH_FAILED;
    }
    rawTiling->SetDataSize(sizeof(tilingData));
    context->SetBlockDim(static_cast<uint32_t>(aivNum));
    context->SetTilingKey(0);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus Conv3DGeluGradInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* zShape = context->GetInputShape(kInputZ);
    gert::Shape* dzShape = context->GetOutputShape(0);
    if (zShape == nullptr || dzShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *dzShape = *zShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Conv3DGeluGradInferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class Conv3DGeluGrad : public OpDef {
public:
    explicit Conv3DGeluGrad(const char* name) : OpDef(name)
    {
        this->Input("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grad_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::Conv3DGeluGradInferShape).SetInferDataType(ge::Conv3DGeluGradInferDataType);
        this->AICore().SetTiling(optiling::Conv3DGeluGradTilingFunc);

        OpAICoreConfig cfg;
        cfg.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "conv3d_gelu_grad")
            .ExtendCfgInfo("opInterface.value", "conv3d_gelu_grad")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "false");
        this->AICore().AddConfig("ascend910b", cfg);
    }
};

OP_ADD(Conv3DGeluGrad);
} // namespace ops
