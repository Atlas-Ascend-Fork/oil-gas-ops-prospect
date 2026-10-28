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
 * Conv3DGelu host: op definition + infer + tiling。
 *
 * conv3d + GELU 融合算子（高性能 float32/HF32）。cube 阶段复用 conv3d_v2 的解耦 tiling
 * 引擎与 kernel；vector 阶段对 conv 输出逐元素做 tanh 近似 GELU。
 * 目标平台: ascend910b；x/y 为 NDC1HWC0，filter 为 FRACTAL_Z_3D，bias 为 ND。
 */

#include <cstring>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "op_tiling/conv3d_tiling_engine.h"
#include "../op_kernel/conv3d_gelu_tiling_key.h"
#include "../op_kernel/conv3d_gelu_tiling_data.h"

namespace {
constexpr int64_t kAttrStrides = 0;
constexpr int64_t kAttrPads = 1;
constexpr int64_t kAttrDilations = 2;
constexpr int64_t kAttrGroups = 3;
constexpr int64_t kAttrEnableHf32 = 4;

constexpr uint32_t kInputX = 0;
constexpr uint32_t kInputFilter = 1;
constexpr uint32_t kInputBias = 2;
constexpr uint32_t kOutputY = 0;

constexpr uint32_t kPbufL0aMask = 1;
constexpr uint32_t kPbufL0bMask = 2;

// gelu 向量阶段：MIX_AIC_1_2，每个 cube 核对应 2 个 AIV 核
constexpr uint32_t kAivPerCube = 2;
constexpr int64_t kGeluUbFactor = 8192;

int64_t ConvOutSize(int64_t inSize, int64_t padBefore, int64_t padAfter, int64_t kernel, int64_t dilation,
                    int64_t stride)
{
    return (inSize + padBefore + padAfter - dilation * (kernel - 1) - 1) / stride + 1;
}

void ReadList3(const gert::RuntimeAttrs* attrs, int64_t idx, int64_t def, int64_t out[3])
{
    out[0] = out[1] = out[2] = def;
    if (attrs == nullptr) {
        return;
    }
    const auto* list = attrs->GetListInt(idx);
    if (list == nullptr || list->GetSize() == 0) {
        return;
    }
    if (list->GetSize() == 1) {
        out[0] = out[1] = out[2] = list->GetData()[0];
    } else if (list->GetSize() >= 3) {
        for (int i = 0; i < 3; ++i) {
            out[i] = list->GetData()[i];
        }
    }
}

void ReadPads(const gert::RuntimeAttrs* attrs, int64_t out[6])
{
    for (int i = 0; i < 6; ++i) {
        out[i] = 0;
    }
    if (attrs == nullptr) {
        return;
    }
    const auto* list = attrs->GetListInt(kAttrPads);
    if (list == nullptr || list->GetSize() == 0) {
        return;
    }
    if (list->GetSize() == 3) {
        for (int i = 0; i < 3; ++i) {
            out[i * 2] = list->GetData()[i];
            out[i * 2 + 1] = list->GetData()[i];
        }
    } else if (list->GetSize() >= 6) {
        for (int i = 0; i < 6; ++i) {
            out[i] = list->GetData()[i];
        }
    }
}

int64_t ReadGroups(const gert::RuntimeAttrs* attrs)
{
    if (attrs == nullptr) {
        return 1;
    }
    const int64_t* g = attrs->GetInt(kAttrGroups);
    return (g == nullptr || *g <= 0) ? 1 : *g;
}

bool ReadBool(const gert::RuntimeAttrs* attrs, int64_t idx, bool def)
{
    if (attrs == nullptr) {
        return def;
    }
    const bool* value = attrs->GetBool(idx);
    return value == nullptr ? def : *value;
}
} // namespace

namespace optiling {

using namespace Ops::NN::Conv3dTilingEngineApi;
using namespace Conv3dGeluTilingKey;

static ge::graphStatus Conv3dGeluTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* xShape = context->GetInputShape(kInputX);
    const gert::StorageShape* wShape = context->GetInputShape(kInputFilter);
    const gert::StorageShape* yShape = context->GetOutputShape(kOutputY);
    if (xShape == nullptr || wShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& x = xShape->GetOriginShape(); // NCDHW
    const auto& w = wShape->GetOriginShape(); // NCDHW (Cout,Cin,kd,kh,kw)
    if (x.GetDimNum() != 5 || w.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    int64_t stride[3] = {1, 1, 1};
    int64_t dilation[3] = {1, 1, 1};
    int64_t pads[6] = {0, 0, 0, 0, 0, 0};
    ReadList3(attrs, kAttrStrides, 1, stride);
    ReadList3(attrs, kAttrDilations, 1, dilation);
    ReadPads(attrs, pads);
    int64_t groups = ReadGroups(attrs);
    // 分组卷积的 kernel 分支未做精度验收。aclnn / Python 同样只接受 groups=1。
    if (groups != 1) {
        return ge::GRAPH_FAILED;
    }
    bool enableHf32 = ReadBool(attrs, kAttrEnableHf32, false);

    int64_t batch = x.GetDim(0);
    int64_t cin = x.GetDim(1);
    int64_t di = x.GetDim(2);
    int64_t hi = x.GetDim(3);
    int64_t wi = x.GetDim(4);
    int64_t cout = w.GetDim(0);
    int64_t kd = w.GetDim(2);
    int64_t kh = w.GetDim(3);
    int64_t kw = w.GetDim(4);
    int64_t dout = ConvOutSize(di, pads[0], pads[1], kd, dilation[0], stride[0]);
    int64_t hout = ConvOutSize(hi, pads[2], pads[3], kh, dilation[1], stride[1]);
    int64_t wout = ConvOutSize(wi, pads[4], pads[5], kw, dilation[2], stride[2]);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    Conv3dTilingEngine engine("Conv3DGelu");
    engine.platformInfo_.aicoreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, engine.platformInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, engine.platformInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, engine.platformInfo_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, engine.platformInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, engine.platformInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::BT, engine.platformInfo_.btSize);
    ascendcPlatform.GetCoreMemBw(platform_ascendc::CoreMemType::L2, engine.platformInfo_.l2Rate);
    if (engine.platformInfo_.l2Rate == 0) {
        engine.platformInfo_.l2Rate = 1;
    }
    engine.platformInfo_.socVersion = "ascend910b";
    engine.SetInitialized(true);

    bool hasBias = context->GetOptionalInputShape(kInputBias) != nullptr;

    // float32 支持两种格式组合（与 conv3d_v2 ascend910b float32 一致）：
    //   - NDC1HWC0：x/y = NDC1HWC0, filter = FRACTAL_Z_3D（常规 cube 路径）；
    //   - NCDHW   ：x/filter/y 均为 NCDHW（1x1x1 pointwise 路径）。
    // 实际走哪条由框架在已注册列表中选中的 x 存储格式决定。
    auto xDesc = context->GetInputDesc(kInputX);
    if (xDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    bool isNCDHW = (static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_NCDHW);

    engine.SetOrgFmapShape({batch, cin, di, hi, wi});
    engine.SetOrgWeightShape({cout, cin, kd, kh, kw});
    engine.SetOrgOutputShape({batch, cout, dout, hout, wout});
    engine.SetStride({stride[0], stride[1], stride[2]}); // D,H,W
    engine.SetDilation({dilation[0], dilation[1], dilation[2]});
    engine.SetPadding({pads[0], pads[1], pads[2], pads[3], pads[4], pads[5]});
    engine.SetGroups(groups);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::FLOAT32,
                       Conv3dApiTiling::ConvDtype::FLOAT32);
    if (isNCDHW) {
        engine.SetFormat(Conv3dApiTiling::ConvFormat::NCDHW, Conv3dApiTiling::ConvFormat::NCDHW,
                         Conv3dApiTiling::ConvFormat::NCDHW);
    } else {
        engine.SetFormat(Conv3dApiTiling::ConvFormat::NDC1HWC0, Conv3dApiTiling::ConvFormat::FRACTAL_Z_3D,
                         Conv3dApiTiling::ConvFormat::NDC1HWC0);
    }
    engine.SetBias(hasBias, Conv3dApiTiling::ConvDtype::FLOAT32);
    if (hasBias) {
        // bias 为 1D [Cout]，tiling 引擎校验要求显式提供 bias shape，否则 CheckBiasShape 失败。
        engine.SetBiasShape({cout});
    }
    engine.SetScale(false, Conv3dApiTiling::ConvDtype::FLOAT32);
    engine.SetHF32(enableHf32);

    Conv3DGeluTilingData tilingData;
    if (!engine.GetConv3DV2TilingData(tilingData.convTiling)) {
        return ge::GRAPH_FAILED;
    }

    const auto& runInfo = tilingData.convTiling.conv3dRunInfo;
    uint32_t blockDim = runInfo.batchDim * runInfo.nDim * runInfo.mDim * runInfo.doDim * runInfo.groupDim;
    if (blockDim == 0) {
        blockDim = 1;
    }
    context->SetBlockDim(blockDim);

    // gelu 向量阶段切分：对 conv 输出（NDC1HWC0 存储）逐元素处理。
    // 总元素 = 输出张量存储 shape 的元素数；AIV 核数 = blockDim * 2（MIX_AIC_1_2）。
    int64_t geluTotalNum = yShape->GetStorageShape().GetShapeSize();
    int64_t aivNum = static_cast<int64_t>(blockDim) * kAivPerCube;
    if (aivNum <= 0) {
        aivNum = 1;
    }
    tilingData.geluTotalNum = geluTotalNum;
    tilingData.geluBlockFactor = (geluTotalNum + aivNum - 1) / aivNum;
    tilingData.geluUbFactor = kGeluUbFactor;

    gert::TilingData* rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < sizeof(tilingData)) {
        return ge::GRAPH_FAILED;
    }
    if (memcpy_s(rawTiling->GetData(), rawTiling->GetCapacity(), &tilingData, sizeof(tilingData)) != 0) {
        return ge::GRAPH_FAILED;
    }
    rawTiling->SetDataSize(sizeof(tilingData));

    uint64_t pbufferFlag = tilingData.convTiling.conv3dApiTiling.pBufferFlag;
    uint64_t bl1Bypass = tilingData.convTiling.conv3dApiTiling.bl1BypassFlag;
    uint64_t convL0PingPong = L0_PINGPONG_ALL_CLOSE;
    if ((pbufferFlag & kPbufL0aMask) && (pbufferFlag & kPbufL0bMask)) {
        convL0PingPong = L0_PINGPONG_ALL_OPEN;
    } else if (pbufferFlag & kPbufL0aMask) {
        convL0PingPong = L0_PINGPONG_L0A_OPEN;
    } else if (pbufferFlag & kPbufL0bMask) {
        convL0PingPong = L0_PINGPONG_L0B_OPEN;
    }
    uint64_t groupConvType = tilingData.convTiling.conv3dApiTiling.groups > 1 ? GROUPCONV_WEIGHT_GFZ : NOGROUP_CONV;
    uint64_t outputOrder = static_cast<uint64_t>(tilingData.convTiling.conv3dApiTiling.outputOrder);
    context->SetTilingKey(GET_TPL_TILING_KEY(convL0PingPong, bl1Bypass, groupConvType, outputOrder));

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(16) * 1024 * 1024; // 16MB system workspace

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static ge::graphStatus Conv3dGeluInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* wShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || wShape == nullptr || yShape == nullptr || xShape->GetDimNum() != 5 ||
        wShape->GetDimNum() != 5) {
        return GRAPH_FAILED;
    }

    int64_t stride[3] = {1, 1, 1};
    int64_t dilation[3] = {1, 1, 1};
    int64_t pads[6] = {0, 0, 0, 0, 0, 0};
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    ReadList3(attrs, kAttrStrides, 1, stride);
    ReadList3(attrs, kAttrDilations, 1, dilation);
    ReadPads(attrs, pads);
    if (ReadGroups(attrs) != 1) {
        return GRAPH_FAILED;
    }

    yShape->SetDimNum(5);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, wShape->GetDim(0));
    yShape->SetDim(2, ConvOutSize(xShape->GetDim(2), pads[0], pads[1], wShape->GetDim(2), dilation[0], stride[0]));
    yShape->SetDim(3, ConvOutSize(xShape->GetDim(3), pads[2], pads[3], wShape->GetDim(3), dilation[1], stride[1]));
    yShape->SetDim(4, ConvOutSize(xShape->GetDim(4), pads[4], pads[5], wShape->GetDim(4), dilation[2], stride[2]));
    return GRAPH_SUCCESS;
}

static ge::graphStatus Conv3dGeluInferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class Conv3DGelu : public OpDef {
public:
    explicit Conv3DGelu(const char* name) : OpDef(name)
    {
        // float32 支持的两种格式组合（对齐 conv3d_v2 ascend910b float32）：
        //   组合0: x/y = NDC1HWC0, filter = FRACTAL_Z_3D；组合1: x/filter/y = NCDHW。
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW});
        this->Input("filter")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW});

        this->Attr("strides").AttrType(OPTIONAL).ListInt({1, 1, 1});
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0, 0, 0, 0, 0, 0});
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("enable_hf32").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::Conv3dGeluInferShape).SetInferDataType(ge::Conv3dGeluInferDataType);
        this->AICore().SetTiling(optiling::Conv3dGeluTilingFunc);

        OpAICoreConfig cfg;
        cfg.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "conv3d_gelu")
            .ExtendCfgInfo("opInterface.value", "conv3d_gelu")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "false");
        this->AICore().AddConfig("ascend910b", cfg);
    }
};

OP_ADD(Conv3DGelu);
} // namespace ops
