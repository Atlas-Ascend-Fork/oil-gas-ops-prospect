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

#include "custom_gruv2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

constexpr uint32_t kDefaultAivCoreNum = 24;
// Per-core unified buffer byte budget on 910B2 (192KB total minus a 16KB
// reservation for temporaries and compiler-generated scratch).
constexpr uint64_t kUbByteBudget = 176 * 1024;

inline uint32_t CeilDivU32(uint32_t dividend, uint32_t divisor) { return (dividend + divisor - 1) / divisor; }

// Reads the optional "bidirectional" attribute (0 or 1). Absent attribute and
// null attribute storage both fall back to unidirectional (0).
inline uint32_t ReadBidirectionalFlag(const gert::RuntimeAttrs* attrStore)
{
    if (attrStore == nullptr || attrStore->GetAttrNum() == 0) {
        return 0U;
    }
    const bool* flagPtr = attrStore->GetAttrPointer<bool>(0);
    return (flagPtr != nullptr && *flagPtr) ? 1U : 0U;
}

inline bool FailEmptyTiling(gert::TilingContext* context)
{
    context->SetTilingKey(0);
    context->GetRawTilingData()->SetDataSize(0);
    return false;
}

} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xStorage = context->GetInputShape(0);
    const gert::StorageShape* wiStorage = context->GetInputShape(1);
    const gert::StorageShape* whStorage = context->GetInputShape(2);
    if (xStorage == nullptr || wiStorage == nullptr || whStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& xDims = xStorage->GetStorageShape();
    const gert::Shape& wiDims = wiStorage->GetStorageShape();
    const gert::Shape& whDims = whStorage->GetStorageShape();
    if (xDims.GetDimNum() < 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batch = static_cast<uint32_t>(xDims.GetDim(0));
    const uint32_t seq = static_cast<uint32_t>(xDims.GetDim(1));
    const uint32_t featDim = static_cast<uint32_t>(xDims.GetDim(2));

    const uint32_t bidirectional = ReadBidirectionalFlag(context->GetAttrs());
    const uint32_t dirCount = (bidirectional != 0U) ? 2U : 1U;

    const uint32_t hidden = static_cast<uint32_t>(wiDims.GetDim(0) / (3 * dirCount));
    if (hidden == 0U) {
        return ge::GRAPH_FAILED;
    }
    if (whDims.GetDim(0) != wiDims.GetDim(0) || whDims.GetDim(1) != hidden) {
        return ge::GRAPH_FAILED;
    }

    // Round the hidden width up to a multiple of 8 so that vectorized unit
    // strides stay aligned during the recurrent scan.
    const uint32_t hiddenAligned = (hidden + 7U) & ~7U;

    uint32_t aivCoreNum = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetCoreNumAiv();
    if (aivCoreNum == 0U) {
        aivCoreNum = kDefaultAivCoreNum;
    }

    uint32_t blockCount = (batch < aivCoreNum) ? batch : aivCoreNum;
    if (blockCount == 0U) {
        blockCount = 1U;
    }
    uint32_t batchPerBlock = CeilDivU32(batch, blockCount);

    // Unified-buffer footprint per core: hidden state, three gate accumulators
    // and four scratch vectors, all in fp32.
    uint64_t ubBytes = static_cast<uint64_t>(batchPerBlock * hiddenAligned + 7U * hiddenAligned) * sizeof(float);
    if (ubBytes > kUbByteBudget) {
        const uint64_t ubSlots = kUbByteBudget / sizeof(float);
        const uint64_t fixedSlots = static_cast<uint64_t>(7U) * hiddenAligned;
        if (ubSlots <= fixedSlots) {
            FailEmptyTiling(context);
            return ge::GRAPH_FAILED;
        }
        const uint32_t maxBatchPerBlock = static_cast<uint32_t>((ubSlots - fixedSlots) / hiddenAligned);
        if (maxBatchPerBlock == 0U) {
            FailEmptyTiling(context);
            return ge::GRAPH_FAILED;
        }
        blockCount = CeilDivU32(batch, maxBatchPerBlock);
        batchPerBlock = CeilDivU32(batch, blockCount);
    }

    const bool hasBias = (context->GetInputShape(3) != nullptr) && (context->GetInputShape(4) != nullptr);

    CustomGRUV2TilingData tilingData;
    tilingData.set_batchSize(batch);
    tilingData.set_seqLen(seq);
    tilingData.set_inputSize(featDim);
    tilingData.set_hiddenSize(hidden);
    tilingData.set_bidirectional(bidirectional);
    tilingData.set_hasBias(hasBias ? 1 : 0);

    context->SetBlockDim(blockCount);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
namespace {

inline uint32_t DirectionCountFromAttrs(const gert::RuntimeAttrs* attrStore)
{
    if (attrStore == nullptr || attrStore->GetAttrNum() == 0) {
        return 1U;
    }
    const bool* flagPtr = attrStore->GetAttrPointer<bool>(0);
    return (flagPtr != nullptr && *flagPtr) ? 2U : 1U;
}

} // namespace

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* wiShape = context->GetInputShape(1);
    if (xShape == nullptr || wiShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (xShape->GetDimNum() < 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batch = static_cast<uint32_t>(xShape->GetDim(0));
    const uint32_t seq = static_cast<uint32_t>(xShape->GetDim(1));
    const uint32_t dirCount = DirectionCountFromAttrs(context->GetAttrs());
    const uint32_t hidden = static_cast<uint32_t>(wiShape->GetDim(0) / (3 * dirCount));

    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(4);
    yShape->SetDim(0, dirCount);
    yShape->SetDim(1, batch);
    yShape->SetDim(2, seq);
    yShape->SetDim(3, hidden);

    gert::Shape* hShape = context->GetOutputShape(1);
    hShape->SetDimNum(3);
    hShape->SetDim(0, dirCount);
    hShape->SetDim(1, batch);
    hShape->SetDim(2, hidden);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const ge::DataType inputDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDtype);
    context->SetOutputDataType(1, inputDtype);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {
namespace {

struct ParamSpec {
    const char* name;
    bool required;
};

const ParamSpec kInputSpecs[] = {
    {"x", true}, {"weight_ih", true}, {"weight_hh", true}, {"bias_ih", false}, {"bias_hh", false},
};

const ParamSpec kOutputSpecs[] = {
    {"y", true},
    {"output_h", true},
};

void DeclareFloatNdParam(OpDef* opDef, const ParamSpec& spec, bool isInput)
{
    if (isInput) {
        opDef->Input(spec.name)
            .ParamType(spec.required ? REQUIRED : OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
    } else {
        opDef->Output(spec.name)
            .ParamType(spec.required ? REQUIRED : OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
    }
}

} // namespace

class CustomGRUV2 : public OpDef {
public:
    explicit CustomGRUV2(const char* opType) : OpDef(opType)
    {
        for (const ParamSpec& spec : kInputSpecs) {
            DeclareFloatNdParam(this, spec, true);
        }
        for (const ParamSpec& spec : kOutputSpecs) {
            DeclareFloatNdParam(this, spec, false);
        }

        this->Attr("bidirectional").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(CustomGRUV2);
} // namespace ops
