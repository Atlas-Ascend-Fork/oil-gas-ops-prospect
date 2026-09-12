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
 * FusedBiasSoftmax host：openfold 注意力前向融合算子。
 *
 *   logits = softmax(logits + bias1(广播/全尺寸) + bias2, -1)，原地写回
 *
 * 对标 openfold attn_core_inplace_cuda.forward_ 的三次 kernel 发射
 * （bias_1 add / bias_2 add / in-place softmax），单 kernel 消除中间访存。
 */

#include <cstdint>

#include "register/op_def_registry.h"

#include "og_tiling_context.h"

#include "fused_bias_softmax_tiling_data_def.h"
#include "fused_bias_softmax_tiling_impl.h"

namespace {
constexpr uint32_t kShapeInput = 0;
} // namespace

namespace optiling {
ge::graphStatus FusedBiasSoftmaxTilingFunc(gert::TilingContext* context); // 定义于 fused_bias_softmax_tiling.cpp
} // namespace optiling

namespace ge {

static ge::graphStatus FusedBiasSoftmaxInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(kShapeInput);
    if (xShape == nullptr || context->GetOutputShape(kShapeInput) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(kShapeInput) = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FusedBiasSoftmaxInferDataType(gert::InferDataTypeContext* context)
{
    auto inputDtype = context->GetInputDataType(kShapeInput);
    context->SetOutputDataType(kShapeInput, inputDtype);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class FusedBiasSoftmax : public OpDef {
public:
    explicit FusedBiasSoftmax(const char* name) : OpDef(name)
    {
        // x / bias1 / bias2 / y：fp32 / fp16 / bf16，ND；bias1、bias2 可选
        const ge::DataType dtypes[] = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
        const ge::Format fmts[] = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({dtypes[0], dtypes[1], dtypes[2]})
            .Format({fmts[0], fmts[1], fmts[2]})
            .UnknownShapeFormat({fmts[0], fmts[1], fmts[2]})
            .AutoContiguous();
        this->Input("bias1")
            .ParamType(OPTIONAL)
            .DataType({dtypes[0], dtypes[1], dtypes[2]})
            .Format({fmts[0], fmts[1], fmts[2]})
            .UnknownShapeFormat({fmts[0], fmts[1], fmts[2]})
            .AutoContiguous();
        this->Input("bias2")
            .ParamType(OPTIONAL)
            .DataType({dtypes[0], dtypes[1], dtypes[2]})
            .Format({fmts[0], fmts[1], fmts[2]})
            .UnknownShapeFormat({fmts[0], fmts[1], fmts[2]})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({dtypes[0], dtypes[1], dtypes[2]})
            .Format({fmts[0], fmts[1], fmts[2]})
            .UnknownShapeFormat({fmts[0], fmts[1], fmts[2]});

        this->SetInferShape(ge::FusedBiasSoftmaxInferShape).SetInferDataType(ge::FusedBiasSoftmaxInferDataType);
        this->AICore().SetTiling(optiling::FusedBiasSoftmaxTilingFunc);

        // aclnn autogen：opFile/opInterface 指向 op_kernel/fused_bias_softmax.cpp
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true);
        config.DynamicFormatFlag(true);
        config.DynamicRankSupportFlag(true);
        config.DynamicShapeSupportFlag(true);
        config.NeedCheckSupportFlag(false);
        config.PrecisionReduceFlag(false);
        config.ExtendCfgInfo("opFile.value", "fused_bias_softmax");
        config.ExtendCfgInfo("opInterface.value", "fused_bias_softmax");
        config.ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        config.ExtendCfgInfo("jitCompile.flag", "false");
        this->AICore().AddConfig("ascend910b", config);
    }
};

OP_ADD(FusedBiasSoftmax);
} // namespace ops
