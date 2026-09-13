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
 * FusedSoftmaxGrad host：openfold 注意力反向融合算子。
 *
 *   g = go @ V^T（plane 对齐，Q≠S 合法）;
 *   probs ← probs ⊙ (g − rowsum(probs⊙g))，fp32 域，原地写回
 *
 * 对标 openfold attn_core_inplace_cuda.backward_（kernel 内对角内积 + JVP +
 * 原地覆写），替换 torch 分解序列 bmm + _softmax_backward_data + copy_。
 *
 * 架构：全部 bf16 走 MIX_AIC_1_2（未对齐时 AIV 先 pad 再 AIC 矩乘 g(half)
 * → SyncAll → AIV JVP）；fp32/fp16 走纯向量（tiling key 0/1）。fp32 走 cube
 * 受 fixpipe C 16bit 结构上限，见 PR 测试报告。
 */

#include <cstdint>

#include "register/op_def_registry.h"

#include "fused_softmax_grad_tiling_data_def.h"

namespace {
constexpr uint32_t kShapeProbs = 0;
} // namespace

namespace optiling {
ge::graphStatus FusedSoftmaxGradTilingFunc(gert::TilingContext* context); // 定义于 *_tiling.cpp
} // namespace optiling

namespace ge {

static ge::graphStatus FusedSoftmaxGradInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* probsShape = context->GetInputShape(kShapeProbs);
    gert::Shape* outShape = context->GetOutputShape(kShapeProbs);
    if (probsShape == nullptr || outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outShape = *probsShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FusedSoftmaxGradInferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kShapeProbs, context->GetInputDataType(kShapeProbs));
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class FusedSoftmaxGrad : public OpDef {
public:
    explicit FusedSoftmaxGrad(const char* name) : OpDef(name)
    {
        const ge::DataType dtypes[] = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
        const ge::Format fmts[] = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("probs")
            .ParamType(REQUIRED)
            .DataType({dtypes[0], dtypes[1], dtypes[2]})
            .Format({fmts[0], fmts[1], fmts[2]})
            .UnknownShapeFormat({fmts[0], fmts[1], fmts[2]})
            .AutoContiguous();
        this->Input("gradOutput")
            .ParamType(REQUIRED)
            .DataType({dtypes[0], dtypes[1], dtypes[2]})
            .Format({fmts[0], fmts[1], fmts[2]})
            .UnknownShapeFormat({fmts[0], fmts[1], fmts[2]})
            .AutoContiguous();
        this->Input("values")
            .ParamType(REQUIRED)
            .DataType({dtypes[0], dtypes[1], dtypes[2]})
            .Format({fmts[0], fmts[1], fmts[2]})
            .UnknownShapeFormat({fmts[0], fmts[1], fmts[2]})
            .AutoContiguous();
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({dtypes[0], dtypes[1], dtypes[2]})
            .Format({fmts[0], fmts[1], fmts[2]})
            .UnknownShapeFormat({fmts[0], fmts[1], fmts[2]});

        this->SetInferShape(ge::FusedSoftmaxGradInferShape).SetInferDataType(ge::FusedSoftmaxGradInferDataType);
        this->AICore().SetTiling(optiling::FusedSoftmaxGradTilingFunc);

        // 执行器标志与已验证稳定的 openfold 交付版 ops-info 完全对齐
        // （五标志全默认 true；不设 aclnnSupport/jitCompile——msopgen 默认
        // 即生成 aclnn 两段式入口，实测该标志组合下 MIX kernel 无 launch 异常）
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true);
        config.DynamicFormatFlag(true);
        config.DynamicRankSupportFlag(true);
        config.DynamicShapeSupportFlag(true);
        config.NeedCheckSupportFlag(false);
        config.PrecisionReduceFlag(true);
        config.ExtendCfgInfo("opFile.value", "fused_softmax_grad");
        config.ExtendCfgInfo("opInterface.value", "fused_softmax_grad");
        this->AICore().AddConfig("ascend910b", config);
    }
};

OP_ADD(FusedSoftmaxGrad);
} // namespace ops
