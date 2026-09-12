/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
 * ComplexMulGrad host：ComplexMul 反向。
 */

#include <cstdint>

#include "register/op_def_registry.h"

#include "og_tiling_context.h"

#include "complex_mul_grad_tiling_data_def.h"
#include "complex_mul_grad_tiling_impl.h"
#include "../op_kernel/complex_mul_grad_tiling.h"

namespace {
constexpr uint32_t kShapeInput = 0;
constexpr uint32_t kOutputCount = 4;
} // namespace

namespace optiling {

static ge::graphStatus ComplexMulGradTilingFunc(gert::TilingContext* context)
{
    return OgOpsCommon::RunElementwiseTiling<ComplexMulGradTilingData, ComplexMulGradHost::ComplexMulGradPlan>(
        context, kShapeInput, ComplexMulGradHost::ComputeComplexMulGradTiling);
}

} // namespace optiling

namespace ge {

static ge::graphStatus ComplexMulGradInferShape(gert::InferShapeContext* context)
{
    return OgOpsCommon::InferSameShapeOutputs(context, kShapeInput, kOutputCount);
}

static ge::graphStatus ComplexMulGradInferDataType(gert::InferDataTypeContext* context)
{
    return OgOpsCommon::InferFloatOutputs(context, kOutputCount);
}

} // namespace ge

namespace ops {

class ComplexMulGrad : public OpDef {
public:
    explicit ComplexMulGrad(const char* name) : OpDef(name)
    {
        DeclareRequiredFloatNdIo(*this, {"gwr", "gwi", "ur", "ui", "vr", "vi"}, {"gur", "gui", "gvr", "gvi"});
        this->SetInferShape(ge::ComplexMulGradInferShape).SetInferDataType(ge::ComplexMulGradInferDataType);
        this->AICore().SetTiling(optiling::ComplexMulGradTilingFunc);
        AddAscend910bElementwiseConfig(*this, "complex_mul_grad");
    }
};

OP_ADD(ComplexMulGrad);
} // namespace ops
