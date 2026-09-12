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
 * ComplexMul host：CBS op_v 复数逐点乘。
 *   wr = ur*vr - ui*vi
 *   wi = ur*vi + ui*vr
 */

#include <cstdint>

#include "register/op_def_registry.h"

#include "og_tiling_context.h"

#include "complex_mul_tiling_data_def.h"
#include "complex_mul_tiling_impl.h"
#include "../op_kernel/complex_mul_tiling.h"

namespace {
constexpr uint32_t kShapeInput = 0;
constexpr uint32_t kOutputCount = 2;
} // namespace

namespace optiling {

static ge::graphStatus ComplexMulTilingFunc(gert::TilingContext* context)
{
    return OgOpsCommon::RunElementwiseTiling<ComplexMulTilingData, ComplexMulHost::ComplexMulPlan>(
        context, kShapeInput, ComplexMulHost::ComputeComplexMulTiling);
}

} // namespace optiling

namespace ge {

static ge::graphStatus ComplexMulInferShape(gert::InferShapeContext* context)
{
    return OgOpsCommon::InferSameShapeOutputs(context, kShapeInput, kOutputCount);
}

static ge::graphStatus ComplexMulInferDataType(gert::InferDataTypeContext* context)
{
    return OgOpsCommon::InferFloatOutputs(context, kOutputCount);
}

} // namespace ge

namespace ops {

class ComplexMul : public OpDef {
public:
    explicit ComplexMul(const char* name) : OpDef(name)
    {
        DeclareRequiredFloatNdIo(*this, {"ur", "ui", "vr", "vi"}, {"wr", "wi"});
        this->SetInferShape(ge::ComplexMulInferShape).SetInferDataType(ge::ComplexMulInferDataType);
        this->AICore().SetTiling(optiling::ComplexMulTilingFunc);
        AddAscend910bElementwiseConfig(*this, "complex_mul");
    }
};

OP_ADD(ComplexMul);
} // namespace ops
