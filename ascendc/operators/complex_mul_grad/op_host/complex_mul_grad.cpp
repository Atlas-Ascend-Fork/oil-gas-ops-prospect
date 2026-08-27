/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * ComplexMulGradOg host：ComplexMul 反向。
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

static ge::graphStatus ComplexMulGradTilingFunc(gert::TilingContext *context)
{
    return OgOpsCommon::RunElementwiseTiling<ComplexMulGradTilingData, ComplexMulGradHost::ComplexMulGradPlan>(
        context, kShapeInput, ComplexMulGradHost::ComputeComplexMulGradTiling);
}

} // namespace optiling

namespace ge {

static ge::graphStatus ComplexMulGradInferShape(gert::InferShapeContext *context)
{
    return OgOpsCommon::InferSameShapeOutputs(context, kShapeInput, kOutputCount);
}

static ge::graphStatus ComplexMulGradInferDataType(gert::InferDataTypeContext *context)
{
    return OgOpsCommon::InferFloatOutputs(context, kOutputCount);
}

} // namespace ge

namespace ops {

class ComplexMulGradOg : public OpDef {
public:
    explicit ComplexMulGradOg(const char *name) : OpDef(name)
    {
        DeclareRequiredFloatNdIo(
            *this, {"gwr", "gwi", "ur", "ui", "vr", "vi"}, {"gur", "gui", "gvr", "gvi"});
        this->SetInferShape(ge::ComplexMulGradInferShape).SetInferDataType(ge::ComplexMulGradInferDataType);
        this->AICore().SetTiling(optiling::ComplexMulGradTilingFunc);
        AddAscend910bElementwiseConfig(*this, "complex_mul_grad");
    }
};

OP_ADD(ComplexMulGradOg);
} // namespace ops
