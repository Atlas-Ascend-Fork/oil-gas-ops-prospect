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
 * ComplexMulOg host：CBS op_v 复数逐点乘。
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

static ge::graphStatus ComplexMulTilingFunc(gert::TilingContext *context)
{
    return OgOpsCommon::RunElementwiseTiling<ComplexMulTilingData, ComplexMulHost::ComplexMulPlan>(
        context, kShapeInput, ComplexMulHost::ComputeComplexMulTiling);
}

} // namespace optiling

namespace ge {

static ge::graphStatus ComplexMulInferShape(gert::InferShapeContext *context)
{
    return OgOpsCommon::InferSameShapeOutputs(context, kShapeInput, kOutputCount);
}

static ge::graphStatus ComplexMulInferDataType(gert::InferDataTypeContext *context)
{
    return OgOpsCommon::InferFloatOutputs(context, kOutputCount);
}

} // namespace ge

namespace ops {

class ComplexMulOg : public OpDef {
public:
    explicit ComplexMulOg(const char *name) : OpDef(name)
    {
        DeclareRequiredFloatNdIo(*this, {"ur", "ui", "vr", "vi"}, {"wr", "wi"});
        this->SetInferShape(ge::ComplexMulInferShape).SetInferDataType(ge::ComplexMulInferDataType);
        this->AICore().SetTiling(optiling::ComplexMulTilingFunc);
        AddAscend910bElementwiseConfig(*this, "complex_mul");
    }
};

OP_ADD(ComplexMulOg);
} // namespace ops
