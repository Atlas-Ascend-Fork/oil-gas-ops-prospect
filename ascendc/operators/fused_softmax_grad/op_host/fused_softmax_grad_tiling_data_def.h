/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * FusedSoftmaxGradOg tiling 数据结构（host 侧）。
 * MIX 路径内嵌 TCubeTiling（高阶 Matmul）；device 侧经 GET_TILING_DATA 读取。
 */

#ifndef FUSED_SOFTMAX_GRAD_TILING_DATA_DEF_H
#define FUSED_SOFTMAX_GRAD_TILING_DATA_DEF_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedSoftmaxGradTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, coreNum);  // SetBlockDim（= aicNum, 24）；AIV 数 = coreNum*2
    TILING_DATA_FIELD_DEF(uint64_t, batch);
    TILING_DATA_FIELD_DEF(uint64_t, channel);
    TILING_DATA_FIELD_DEF(uint64_t, height);   // Q（Q≠S 合法，openfold MSA/三角注意力形态）
    TILING_DATA_FIELD_DEF(uint64_t, width);    // S
    TILING_DATA_FIELD_DEF(uint64_t, dLen);     // D（gradOutput/values 末维）
    TILING_DATA_FIELD_DEF(uint64_t, planes);   // B*H

    TILING_DATA_FIELD_DEF(uint64_t, useMix);       // 1 = Cube+Vector MIX 路径；0 = 纯向量路径
    TILING_DATA_FIELD_DEF(uint64_t, padS);         // S 对齐后行宽（fp32 路径 8 对齐 / 2B 路径 16 对齐）
    TILING_DATA_FIELD_DEF(uint64_t, padD);         // D 对齐后行宽（同上）
    TILING_DATA_FIELD_DEF(uint64_t, rowsPerIter);  // MIX 路径 AIV 每迭代行数（UB 预算）

    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingData); // 高阶 Matmul tiling（MIX 路径）

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedSoftmaxGradOg, FusedSoftmaxGradTilingData)
} // namespace optiling

#endif // FUSED_SOFTMAX_GRAD_TILING_DATA_DEF_H
