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
 * FusedSoftmaxGrad tiling 数据结构（host 侧）。
 * MIX 路径内嵌 TCubeTiling（高阶 Matmul）；device 侧经 GET_TILING_DATA 读取。
 */

#ifndef FUSED_SOFTMAX_GRAD_TILING_DATA_DEF_H
#define FUSED_SOFTMAX_GRAD_TILING_DATA_DEF_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedSoftmaxGradTilingData)
TILING_DATA_FIELD_DEF(uint64_t, coreNum); // SetBlockDim（= aicNum, 24）；AIV 数 = coreNum*2
TILING_DATA_FIELD_DEF(uint64_t, batch);
TILING_DATA_FIELD_DEF(uint64_t, channel);
TILING_DATA_FIELD_DEF(uint64_t, height); // Q（Q≠S 合法，openfold MSA/三角注意力形态）
TILING_DATA_FIELD_DEF(uint64_t, width);  // S
TILING_DATA_FIELD_DEF(uint64_t, dLen);   // D（gradOutput/values 末维）
TILING_DATA_FIELD_DEF(uint64_t, planes); // B*H

TILING_DATA_FIELD_DEF(uint64_t, useMix);      // 1 = Cube+Vector MIX 路径；0 = 纯向量路径
TILING_DATA_FIELD_DEF(uint64_t, padS);        // S 对齐后行宽（fp32 路径 8 对齐 / 2B 路径 16 对齐）
TILING_DATA_FIELD_DEF(uint64_t, padD);        // D 对齐后行宽（同上）
TILING_DATA_FIELD_DEF(uint64_t, rowsPerIter); // MIX 路径 AIV 每迭代行数（UB 预算）

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingData); // 高阶 Matmul tiling（MIX 路径）

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedSoftmaxGrad, FusedSoftmaxGradTilingData)
} // namespace optiling

#endif // FUSED_SOFTMAX_GRAD_TILING_DATA_DEF_H
