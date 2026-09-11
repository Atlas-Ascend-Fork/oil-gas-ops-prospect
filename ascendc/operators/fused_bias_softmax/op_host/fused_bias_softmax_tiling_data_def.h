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
 * FusedBiasSoftmaxOg tiling 数据结构（host 侧，BEGIN_TILING_DATA_DEF）。
 * 字段顺序与 op_kernel 侧 GET_TILING_DATA 的 device 镜像一致。
 */

#ifndef FUSED_BIAS_SOFTMAX_TILING_DATA_DEF_H
#define FUSED_BIAS_SOFTMAX_TILING_DATA_DEF_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedBiasSoftmaxTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, coreNum);

    TILING_DATA_FIELD_DEF(uint64_t, batch);   // 输入 x 的第 1 维大小
    TILING_DATA_FIELD_DEF(uint64_t, channel); // 输入 x 的第 2 维大小
    TILING_DATA_FIELD_DEF(uint64_t, height);  // 输入 x 的第 3 维大小（= qPerPlane）
    TILING_DATA_FIELD_DEF(uint64_t, width);   // 输入 x 的第 4 维大小

    TILING_DATA_FIELD_DEF(uint64_t, hasB1);   // bias1 是否存在
    TILING_DATA_FIELD_DEF(uint64_t, hasB2);   // bias2 是否存在
    TILING_DATA_FIELD_DEF(uint64_t, b1Full);  // bias1 为全尺寸（非广播）标志
    TILING_DATA_FIELD_DEF(uint64_t, qPerPlane);

    TILING_DATA_FIELD_DEF(uint64_t, paddingNum); // 对齐需要补充的数据个数
    TILING_DATA_FIELD_DEF(uint64_t, padLineNum); // 对齐后的一行长度

    TILING_DATA_FIELD_DEF(uint64_t, headCoreNum);

    TILING_DATA_FIELD_DEF(uint64_t, lineHeadCore);
    TILING_DATA_FIELD_DEF(uint64_t, iterHeadCore);
    TILING_DATA_FIELD_DEF(uint64_t, lineHeadIter);
    TILING_DATA_FIELD_DEF(uint64_t, lineLastHeadIter);

    TILING_DATA_FIELD_DEF(uint64_t, lineTailCore);
    TILING_DATA_FIELD_DEF(uint64_t, iterTailCore);
    TILING_DATA_FIELD_DEF(uint64_t, lineTailIter);
    TILING_DATA_FIELD_DEF(uint64_t, lineLastTailIter);

    TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData); // SoftMax 高阶 api 所需 tiling

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedBiasSoftmaxOg, FusedBiasSoftmaxTilingData)
} // namespace optiling

#endif // FUSED_BIAS_SOFTMAX_TILING_DATA_DEF_H
