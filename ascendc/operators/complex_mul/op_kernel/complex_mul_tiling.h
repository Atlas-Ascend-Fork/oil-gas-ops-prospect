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
 * ComplexMulOg tiling POD（与 host BEGIN_TILING_DATA_DEF 字段顺序一致）。
 *
 * 语义：复数逐点乘（CBS op_v）
 *   wr = ur*vr - ui*vi
 *   wi = ur*vi + ui*vr
 */

#ifndef COMPLEX_MUL_TILING_H
#define COMPLEX_MUL_TILING_H

#include <cstdint>

// 单路径 AIV 算子默认编译 key=0（kernelList 后缀 _0）
constexpr uint64_t COMPLEX_MUL_TILING_KEY = 0;

struct ComplexMulTilingData {
    int64_t totalElems;   // 每路张量元素总数
    int64_t tileFloats;   // 单路单拍 float 数（32B 对齐）
    int64_t coreNum;      // AIV 核数
};

#endif // COMPLEX_MUL_TILING_H
