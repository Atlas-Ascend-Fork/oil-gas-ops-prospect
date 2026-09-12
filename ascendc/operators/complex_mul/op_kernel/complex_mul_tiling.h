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
 * ComplexMul tiling POD（与 host BEGIN_TILING_DATA_DEF 字段顺序一致）。
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
    int64_t totalElems; // 每路张量元素总数
    int64_t tileFloats; // 单路单拍 float 数（32B 对齐）
    int64_t coreNum;    // AIV 核数
};

#endif // COMPLEX_MUL_TILING_H
