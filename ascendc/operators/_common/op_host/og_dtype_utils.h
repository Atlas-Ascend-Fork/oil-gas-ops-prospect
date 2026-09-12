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
 * 融合算子族共用的 dtype 映射与维度常量（消除各 tiling.cpp 间的重复代码）。
 */

#ifndef OG_DTYPE_UTILS_H
#define OG_DTYPE_UTILS_H

#include <cstdint>

#include "register/op_def_registry.h"

namespace OgOpsCommon {

constexpr uint64_t kOgDim0 = 0;
constexpr uint64_t kOgDim1 = 1;
constexpr uint64_t kOgDim2 = 2;
constexpr uint64_t kOgDim3 = 3;
constexpr uint64_t kOgDimNum = 4;

/** 元素字节数：fp32→4，fp16/bf16→2，其他→0（非法）。 */
inline uint64_t GetOgDtypeSize(int32_t dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return 4;
        case ge::DT_BF16:
        case ge::DT_FLOAT16:
            return 2;
        default:
            return 0;
    }
}

/** dtype key：fp32→0，fp16→1，bf16→2（与各 kernel 侧 TILING_KEY 约定一致）。 */
inline uint64_t GetOgDtypeKey(int32_t dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return 0;
        case ge::DT_FLOAT16:
            return 1;
        default:
            return 2;
    }
}

} // namespace OgOpsCommon

#endif // OG_DTYPE_UTILS_H
