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
