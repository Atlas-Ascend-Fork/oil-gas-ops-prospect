/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPLEX_MUL_GRAD_TILING_H
#define COMPLEX_MUL_GRAD_TILING_H

#include <cstdint>

constexpr uint64_t COMPLEX_MUL_GRAD_TILING_KEY = 0;

struct ComplexMulGradTilingData {
    int64_t totalElems;
    int64_t tileFloats;
    int64_t coreNum;
};

#endif // COMPLEX_MUL_GRAD_TILING_H
