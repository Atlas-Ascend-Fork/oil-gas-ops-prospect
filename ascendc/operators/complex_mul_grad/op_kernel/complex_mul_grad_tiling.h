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
