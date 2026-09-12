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
 * ComplexMul 切分：从 platform 读出资源后委托给 complex_mul_tiling_core.h。
 *
 * 切分算法本身在 core 头里（不依赖 platform_ascendc），由 ophost UT 覆盖。
 */

#ifndef COMPLEX_MUL_TILING_IMPL_H
#define COMPLEX_MUL_TILING_IMPL_H

#include <cstdint>

#include "tiling/platform/platform_ascendc.h"

#include "complex_mul_tiling_core.h"

namespace ComplexMulHost {

inline bool ComputeComplexMulTiling(int64_t totalElems, platform_ascendc::PlatformAscendC& platform,
                                    ComplexMulTilingData& tiling, ComplexMulPlan& plan)
{
    return ComputeComplexMulTilingCore(totalElems, platform.GetCoreNumAiv(),
                                       static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize()), tiling, plan);
}

} // namespace ComplexMulHost

#endif // COMPLEX_MUL_TILING_IMPL_H
