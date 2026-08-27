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
 * ComplexMulOg 切分：从 platform 读出资源后委托给 complex_mul_tiling_core.h。
 *
 * 切分算法本身在 core 头里（不依赖 platform_ascendc），由 ophost UT 覆盖。
 */

#ifndef COMPLEX_MUL_TILING_IMPL_H
#define COMPLEX_MUL_TILING_IMPL_H

#include <cstdint>

#include "tiling/platform/platform_ascendc.h"

#include "complex_mul_tiling_core.h"

namespace ComplexMulHost {

inline bool ComputeComplexMulTiling(int64_t totalElems, platform_ascendc::PlatformAscendC &platform,
                                    ComplexMulTilingData &tiling, ComplexMulPlan &plan)
{
    return ComputeComplexMulTilingCore(totalElems, platform.GetCoreNumAiv(),
                                       static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize()), tiling, plan);
}

} // namespace ComplexMulHost

#endif // COMPLEX_MUL_TILING_IMPL_H
