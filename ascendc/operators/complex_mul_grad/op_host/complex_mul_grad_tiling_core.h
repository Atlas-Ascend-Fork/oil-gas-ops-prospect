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
 * ComplexMulGrad 切分算法核心（不依赖 platform_ascendc / gert）。
 * 抽出理由同 complex_mul_tiling_core.h：让 ophost UT 能直接覆盖切分算法。
 * 算法实现与前向共用 og_elementwise_tiling.h，这里只声明本算子的 UB 份数。
 *
 * 方案 C UB 预算：输入 gwr,gwi,ur,ui,vr,vi（6）+ 输出 gur,gui,gvr,gvi（4）+ tmp
 *   2*(6+4)*tileB + 1*tileB = 21*tileB ≤ 176KB → tileFloats ≤ 2048
 */

#ifndef COMPLEX_MUL_GRAD_TILING_CORE_H
#define COMPLEX_MUL_GRAD_TILING_CORE_H

#include <cstdint>

#include "og_elementwise_tiling.h"

#include "../op_kernel/complex_mul_grad_tiling.h"

namespace ComplexMulGradHost {

// inQue depth2 * 6 路 + outQue depth2 * 4 路 + tmp * 1
constexpr int64_t UB_SLOT_FACTOR = 21;

constexpr int64_t ALIGN_FLOATS = OgOpsCommon::ALIGN_FLOATS;
constexpr int64_t UB_BUDGET_BYTES = OgOpsCommon::UB_BUDGET_BYTES;
constexpr int64_t TILE_FLOATS_CAP = (UB_BUDGET_BYTES / (UB_SLOT_FACTOR * static_cast<int64_t>(sizeof(float)))) /
                                    ALIGN_FLOATS * ALIGN_FLOATS;

struct ComplexMulGradPlan {
    uint64_t tilingKey = COMPLEX_MUL_GRAD_TILING_KEY;
    uint32_t blockDim = 0;
    uint64_t totalWorkspace = 0;
};

/** 平台无关的切分核心。aivNum / libApiWorkspace 由调用方从 platform 读出后传入。 */
inline bool ComputeComplexMulGradTilingCore(int64_t totalElems, uint32_t aivNum, uint64_t libApiWorkspace,
                                            ComplexMulGradTilingData& tiling, ComplexMulGradPlan& plan)
{
    return OgOpsCommon::ComputeElementwiseTilingCore(totalElems, aivNum, libApiWorkspace, TILE_FLOATS_CAP,
                                                     COMPLEX_MUL_GRAD_TILING_KEY, tiling, plan);
}

} // namespace ComplexMulGradHost

#endif // COMPLEX_MUL_GRAD_TILING_CORE_H
