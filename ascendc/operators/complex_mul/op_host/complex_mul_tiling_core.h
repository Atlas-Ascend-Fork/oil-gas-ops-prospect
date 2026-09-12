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
 * ComplexMul 切分算法核心（不依赖 platform_ascendc / gert）。
 *
 * 抽出目的：tiling 入口 ComplexMulTilingFunc 需要 gert::TilingContext，而
 * ComputeComplexMulTiling 需要 platform_ascendc::PlatformAscendC，两者在
 * CANN 二进制包下都无法在 UT 里构造（无 gert faker）。把与平台无关的切分算法
 * 隔离到本文件后，ophost UT 可以直接覆盖（见 tests/ascendc/ophost/）。
 *
 * 算法本身与 complex_mul_grad 共用，实现在 og_elementwise_tiling.h；本文件只
 * 声明本算子的 UB 份数。
 *
 * 方案 C（inQue 拼 4 路 + outQue 拼 2 路，double buffer）UB 预算：
 *   2*(4+2)*tileB + 1*tileB(tmp) = 13*tileB ≤ 176KB
 *   → tileFloats ≤ 176*1024/(13*4) ≈ 3464，按 32B 对齐向下取整
 */

#ifndef COMPLEX_MUL_TILING_CORE_H
#define COMPLEX_MUL_TILING_CORE_H

#include <cstdint>

#include "og_elementwise_tiling.h"

#include "../op_kernel/complex_mul_tiling.h"

namespace ComplexMulHost {

// inQue depth2 * 4 路 + outQue depth2 * 2 路 + tmp * 1
constexpr int64_t UB_SLOT_FACTOR = 13;

constexpr int64_t ALIGN_FLOATS = OgOpsCommon::ALIGN_FLOATS;
constexpr int64_t UB_BUDGET_BYTES = OgOpsCommon::UB_BUDGET_BYTES;
constexpr int64_t TILE_FLOATS_CAP = (UB_BUDGET_BYTES / (UB_SLOT_FACTOR * static_cast<int64_t>(sizeof(float)))) /
                                    ALIGN_FLOATS * ALIGN_FLOATS;

struct ComplexMulPlan {
    uint64_t tilingKey = COMPLEX_MUL_TILING_KEY;
    uint32_t blockDim = 0;
    uint64_t totalWorkspace = 0;
};

/**
 * 平台无关的切分核心。aivNum / libApiWorkspace 由调用方从 platform 读出后传入，
 * 便于 UT 直接构造。
 */
inline bool ComputeComplexMulTilingCore(int64_t totalElems, uint32_t aivNum, uint64_t libApiWorkspace,
                                        ComplexMulTilingData& tiling, ComplexMulPlan& plan)
{
    return OgOpsCommon::ComputeElementwiseTilingCore(totalElems, aivNum, libApiWorkspace, TILE_FLOATS_CAP,
                                                     COMPLEX_MUL_TILING_KEY, tiling, plan);
}

} // namespace ComplexMulHost

#endif // COMPLEX_MUL_TILING_CORE_H
