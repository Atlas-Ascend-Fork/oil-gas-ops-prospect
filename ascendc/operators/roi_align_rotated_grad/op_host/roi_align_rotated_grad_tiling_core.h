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
 * ROIAlignRotated 反向切分算法核心（不依赖 platform_ascendc / gert / CANN 头）。
 *
 * 反向属于“写索引型（Scatter 系）”：多个输出 bin 会命中同一个 input 像素。
 * 910B3 上不能用 atomicAdd 直接散射写 GM（跨核同 64B line 会丢 sector）。
 * 本文件按 grad_input 元素空间做 64B 对齐 slab 切分，每核独占连续整块；kernel
 * 在 UB 内清零 + 累加 + DataCopyPad 整片回写。
 */

#ifndef ROI_ALIGN_ROTATED_GRAD_TILING_CORE_H
#define ROI_ALIGN_ROTATED_GRAD_TILING_CORE_H

#include <cstdint>

#include "og_index_tiling.h"

#include "../op_kernel/roi_align_rotated_grad_tiling.h"

namespace RoiAlignRotatedGradHost {

/** 反向单 slab 元素上限：16KB，给 UB 留出清零/累加/回写的临时空间。 */
constexpr int64_t ROI_ALIGN_ROTATED_GRAD_SLAB_FLOATS_CAP = 16 * 1024 / static_cast<int64_t>(sizeof(float));

struct RoiAlignRotatedGradPlan {
    uint64_t tilingKey = ROI_ALIGN_ROTATED_GRAD_TILING_KEY;
    uint32_t blockDim = 0;
    uint64_t totalWorkspace = 0;
};

inline bool ValidateRoiAlignRotatedGradShape(int64_t n, int64_t c, int64_t h, int64_t w, int64_t k, int64_t pooledH,
                                             int64_t pooledW, int64_t samplingRatio, float spatialScale)
{
    if (n < 0 || c < 0 || h < 0 || w < 0 || k < 0) {
        return false;
    }
    if (pooledH <= 0 || pooledW <= 0 || samplingRatio < 0) {
        return false;
    }
    return spatialScale > 0.0F;
}

/**
 * 平台无关切分核心。
 *
 * totalIn = N*C*H*W，按 64B slab 切分。最后一个 slab 可以短于 slabFloats。
 * 每个 slab 可能跨多个 (n,c) 平面，kernel 侧从 slab 的 flat [start,end) 反推
 * n 范围，并对 ROI 的 batch index 做过滤；贡献的 flat index 必须落在本 slab 内。
 */
inline bool ComputeRoiAlignRotatedGradTilingCore(int64_t n, int64_t c, int64_t h, int64_t w, int64_t k, int64_t pooledH,
                                                 int64_t pooledW, int64_t samplingRatio, float spatialScale,
                                                 uint32_t aivNum, ROIAlignRotatedGradTilingData& tiling,
                                                 RoiAlignRotatedGradPlan& plan)
{
    if (!ValidateRoiAlignRotatedGradShape(n, c, h, w, k, pooledH, pooledW, samplingRatio, spatialScale)) {
        return false;
    }

    int64_t totalIn = 0;
    if (!OgOpsCommon::IndexMulChecked(n, c, totalIn) || !OgOpsCommon::IndexMulInto(h, totalIn) ||
        !OgOpsCommon::IndexMulInto(w, totalIn)) {
        return false;
    }

    OgOpsCommon::FlatSlabPlan slab;
    if (!OgOpsCommon::ComputeFlatSlabPlan(totalIn, aivNum, ROI_ALIGN_ROTATED_GRAD_SLAB_FLOATS_CAP, slab)) {
        return false;
    }

    tiling.n = n;
    tiling.c = c;
    tiling.h = h;
    tiling.w = w;
    tiling.k = k;
    tiling.pooledH = pooledH;
    tiling.pooledW = pooledW;
    tiling.samplingRatio = samplingRatio;
    tiling.totalIn = slab.total;
    tiling.slabFloats = slab.slabFloats;
    tiling.numSlabs = slab.numSlabs;
    tiling.blockDim = static_cast<int64_t>(slab.blockDim);
    tiling.baseSlabs = slab.baseSlabs;
    tiling.remSlabs = slab.remSlabs;
    tiling.spatialScale = spatialScale;

    plan.tilingKey = ROI_ALIGN_ROTATED_GRAD_TILING_KEY;
    plan.blockDim = slab.blockDim;
    plan.totalWorkspace = 0;
    return true;
}

} // namespace RoiAlignRotatedGradHost

#endif // ROI_ALIGN_ROTATED_GRAD_TILING_CORE_H
