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
 * ROIAlignRotated 前向切分算法核心（不依赖 platform_ascendc / gert / CANN 头）。
 *
 * 抽出目的：TilingFunc 依赖 gert::TilingContext、PlatformAscendC，CANN 二进制包
 * 下无法在 UT 中构造；把与平台无关的切分算法隔离到本文件，ophost UT 直接驱动。
 *
 * 基线策略：按输出元素总数 totalOut = K*C*PH*PW 做 64B 对齐 slab 切分。
 * 前向每个输出元素独立，slab 单属主；反向见 roi_align_rotated_grad_tiling_core.h。
 */

#ifndef ROI_ALIGN_ROTATED_TILING_CORE_H
#define ROI_ALIGN_ROTATED_TILING_CORE_H

#include <cstdint>

#include "og_index_tiling.h"

#include "../op_kernel/roi_align_rotated_tiling.h"

namespace RoiAlignRotatedHost {

/** 前向单 slab 元素上限：16KB，给 UB 留出输出 staging 与标量临时空间。 */
constexpr int64_t ROI_ALIGN_ROTATED_SLAB_FLOATS_CAP = 16 * 1024 / static_cast<int64_t>(sizeof(float));

struct RoiAlignRotatedPlan {
    uint64_t tilingKey = ROI_ALIGN_ROTATED_TILING_KEY;
    uint32_t blockDim = 0;
    uint64_t totalWorkspace = 0;
};

/** 形状/属性校验；与 Python / host tiling 入口保持一致。 */
inline bool ValidateRoiAlignRotatedShape(int64_t n, int64_t c, int64_t h, int64_t w, int64_t k, int64_t pooledH,
                                         int64_t pooledW, int64_t samplingRatio, float spatialScale)
{
    if (n < 0 || c < 0 || h < 0 || w < 0 || k < 0) {
        return false;
    }
    if (pooledH <= 0 || pooledW <= 0 || samplingRatio < 0) {
        return false;
    }
    // NaN 也会使比较为 false，从而被拒绝。
    return spatialScale > 0.0F;
}

/**
 * 平台无关切分核心。
 *
 * @param n/c/h/w   input 形状
 * @param k         rois 行数
 * @param pooledH/pooledW  输出 bin 数
 * @param samplingRatio    0 = 自适应采样；>0 = 固定采样
 * @param spatialScale     ROI 坐标缩放
 * @param aivNum           AIV 核数（0 时兜底为 1）
 * @param tiling           输出 tiling 数据
 * @param plan             blockDim / tilingKey / workspace
 * @return false = 形状/属性非法或溢出
 */
inline bool ComputeRoiAlignRotatedTilingCore(int64_t n, int64_t c, int64_t h, int64_t w, int64_t k, int64_t pooledH,
                                             int64_t pooledW, int64_t samplingRatio, float spatialScale,
                                             uint32_t aivNum, ROIAlignRotatedTilingData& tiling,
                                             RoiAlignRotatedPlan& plan)
{
    if (!ValidateRoiAlignRotatedShape(n, c, h, w, k, pooledH, pooledW, samplingRatio, spatialScale)) {
        return false;
    }

    int64_t totalOut = 0;
    if (!OgOpsCommon::IndexMulChecked(k, c, totalOut) || !OgOpsCommon::IndexMulInto(pooledH, totalOut) ||
        !OgOpsCommon::IndexMulInto(pooledW, totalOut)) {
        return false;
    }

    OgOpsCommon::FlatSlabPlan slab;
    if (!OgOpsCommon::ComputeFlatSlabPlan(totalOut, aivNum, ROI_ALIGN_ROTATED_SLAB_FLOATS_CAP, slab)) {
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
    tiling.totalOut = slab.total;
    tiling.slabFloats = slab.slabFloats;
    tiling.numSlabs = slab.numSlabs;
    tiling.blockDim = static_cast<int64_t>(slab.blockDim);
    tiling.baseSlabs = slab.baseSlabs;
    tiling.remSlabs = slab.remSlabs;
    tiling.spatialScale = spatialScale;

    plan.tilingKey = ROI_ALIGN_ROTATED_TILING_KEY;
    plan.blockDim = slab.blockDim;
    plan.totalWorkspace = 0;
    return true;
}

} // namespace RoiAlignRotatedHost

#endif // ROI_ALIGN_ROTATED_TILING_CORE_H
