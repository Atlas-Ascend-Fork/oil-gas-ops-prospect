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
 * ROIAlignRotated 前向 tiling POD（与 host BEGIN_TILING_DATA_DEF 字段顺序一致）。
 *
 *   input [N,C,H,W] + rois [K,6] -> out [K,C,PH,PW]
 *
 * rois 每行: (batch_idx, x_ctr, y_ctr, width, height, angle_deg)。
 * 输出元素空间按 64B slab 切分，kernel 侧用 FlatSlabPlan 的同一组字段。
 */

#ifndef ROI_ALIGN_ROTATED_TILING_H
#define ROI_ALIGN_ROTATED_TILING_H

#include <cstdint>

constexpr uint64_t ROI_ALIGN_ROTATED_TILING_KEY = 0;

struct ROIAlignRotatedTilingData {
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    int64_t k;
    int64_t pooledH;
    int64_t pooledW;
    int64_t samplingRatio;
    int64_t totalOut;
    int64_t slabFloats;
    int64_t numSlabs;
    int64_t blockDim;
    int64_t baseSlabs;
    int64_t remSlabs;
    float spatialScale;
};

#endif // ROI_ALIGN_ROTATED_TILING_H
