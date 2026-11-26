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
 * ROIAlignRotated 反向 tiling POD（与 host BEGIN_TILING_DATA_DEF 字段顺序一致）。
 *
 *   grad_output [K,C,PH,PW] + rois [K,6] -> grad_input [N,C,H,W]
 *
 * grad_input 按 64B slab 独占切分：每核在 UB 内清零、累加、整片回写，
 * 不用 atomicAdd，结果确定。
 */

#ifndef ROI_ALIGN_ROTATED_GRAD_TILING_H
#define ROI_ALIGN_ROTATED_GRAD_TILING_H

#include <cstdint>

constexpr uint64_t ROI_ALIGN_ROTATED_GRAD_TILING_KEY = 0;

struct ROIAlignRotatedGradTilingData {
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    int64_t k;
    int64_t pooledH;
    int64_t pooledW;
    int64_t samplingRatio;
    int64_t totalIn;
    int64_t slabFloats;
    int64_t numSlabs;
    int64_t blockDim;
    int64_t baseSlabs;
    int64_t remSlabs;
    float spatialScale;
};

#endif // ROI_ALIGN_ROTATED_GRAD_TILING_H
