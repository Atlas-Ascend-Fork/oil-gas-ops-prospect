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
 * ROIAlignRotatedGrad host 侧 tiling 数据结构（与 op_kernel 侧 plain struct 一致）。
 */

#ifndef ROI_ALIGN_ROTATED_GRAD_TILING_DATA_DEF_H
#define ROI_ALIGN_ROTATED_GRAD_TILING_DATA_DEF_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ROIAlignRotatedGradTilingDataDef)
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, c);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, w);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, pooledH);
TILING_DATA_FIELD_DEF(int64_t, pooledW);
TILING_DATA_FIELD_DEF(int64_t, samplingRatio);
TILING_DATA_FIELD_DEF(int64_t, totalIn);
TILING_DATA_FIELD_DEF(int64_t, slabFloats);
TILING_DATA_FIELD_DEF(int64_t, numSlabs);
TILING_DATA_FIELD_DEF(int64_t, blockDim);
TILING_DATA_FIELD_DEF(int64_t, baseSlabs);
TILING_DATA_FIELD_DEF(int64_t, remSlabs);
TILING_DATA_FIELD_DEF(float, spatialScale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ROIAlignRotatedGrad, ROIAlignRotatedGradTilingDataDef)

} // namespace optiling

#endif // ROI_ALIGN_ROTATED_GRAD_TILING_DATA_DEF_H
