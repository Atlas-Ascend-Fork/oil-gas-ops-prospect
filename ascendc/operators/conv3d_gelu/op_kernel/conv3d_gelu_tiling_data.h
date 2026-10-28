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
 * Conv3DGelu 组合 tiling 数据。
 *
 * 前半部分为 conv3d_v2 的完整 tiling（cube 阶段直接复用），后半为 gelu 向量阶段
 * 的逐元素切分参数（AIV 核按 totalNum 均分）。host 与 kernel 共用本结构。
 */

#ifndef CONV3D_GELU_TILING_DATA_H
#define CONV3D_GELU_TILING_DATA_H

#include <cstdint>
#include "conv3d_v2_tiling_data.h"

#pragma pack(push, 8)

struct Conv3DGeluTilingData {
    Ops::NN::Conv3dV2::Conv3DV2TilingData convTiling;
    // gelu 向量阶段：对 conv 输出（NDC1HWC0 存储）逐元素做激活
    int64_t geluTotalNum = 0;    // 输出张量的存储元素总数
    int64_t geluBlockFactor = 0; // 每个 AIV 核处理的元素数（host 按 aivNum 均分）
    int64_t geluUbFactor = 0;    // 单次 UB 处理的元素数
};

#pragma pack(pop)

#endif // CONV3D_GELU_TILING_DATA_H
