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
 * Conv3DGelu 路径选择。host tiling 与 aclnn 必须用同一套条件。
 *
 * 1×1 直写只在实测快过私有格式往返时启用：Cin 太小或 Cout 太大时，
 * NCDHW pointwise cube 比 CANN conv3d 慢。
 * 小通道向量直算已实现但逐行 Gather 比 CANN conv3d 慢数倍，默认关闭。
 */

#ifndef CONV3D_GELU_ROUTE_H
#define CONV3D_GELU_ROUTE_H

#include <cstdint>

inline bool Conv3dGeluUseDirect1x1(int64_t cin, int64_t cout, int64_t groups, int64_t kd, int64_t kh, int64_t kw,
                                   int64_t sd, int64_t sh, int64_t sw, int64_t dd, int64_t dh, int64_t dw,
                                   const int64_t pads[6])
{
    if (groups != 1 || kd != 1 || kh != 1 || kw != 1) {
        return false;
    }
    if (sd != 1 || sh != 1 || sw != 1 || dd != 1 || dh != 1 || dw != 1) {
        return false;
    }
    for (int i = 0; i < 6; ++i) {
        if (pads[i] != 0) {
            return false;
        }
    }
    // enc 1→32（Cin=1）和 latent 256→512（Cout=512）直写会慢于拆开的 conv。
    return cin >= 8 && cout <= 64;
}

// 全分辨率、通道很小的 3×3 用 HW 按行切。k≠3 或通道更大时实测更慢，不走这条。
// L1 是否装得下由 tiling 引擎再判，这里只放 shape 条件。
inline bool Conv3dGeluPreferHwMode(bool isPointWise, int64_t groups, int64_t kd, int64_t kh, int64_t kw, int64_t cin,
                                   int64_t cout, int64_t ho, int64_t wo)
{
    return !isPointWise && groups == 1 && kd == 3 && kh == 3 && kw == 3 && cin <= 16 && cout <= 16 && ho >= 128 &&
           wo >= 128;
}

#endif
