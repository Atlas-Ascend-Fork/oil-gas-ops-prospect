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
 * 索引/插值类 AIV kernel 的公共工具：
 *   - tiling POD 从 GM 拷到片上；
 *   - 标量 floor/ceil（ccec 下不能依赖 libm 的 floorf/ceilf）；
 *   - ROIAlign 双线性插值与反向权重；
 *
 * 只在 kernel 侧使用；host 侧对应逻辑在 *_tiling_core.h 与纯 torch 参考实现里。
 */

#ifndef OG_INDEX_KERNEL_H
#define OG_INDEX_KERNEL_H

#include "kernel_operator.h"

namespace OgIndex {

using namespace AscendC;

constexpr float kPi = 3.14159265358979323846F;

/** 把 GM 上的 tiling POD 按 uint32 逐字读进片上结构体。 */
template <typename TilingT>
__aicore__ inline void CopyTilingData(TilingT* dst, GM_ADDR tilingGm)
{
    auto* out = reinterpret_cast<uint32_t*>(dst);
    auto* src = reinterpret_cast<__gm__ uint32_t*>(tilingGm);
    for (uint32_t i = 0; i < sizeof(TilingT) / sizeof(uint32_t); ++i) {
        out[i] = src[i];
    }
}

/** float -> floor(int64)。非负区间与截断一致；负数向下修正。 */
__aicore__ inline int64_t FloorToInt(float v)
{
    int64_t i = static_cast<int64_t>(v);
    if (v < 0.0F && static_cast<float>(i) != v) {
        --i;
    }
    return i;
}

/** float -> ceil(int64)。正数向上修正；负数与截断一致。 */
__aicore__ inline int64_t CeilToInt(float v)
{
    int64_t i = static_cast<int64_t>(v);
    if (v > 0.0F && static_cast<float>(i) != v) {
        ++i;
    }
    return i;
}

__aicore__ inline int64_t MaxInt64(int64_t a, int64_t b) { return a > b ? a : b; }

__aicore__ inline int64_t MinInt64(int64_t a, int64_t b) { return a < b ? a : b; }

/**
 * ROIAlign 双线性插值，严格对齐 detectron2 ROIAlignRotated CUDA：
 *   - y < -1 || y > H || x < -1 || x > W -> 0；
 *   - y/x < 0 夹到 0；
 *   - y_low >= H-1 时 y_low = y_high = H-1 且 y = y_low；x 同理；
 *   - 4 邻域权重 hy*hx, hy*lx, ly*hx, ly*lx。
 *
 * @param base  input[n,c] 平面的起始 offset
 */
template <typename GlobalT>
__aicore__ inline float Bilinear(const GlobalT& input, int64_t base, int64_t height, int64_t width, float y, float x)
{
    if (height <= 0 || width <= 0) {
        return 0.0F;
    }
    if (y < -1.0F || y > static_cast<float>(height) || x < -1.0F || x > static_cast<float>(width)) {
        return 0.0F;
    }
    if (y < 0.0F) {
        y = 0.0F;
    }
    if (x < 0.0F) {
        x = 0.0F;
    }

    int64_t yLow = static_cast<int64_t>(y);
    int64_t xLow = static_cast<int64_t>(x);
    int64_t yHigh = 0;
    int64_t xHigh = 0;
    if (yLow >= height - 1) {
        yHigh = yLow = height - 1;
        y = static_cast<float>(yLow);
    } else {
        yHigh = yLow + 1;
    }
    if (xLow >= width - 1) {
        xHigh = xLow = width - 1;
        x = static_cast<float>(xLow);
    } else {
        xHigh = xLow + 1;
    }

    const float ly = y - static_cast<float>(yLow);
    const float lx = x - static_cast<float>(xLow);
    const float hy = 1.0F - ly;
    const float hx = 1.0F - lx;
    const float v1 = input.GetValue(base + yLow * width + xLow);
    const float v2 = input.GetValue(base + yLow * width + xHigh);
    const float v3 = input.GetValue(base + yHigh * width + xLow);
    const float v4 = input.GetValue(base + yHigh * width + xHigh);
    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

/** ROIAlign 反向的 4 邻域权重与索引。valid=false 时索引为 -1、权重为 0。 */
struct BilinearGrad {
    float w1 = 0.0F;
    float w2 = 0.0F;
    float w3 = 0.0F;
    float w4 = 0.0F;
    int64_t xLow = -1;
    int64_t xHigh = -1;
    int64_t yLow = -1;
    int64_t yHigh = -1;
    bool valid = false;
};

__aicore__ inline BilinearGrad ComputeBilinearGrad(int64_t height, int64_t width, float y, float x)
{
    BilinearGrad result;
    if (height <= 0 || width <= 0) {
        return result;
    }
    if (y < -1.0F || y > static_cast<float>(height) || x < -1.0F || x > static_cast<float>(width)) {
        return result;
    }
    if (y < 0.0F) {
        y = 0.0F;
    }
    if (x < 0.0F) {
        x = 0.0F;
    }

    result.yLow = static_cast<int64_t>(y);
    result.xLow = static_cast<int64_t>(x);
    if (result.yLow >= height - 1) {
        result.yHigh = result.yLow = height - 1;
        y = static_cast<float>(result.yLow);
    } else {
        result.yHigh = result.yLow + 1;
    }
    if (result.xLow >= width - 1) {
        result.xHigh = result.xLow = width - 1;
        x = static_cast<float>(result.xLow);
    } else {
        result.xHigh = result.xLow + 1;
    }

    const float ly = y - static_cast<float>(result.yLow);
    const float lx = x - static_cast<float>(result.xLow);
    const float hy = 1.0F - ly;
    const float hx = 1.0F - lx;
    result.w1 = hy * hx;
    result.w2 = hy * lx;
    result.w3 = ly * hx;
    result.w4 = ly * lx;
    result.valid = true;
    return result;
}

} // namespace OgIndex

#endif // OG_INDEX_KERNEL_H
