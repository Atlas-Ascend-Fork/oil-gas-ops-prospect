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
 * CustomGRUV2 host tiling 的 UB 占用公式（独立纯函数，便于 gtest 边界测试）。
 *
 * kernel 侧每 block 的常驻 vector 缓冲（custom_gruv2.cpp TPipe InitBuffer）：
 *   - 每样本 hidden 状态 hBuf：batchPerBlock * Ha
 *   - 固定 8 个 Ha 宽度矢量化 buffer：
 *       gBuf（gates r/z/n，3 段）= 3 * Ha
 *       rBuf / zBuf / t1Buf / t2Buf = 4 * Ha
 *       oneBuf（Div 的 1.0 常量，D2 数值修复新增）= 1 * Ha
 *   合计 ubBytes = (batchPerBlock + kFixedUbVecCount) * Ha * sizeof(float)。
 *
 * 历史注记：Host 曾按 7 个固定矢量估算（gates + 4 scratch），oneBuf 引入后
 * 未同步造成大 hidden（如 B=1,H=5632）被错误放行（评审 7680085 阻塞项）。
 */
#ifndef CUSTOM_GRUV2_UB_FOOTPRINT_H
#define CUSTOM_GRUV2_UB_FOOTPRINT_H

#include <cstdint>

namespace CustomGRUV2Host {

/// gates(r/z/n 3 段) + r/z/t1/t2(4) + oneBuf(1, D2 数值修复新增)。
constexpr uint64_t kFixedUbVecCount = 8U;

/// 每 block 的 UB 占用（字节），batchPerBlock 与 hiddenAligned 均需 > 0。
inline uint64_t UbytesPerBlock(uint32_t batchPerBlock, uint32_t hiddenAligned)
{
    return static_cast<uint64_t>(batchPerBlock + kFixedUbVecCount) * static_cast<uint64_t>(hiddenAligned) *
           sizeof(float);
}

/// 给定 hiddenAligned 与 UB 预算，本核可容纳的最大 batchPerBlock（0 表示放不下）。
inline uint32_t MaxBatchPerBlock(uint32_t hiddenAligned, uint64_t ubByteBudget)
{
    if (hiddenAligned == 0U) {
        return 0U;
    }
    const uint64_t ubSlots = ubByteBudget / sizeof(float);
    const uint64_t fixedSlots = kFixedUbVecCount * static_cast<uint64_t>(hiddenAligned);
    if (ubSlots <= fixedSlots) {
        return 0U;
    }
    return static_cast<uint32_t>((ubSlots - fixedSlots) / hiddenAligned);
}

} // namespace CustomGRUV2Host

#endif // CUSTOM_GRUV2_UB_FOOTPRINT_H
