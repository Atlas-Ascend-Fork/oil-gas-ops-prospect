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
 * Platform-independent CustomGRUV2 batch tiling calculation.
 *
 * The CANN TilingContext adapter calls this function, and host unit tests call
 * the same function directly. Keep this header independent of CANN headers.
 */
#ifndef CUSTOM_GRUV2_TILING_CORE_H
#define CUSTOM_GRUV2_TILING_CORE_H

#include <cstdint>

#include "custom_gruv2_ub_footprint.h"

namespace CustomGRUV2Host {

constexpr uint32_t kDefaultAivCoreNum = 24;
constexpr uint64_t kUbByteBudget = 176 * 1024;
constexpr uint32_t kHiddenAlignment = 8;

struct CustomGRUV2TilingPlan {
    uint32_t blockDim = 0;
    uint32_t batchPerBlock = 0;
    uint32_t hiddenAligned = 0;
    uint64_t ubBytesPerBlock = 0;
    bool usedDefaultAivCoreNum = false;
};

inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return value / divisor + (value % divisor != 0U ? 1U : 0U);
}

/**
 * Compute blockDim and the maximum local batch footprint.
 *
 * A zero AIV core count uses the same explicit 24-core fallback as the CANN
 * adapter. The kernel allocates `(localBatch + 8) * hiddenAligned`
 * fp32 values per block, so the returned UB estimate must not exceed the
 * supplied budget.
 */
inline bool ComputeCustomGRUV2TilingCore(uint32_t batchSize, uint32_t hiddenSize, uint32_t aivCoreNum,
                                         uint64_t ubByteBudget, CustomGRUV2TilingPlan& plan)
{
    plan = CustomGRUV2TilingPlan{};
    if (hiddenSize == 0U || ubByteBudget == 0U) {
        return false;
    }

    if (aivCoreNum == 0U) {
        aivCoreNum = kDefaultAivCoreNum;
        plan.usedDefaultAivCoreNum = true;
    }

    // Reject before alignment if even the eight fixed vectors cannot
    // fit. This also keeps the alignment and following products in range.
    const uint64_t ubSlots = ubByteBudget / sizeof(float);
    const uint64_t maxHiddenAligned = ubSlots / kFixedUbVecCount;
    if (hiddenSize > maxHiddenAligned) {
        return false;
    }

    const uint32_t hiddenAligned = ((hiddenSize + kHiddenAlignment - 1U) / kHiddenAlignment) * kHiddenAlignment;
    const uint64_t fixedSlots = kFixedUbVecCount * hiddenAligned;
    if (fixedSlots >= ubSlots) {
        return false;
    }

    uint32_t blockDim = (batchSize < aivCoreNum) ? batchSize : aivCoreNum;
    if (blockDim == 0U) {
        // Preserve the existing empty-batch behavior: launch one no-op block.
        blockDim = 1U;
    }
    uint32_t batchPerBlock = CeilDiv(batchSize, blockDim);
    uint64_t ubBytesPerBlock = UbytesPerBlock(batchPerBlock, hiddenAligned);

    if (ubBytesPerBlock > ubByteBudget) {
        const uint64_t maxBatchPerBlock = MaxBatchPerBlock(hiddenAligned, ubByteBudget);
        if (maxBatchPerBlock == 0U) {
            return false;
        }

        const uint64_t adjustedBlockDim = (static_cast<uint64_t>(batchSize) + maxBatchPerBlock - 1U) / maxBatchPerBlock;
        if (adjustedBlockDim == 0U || adjustedBlockDim > batchSize) {
            return false;
        }
        blockDim = static_cast<uint32_t>(adjustedBlockDim);
        batchPerBlock = CeilDiv(batchSize, blockDim);
        ubBytesPerBlock = UbytesPerBlock(batchPerBlock, hiddenAligned);
    }

    if (ubBytesPerBlock > ubByteBudget) {
        return false;
    }

    plan.blockDim = blockDim;
    plan.batchPerBlock = batchPerBlock;
    plan.hiddenAligned = hiddenAligned;
    plan.ubBytesPerBlock = ubBytesPerBlock;
    return true;
}

} // namespace CustomGRUV2Host

#endif // CUSTOM_GRUV2_TILING_CORE_H
