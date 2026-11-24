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
 * CustomGRUV2 host UT for the platform-independent batch tiling core.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <utility>

#include "custom_gruv2_tiling_core.h"

namespace {

using CustomGRUV2Host::ComputeCustomGRUV2TilingCore;
using CustomGRUV2Host::CustomGRUV2TilingPlan;

constexpr uint64_t kUbBudget = CustomGRUV2Host::kUbByteBudget;

TEST(CustomGRUV2TilingCore, UsesAvailableCoresForTypicalBatch)
{
    // batch=32 across 48 AIVs: one sample per block and hidden remains 8-wide aligned.
    CustomGRUV2TilingPlan plan;
    ASSERT_TRUE(ComputeCustomGRUV2TilingCore(32, 24, 48, kUbBudget, plan));

    EXPECT_EQ(plan.blockDim, 32U);
    EXPECT_EQ(plan.batchPerBlock, 1U);
    EXPECT_EQ(plan.hiddenAligned, 24U);
    EXPECT_EQ(plan.ubBytesPerBlock, (1U + 8U) * 24U * sizeof(float));
    EXPECT_LE(plan.ubBytesPerBlock, kUbBudget);
    EXPECT_FALSE(plan.usedDefaultAivCoreNum);
}

TEST(CustomGRUV2TilingCore, RoundsHiddenAndCapsBlockCountByBatch)
{
    // A non-aligned hidden size is padded to 8; a small batch never launches empty blocks.
    CustomGRUV2TilingPlan plan;
    ASSERT_TRUE(ComputeCustomGRUV2TilingCore(3, 7, 48, kUbBudget, plan));

    EXPECT_EQ(plan.blockDim, 3U);
    EXPECT_EQ(plan.batchPerBlock, 1U);
    EXPECT_EQ(plan.hiddenAligned, 8U);
    EXPECT_LE(plan.ubBytesPerBlock, kUbBudget);
}

TEST(CustomGRUV2TilingCore, FallsBackToDocumentedDefaultWhenCoreCountIsZero)
{
    CustomGRUV2TilingPlan plan;
    ASSERT_TRUE(ComputeCustomGRUV2TilingCore(32, 24, 0, kUbBudget, plan));

    EXPECT_TRUE(plan.usedDefaultAivCoreNum);
    EXPECT_EQ(plan.blockDim, CustomGRUV2Host::kDefaultAivCoreNum);
    EXPECT_EQ(plan.batchPerBlock, 2U);
}

TEST(CustomGRUV2TilingCore, ReducesLocalBatchToFitUbBudget)
{
    // The initial 24-core split assigns 2 samples per block. At H=4992 that
    // footprint exceeds the budget, so the plan must split into 32 blocks.
    CustomGRUV2TilingPlan plan;
    ASSERT_TRUE(ComputeCustomGRUV2TilingCore(32, 4992, 24, kUbBudget, plan));

    EXPECT_EQ(plan.hiddenAligned, 4992U);
    EXPECT_EQ(plan.blockDim, 32U);
    EXPECT_EQ(plan.batchPerBlock, 1U);
    EXPECT_LE(plan.ubBytesPerBlock, kUbBudget);
}

TEST(CustomGRUV2TilingCore, BlockRangesCoverEachBatchExactlyOnce)
{
    // Match the kernel's `start = blockIdx * ceil(B / blockDim)` partitioning.
    // This partition may schedule empty trailing blocks; non-empty ranges must
    // still be contiguous, non-overlapping, and cover the full batch.
    for (const auto& shape : {std::pair<uint32_t, uint32_t>{32, 24}, {32, 4992}, {32, 4993}, {3, 7}}) {
        CustomGRUV2TilingPlan plan;
        ASSERT_TRUE(ComputeCustomGRUV2TilingCore(shape.first, shape.second, 24, kUbBudget, plan));

        uint32_t nextBatch = 0;
        for (uint32_t block = 0; block < plan.blockDim; ++block) {
            const uint32_t rawBegin = block * plan.batchPerBlock;
            if (rawBegin >= shape.first) {
                continue;
            }
            const uint32_t begin = rawBegin;
            const uint32_t end = (begin + plan.batchPerBlock < shape.first) ? begin + plan.batchPerBlock : shape.first;
            EXPECT_EQ(begin, nextBatch) << "block=" << block;
            nextBatch = end;
        }
        EXPECT_EQ(nextBatch, shape.first);
    }
}

TEST(CustomGRUV2TilingCore, RejectsInputsThatCannotFitTheFixedUbFootprint)
{
    CustomGRUV2TilingPlan plan;
    // H=7017 aligns to 7024; eight fixed vectors alone exceed the UB budget.
    EXPECT_FALSE(ComputeCustomGRUV2TilingCore(1, 7017, 24, kUbBudget, plan));
}

TEST(CustomGRUV2TilingCore, PreservesEightVectorBudgetBoundary)
{
    CustomGRUV2TilingPlan plan;
    ASSERT_TRUE(ComputeCustomGRUV2TilingCore(1, 5000, 24, kUbBudget, plan));
    EXPECT_EQ(plan.ubBytesPerBlock, 180000U);
    EXPECT_EQ(plan.ubBytesPerBlock, CustomGRUV2Host::UbytesPerBlock(1, 5000));
    EXPECT_FALSE(ComputeCustomGRUV2TilingCore(1, 5008, 24, kUbBudget, plan));
    // H=5100 aligns to 5104: even one sample per block exceeds the eight-vector budget.
    EXPECT_FALSE(ComputeCustomGRUV2TilingCore(1, 5100, 24, kUbBudget, plan));
    EXPECT_FALSE(ComputeCustomGRUV2TilingCore(32, 5100, 24, kUbBudget, plan));
    // This shape fit the old seven-vector estimate but exceeds the current UB budget.
    EXPECT_FALSE(ComputeCustomGRUV2TilingCore(1, 5632, 24, kUbBudget, plan));
}

TEST(CustomGRUV2TilingCore, PreservesEmptyBatchAndRejectsInvalidBudget)
{
    CustomGRUV2TilingPlan plan;
    // Keep the previous behavior: an empty batch gets one no-op block.
    ASSERT_TRUE(ComputeCustomGRUV2TilingCore(0, 8, 24, kUbBudget, plan));
    EXPECT_EQ(plan.blockDim, 1U);
    EXPECT_EQ(plan.batchPerBlock, 0U);
    EXPECT_FALSE(ComputeCustomGRUV2TilingCore(1, 0, 24, kUbBudget, plan));
    EXPECT_FALSE(ComputeCustomGRUV2TilingCore(1, 8, 24, 0, plan));
}

} // namespace
