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
 * og_index_tiling.h 的 ophost UT：64B 对齐 slab 切分与覆盖性不变量。
 *
 * 这是 ROIAlignRotated(Grad) / SwapAlign2Nat(Grad) 反向共用的切分基础：
 *   - 每个元素恰好属于一个 slab；
 *   - 每个 slab 恰好属于一个核；
 *   - slab 起点 64B 对齐（首 slab 从 0 开始）；
 *   - 最后一个 slab 可短于 slabFloats，且恰好结束于 total。
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "og_index_tiling.h"

namespace {

using OgOpsCommon::ComputeFlatSlabPlan;
using OgOpsCommon::FlatSlabElemRange;
using OgOpsCommon::FlatSlabPlan;
using OgOpsCommon::FlatSlabRangeForCore;
using OgOpsCommon::INDEX_ALIGN_FLOATS;
using OgOpsCommon::IndexMulChecked;

void ExpectPlanCoversEverything(int64_t total, uint32_t aivNum, int64_t cap)
{
    FlatSlabPlan plan;
    ASSERT_TRUE(ComputeFlatSlabPlan(total, aivNum, cap, plan)) << "total=" << total << " aiv=" << aivNum;

    EXPECT_EQ(plan.total, total);
    if (total == 0) {
        EXPECT_EQ(plan.numSlabs, 0);
        EXPECT_EQ(plan.slabFloats, 0);
        EXPECT_EQ(plan.blockDim, 1U);
        EXPECT_EQ(plan.baseSlabs, 0);
        EXPECT_EQ(plan.remSlabs, 0);
        return;
    }

    ASSERT_GT(plan.slabFloats, 0);
    ASSERT_LE(plan.slabFloats, cap);
    EXPECT_EQ(plan.numSlabs, (total + plan.slabFloats - 1) / plan.slabFloats);
    ASSERT_GE(plan.blockDim, 1U);
    EXPECT_LE(plan.blockDim, (aivNum == 0U ? 1U : aivNum));
    EXPECT_EQ(plan.baseSlabs, plan.numSlabs / static_cast<int64_t>(plan.blockDim));
    EXPECT_EQ(plan.remSlabs, plan.numSlabs % static_cast<int64_t>(plan.blockDim));

    // slab 连续铺满 [0,total)，起点 64B 对齐。
    int64_t cursor = 0;
    for (int64_t slab = 0; slab < plan.numSlabs; ++slab) {
        int64_t start = 0;
        int64_t end = 0;
        FlatSlabElemRange(plan, slab, start, end);
        EXPECT_EQ(start, cursor) << "slab=" << slab << " total=" << total;
        EXPECT_GT(end, start) << "slab=" << slab << " total=" << total;
        EXPECT_EQ(start % INDEX_ALIGN_FLOATS, 0) << "slab=" << slab << " total=" << total;
        cursor = end;
    }
    EXPECT_EQ(cursor, total) << "total=" << total;

    // 核区间连续铺满所有 slab。
    int64_t coreCursor = 0;
    for (uint32_t core = 0; core < plan.blockDim; ++core) {
        int64_t start = -1;
        int64_t end = -1;
        FlatSlabRangeForCore(plan, core, start, end);
        EXPECT_EQ(start, coreCursor) << "core=" << core << " total=" << total;
        EXPECT_GT(end, start) << "core=" << core << " total=" << total;
        coreCursor = end;
    }
    EXPECT_EQ(coreCursor, plan.numSlabs) << "total=" << total;

    // 越界核必须拿到空区间。
    int64_t outStart = -1;
    int64_t outEnd = -1;
    FlatSlabRangeForCore(plan, plan.blockDim, outStart, outEnd);
    EXPECT_EQ(outStart, 0);
    EXPECT_EQ(outEnd, 0);
}

} // namespace

TEST(OgIndexTiling, SlabPlanCoversEveryElementExactlyOnce)
{
    for (int64_t total : {0, 1, 7, 15, 16, 17, 64, 1000, 32768, 32769, 1000000}) {
        for (uint32_t aiv : {0U, 1U, 2U, 20U, 40U, 48U}) {
            for (int64_t cap : {16, 64, 1024, 32768}) {
                ExpectPlanCoversEverything(total, aiv, cap);
            }
        }
    }
}

TEST(OgIndexTiling, SmallTotalSingleSlabStartsAtZero)
{
    for (int64_t total : {1, 7, 15}) {
        FlatSlabPlan plan;
        ASSERT_TRUE(ComputeFlatSlabPlan(total, 40U, 32768, plan));
        EXPECT_EQ(plan.numSlabs, 1);
        EXPECT_EQ(plan.slabFloats, total);
        EXPECT_EQ(plan.blockDim, 1U);
        int64_t start = -1;
        int64_t end = -1;
        FlatSlabElemRange(plan, 0, start, end);
        EXPECT_EQ(start, 0);
        EXPECT_EQ(end, total);
    }
}

TEST(OgIndexTiling, LargeTotalUsesAlignedCapAndShortTail)
{
    const int64_t total = 1000000;
    const int64_t cap = 32768;
    FlatSlabPlan plan;
    ASSERT_TRUE(ComputeFlatSlabPlan(total, 40U, cap, plan));
    EXPECT_EQ(plan.slabFloats, cap);
    EXPECT_EQ(plan.slabFloats % INDEX_ALIGN_FLOATS, 0);
    int64_t start = -1;
    int64_t end = -1;
    FlatSlabElemRange(plan, plan.numSlabs - 1, start, end);
    EXPECT_LT(end - start, plan.slabFloats);
    EXPECT_EQ(end, total);
}

TEST(OgIndexTiling, RejectsInvalidArguments)
{
    FlatSlabPlan plan;
    EXPECT_FALSE(ComputeFlatSlabPlan(-1, 40U, 1024, plan));
    EXPECT_FALSE(ComputeFlatSlabPlan(100, 40U, 0, plan));
    EXPECT_FALSE(ComputeFlatSlabPlan(100, 40U, -16, plan));
}

TEST(OgIndexTiling, IndexMulCheckedRejectsNegativeAndOverflow)
{
    int64_t out = 0;
    EXPECT_TRUE(IndexMulChecked(6, 7, out));
    EXPECT_EQ(out, 42);
    EXPECT_FALSE(IndexMulChecked(-1, 7, out));
    EXPECT_FALSE(IndexMulChecked(7, -1, out));

    const int64_t kMax = std::numeric_limits<int64_t>::max();
    EXPECT_FALSE(IndexMulChecked(kMax, 2, out));
    EXPECT_TRUE(IndexMulChecked(kMax, 1, out));
    EXPECT_EQ(out, kMax);
}
