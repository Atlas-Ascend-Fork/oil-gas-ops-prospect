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
 * CustomGRUV2 host UB footprint 边界测试。
 *
 * 背景（评审 7680085 阻塞项）：kernel 为 D2 数值修复新增常量矢量 oneBuf
 * （1 * Ha），固定矢量由 7 升至 8；Host 预算未同步曾导致
 * B=1,H=5632（对齐后 Ha=5632）在旧 (1+7)*Ha 预算下错误放行（恰好
 * 176KiB），kernel 实际需 (1+8)*Ha -> 超出 192KiB UB 总量。
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "custom_gruv2_ub_footprint.h"

namespace {

using CustomGRUV2Host::kFixedUbVecCount;
using CustomGRUV2Host::MaxBatchPerBlock;
using CustomGRUV2Host::UbytesPerBlock;

constexpr uint64_t kUbBudget = 176U * 1024U; // 与 host kUbByteBudget 一致

TEST(CustomGRUV2UbFootprint, CountsEightFixedVectors)
{
    // 典型形状：batchPerBlock=1, Ha=24 → (1+8)*24*4 = 864 bytes。
    EXPECT_EQ(kFixedUbVecCount, 8U);
    EXPECT_EQ(UbytesPerBlock(1, 24), (1U + 8U) * 24U * sizeof(float));
    EXPECT_EQ(UbytesPerBlock(2, 24), (2U + 8U) * 24U * sizeof(float));
}

TEST(CustomGRUV2UbFootprint, RejectsBigHiddenReviewCase)
{
    // 评审给出的交叉点：B=1,H=5632（Ha=5632）。
    // 旧 7 矢量预算 (1+7)*5632*4 = 180224 == 176KiB 恰好放行；
    // kernel 实际需要 (1+8)*5632*4 = 202752 > 192KiB UB 总量 → 必须拒绝。
    EXPECT_EQ((1U + 7U) * 5632U * sizeof(float), kUbBudget);
    EXPECT_GT(UbytesPerBlock(1, 5632), kUbBudget);
    EXPECT_EQ(MaxBatchPerBlock(5632, kUbBudget), 0U);
}

TEST(CustomGRUV2UbFootprint, SingleBatchAcceptRejectBoundary)
{
    // (1+8)*Ha <= 176KiB(=45056 float 槽) → Ha <= 5000：
    //   H=5000（Ha=5000）：45000 bytes ≤ 45056 → 通过；
    //   H=5008（Ha=5008）：45072 bytes > 45056 → 拒绝。
    EXPECT_LE(UbytesPerBlock(1, 5000), kUbBudget);
    EXPECT_GT(UbytesPerBlock(1, 5008), kUbBudget);
    EXPECT_NE(MaxBatchPerBlock(5000, kUbBudget), 0U);
    EXPECT_EQ(MaxBatchPerBlock(5008, kUbBudget), 0U);
}

TEST(CustomGRUV2UbFootprint, LargeHiddenEnvelopeShrinkIsDocumented)
{
    // 收缩记录：D2 修复前 H=5100(B=32,aiv=24,bpb=1) 可执行；
    // 8 固定矢量后：(1+8)*5104*4 超预算 → 内核无法安全运行，拒绝。
    EXPECT_FALSE(UbytesPerBlock(1, 5104) <= kUbBudget);
    EXPECT_EQ(MaxBatchPerBlock(5104, kUbBudget), 0U);
}

TEST(CustomGRUV2UbFootprint, MaxBatchPerBlockScalesWithHidden)
{
    // Ha=1000: (45056-8000)/1000 = 37（expected derive check）。
    EXPECT_EQ(MaxBatchPerBlock(1000, kUbBudget), 37U);
    // 同 budget 小 Ha 时比例上限放大。
    EXPECT_EQ(MaxBatchPerBlock(24, kUbBudget), (45056U - 8U * 24U) / 24U);
}

TEST(CustomGRUV2UbFootprint, ZeroHiddenIsInvalid) { EXPECT_EQ(MaxBatchPerBlock(0, kUbBudget), 0U); }

} // namespace
