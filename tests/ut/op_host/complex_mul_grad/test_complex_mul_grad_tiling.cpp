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
 * ComplexMulGrad ophost UT：切分算法核心（complex_mul_grad_tiling_core.h）。
 *
 * 对齐 ops-nn `<op>/tests/ut/op_host`。与前向共用 og_elementwise_tiling.h，
 * 反向 UB 份数更多、单拍更短。核间覆盖性不变量见前向 UT。
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "complex_mul_grad_tiling_core.h"
#include "complex_mul_tiling_core.h"

namespace {

constexpr uint32_t kAivNum = 40;
constexpr uint64_t kLibApiWorkspace = 16 * 1024 * 1024;

const std::vector<int64_t>& TotalElemsCases()
{
    static const std::vector<int64_t> kCases = {
        0, 1, 7, 8, 9, 1000, 2048, 3072, 4096, 65537, 19660800,
    };
    return kCases;
}

} // namespace

TEST(ComplexMulGradTiling, TileFloatsFitsUbBudget)
{
    // 反向多 2 路输入、2 路输出：inQue(2×6) + outQue(2×4) + tmp = 21 份 tile
    EXPECT_EQ(ComplexMulGradHost::UB_SLOT_FACTOR, 21);
    EXPECT_LE(
        ComplexMulGradHost::TILE_FLOATS_CAP * static_cast<int64_t>(sizeof(float)) * ComplexMulGradHost::UB_SLOT_FACTOR,
        ComplexMulGradHost::UB_BUDGET_BYTES);
    EXPECT_GT(ComplexMulGradHost::TILE_FLOATS_CAP, 0);
    EXPECT_EQ(ComplexMulGradHost::TILE_FLOATS_CAP % ComplexMulGradHost::ALIGN_FLOATS, 0);
}

TEST(ComplexMulGradTiling, TileIsSmallerThanForward)
{
    // 反向占用的 UB 份数更多，单拍元素数必须更少，否则会超 UB
    EXPECT_GT(ComplexMulGradHost::UB_SLOT_FACTOR, ComplexMulHost::UB_SLOT_FACTOR);
    EXPECT_LT(ComplexMulGradHost::TILE_FLOATS_CAP, ComplexMulHost::TILE_FLOATS_CAP);
}

TEST(ComplexMulGradTiling, PlanIsFilledForRepresentativeShapes)
{
    for (int64_t total : TotalElemsCases()) {
        ComplexMulGradTilingData tiling{};
        ComplexMulGradHost::ComplexMulGradPlan plan;
        ASSERT_TRUE(ComplexMulGradHost::ComputeComplexMulGradTilingCore(total, kAivNum, kLibApiWorkspace, tiling, plan))
            << "total=" << total;

        EXPECT_EQ(tiling.totalElems, total) << "total=" << total;
        EXPECT_EQ(tiling.coreNum, static_cast<int64_t>(kAivNum)) << "total=" << total;
        EXPECT_EQ(plan.blockDim, kAivNum) << "total=" << total;
        EXPECT_EQ(plan.tilingKey, COMPLEX_MUL_GRAD_TILING_KEY) << "total=" << total;
        EXPECT_EQ(plan.totalWorkspace, kLibApiWorkspace) << "total=" << total;

        EXPECT_GT(tiling.tileFloats, 0) << "total=" << total;
        EXPECT_EQ(tiling.tileFloats % ComplexMulGradHost::ALIGN_FLOATS, 0) << "total=" << total;
        EXPECT_LE(tiling.tileFloats, ComplexMulGradHost::TILE_FLOATS_CAP) << "total=" << total;
    }
}

TEST(ComplexMulGradTiling, DegenerateInputs)
{
    ComplexMulGradTilingData tiling{};
    ComplexMulGradHost::ComplexMulGradPlan plan;

    EXPECT_FALSE(ComplexMulGradHost::ComputeComplexMulGradTilingCore(-1, kAivNum, 0, tiling, plan));

    // 空张量：合法，tileFloats 仍为正（kernel 侧靠 len<=0 直接返回）
    ASSERT_TRUE(ComplexMulGradHost::ComputeComplexMulGradTilingCore(0, kAivNum, 0, tiling, plan));
    EXPECT_EQ(tiling.totalElems, 0);
    EXPECT_GT(tiling.tileFloats, 0);
}
