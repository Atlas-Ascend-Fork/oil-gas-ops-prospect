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
 * ComplexMul ophost UT：切分算法核心（complex_mul_tiling_core.h）。
 *
 * 对齐 ops-nn `<op>/tests/ut/op_host`。tiling 入口 ComplexMulTilingFunc 依赖
 * gert::TilingContext，ComputeComplexMulTiling 依赖 PlatformAscendC，CANN
 * 二进制包下无法构造，因此直接驱动 core 头。反向见
 * complex_mul_grad 见 tests/ut/op_host/complex_mul_grad/test_complex_mul_grad_tiling.cpp。
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "complex_mul_tiling_core.h"

namespace {

// 910B 的 AIV 核数（cube 20 核 × 2）
constexpr uint32_t kAivNum = 40;
constexpr uint64_t kLibApiWorkspace = 16 * 1024 * 1024;

/** 向上对齐到 kAlign 的倍数（除数是编译期正常量），与 kernel 侧 AlignUp 一致。 */
template <int64_t kAlign>
int64_t AlignUp(int64_t v)
{
    static_assert(kAlign > 0, "kAlign must be positive");
    return (v + kAlign - 1) / kAlign * kAlign;
}

/**
 * 复刻 kernel 侧 SplitByCore（og_elementwise_kernel.h）：
 * per = ceil(total/cores) 再上取整到 kAlign 的倍数；越界核拿到 len=0。
 * UT 用它验证"每个元素恰好被一个核处理一次"。
 *
 * 对齐量与 kernel 侧一样走模板常量；核数用三元表达式钳到 >= 1，两个除数在除法处
 * 都恒为正。
 */
template <int64_t kAlign>
void SplitByCore(int64_t total, int64_t coreNum, int64_t blockIdx, int64_t& offset, int64_t& len)
{
    offset = 0;
    len = 0;
    if (total <= 0) {
        return;
    }
    const int64_t cores = coreNum > 0 ? coreNum : 1;
    const int64_t per = AlignUp<kAlign>((total + cores - 1) / cores);
    const int64_t start = blockIdx * per;
    if (start >= total) {
        return;
    }
    offset = start;
    len = total - start < per ? total - start : per;
}

/** 所有 AIV 核的处理区间必须无重叠地恰好铺满 [0, total)。 */
void ExpectExactCoverage(int64_t total, int64_t coreNum)
{
    std::vector<int> hits(total > 0 ? static_cast<size_t>(total) : 0, 0);
    int64_t covered = 0;
    for (int64_t core = 0; core < coreNum; ++core) {
        int64_t offset = 0;
        int64_t len = 0;
        SplitByCore<ComplexMulHost::ALIGN_FLOATS>(total, coreNum, core, offset, len);
        ASSERT_GE(len, 0) << "total=" << total << " core=" << core;
        ASSERT_LE(offset + len, total) << "越界: total=" << total << " core=" << core;
        for (int64_t i = 0; i < len; ++i) {
            hits[static_cast<size_t>(offset + i)] += 1;
        }
        covered += len;
    }
    EXPECT_EQ(covered, total) << "total=" << total << " coreNum=" << coreNum;
    for (int64_t i = 0; i < total; ++i) {
        EXPECT_EQ(hits[static_cast<size_t>(i)], 1)
            << "元素 " << i << " 被处理 " << hits[static_cast<size_t>(i)] << " 次 (total=" << total << ")";
    }
}

const std::vector<int64_t>& TotalElemsCases()
{
    // 含：0、非 32B 对齐、刚好一拍、跨多拍、真实 shape(1*1*192*320*320)
    static const std::vector<int64_t> kCases = {
        0, 1, 7, 8, 9, 1000, 2048, 3072, 4096, 65537, 19660800,
    };
    return kCases;
}

} // namespace

// ---------------------------------------------------------------- ComplexMul

TEST(ComplexMulTiling, TileFloatsFitsUbBudget)
{
    // 方案 C：inQue(depth2 × 4 路) + outQue(depth2 × 2 路) + tmp = 13 份 tile
    EXPECT_EQ(ComplexMulHost::UB_SLOT_FACTOR, 13);
    EXPECT_LE(ComplexMulHost::TILE_FLOATS_CAP * static_cast<int64_t>(sizeof(float)) * ComplexMulHost::UB_SLOT_FACTOR,
              ComplexMulHost::UB_BUDGET_BYTES);
    // 且必须 32B 对齐、非零
    EXPECT_GT(ComplexMulHost::TILE_FLOATS_CAP, 0);
    EXPECT_EQ(ComplexMulHost::TILE_FLOATS_CAP % ComplexMulHost::ALIGN_FLOATS, 0);
}

TEST(ComplexMulTiling, PlanIsFilledForRepresentativeShapes)
{
    for (int64_t total : TotalElemsCases()) {
        ComplexMulTilingData tiling{};
        ComplexMulHost::ComplexMulPlan plan;
        ASSERT_TRUE(ComplexMulHost::ComputeComplexMulTilingCore(total, kAivNum, kLibApiWorkspace, tiling, plan))
            << "total=" << total;

        EXPECT_EQ(tiling.totalElems, total) << "total=" << total;
        EXPECT_EQ(tiling.coreNum, static_cast<int64_t>(kAivNum)) << "total=" << total;
        EXPECT_EQ(plan.blockDim, kAivNum) << "total=" << total;
        EXPECT_EQ(plan.tilingKey, COMPLEX_MUL_TILING_KEY) << "total=" << total;
        EXPECT_EQ(plan.totalWorkspace, kLibApiWorkspace) << "total=" << total;

        // tileFloats 必须为正、32B 对齐、不超过 UB 上限
        EXPECT_GT(tiling.tileFloats, 0) << "total=" << total;
        EXPECT_EQ(tiling.tileFloats % ComplexMulHost::ALIGN_FLOATS, 0) << "total=" << total;
        EXPECT_LE(tiling.tileFloats, ComplexMulHost::TILE_FLOATS_CAP) << "total=" << total;
    }
}

TEST(ComplexMulTiling, SmallTensorShrinksTileButKeepsAlignment)
{
    // 小于一拍：收缩 tileFloats，但不得小于 32B 对齐粒度
    ComplexMulTilingData tiling{};
    ComplexMulHost::ComplexMulPlan plan;

    ASSERT_TRUE(ComplexMulHost::ComputeComplexMulTilingCore(1000, kAivNum, 0, tiling, plan));
    EXPECT_LT(tiling.tileFloats, ComplexMulHost::TILE_FLOATS_CAP);
    EXPECT_EQ(tiling.tileFloats, 1000 / 8 * 8); // 向下对齐到 8

    ASSERT_TRUE(ComplexMulHost::ComputeComplexMulTilingCore(1, kAivNum, 0, tiling, plan));
    EXPECT_EQ(tiling.tileFloats, ComplexMulHost::ALIGN_FLOATS);

    // 大于一拍：吃满上限
    ASSERT_TRUE(ComplexMulHost::ComputeComplexMulTilingCore(1 << 20, kAivNum, 0, tiling, plan));
    EXPECT_EQ(tiling.tileFloats, ComplexMulHost::TILE_FLOATS_CAP);
}

TEST(ComplexMulTiling, DegenerateInputs)
{
    ComplexMulTilingData tiling{};
    ComplexMulHost::ComplexMulPlan plan;

    // 负元素数：拒绝
    EXPECT_FALSE(ComplexMulHost::ComputeComplexMulTilingCore(-1, kAivNum, 0, tiling, plan));

    // aivNum=0：兜底为 1，不得除零或产生 blockDim=0
    ASSERT_TRUE(ComplexMulHost::ComputeComplexMulTilingCore(1024, 0, 0, tiling, plan));
    EXPECT_EQ(plan.blockDim, 1u);
    EXPECT_EQ(tiling.coreNum, 1);

    // 空张量：合法，tileFloats 仍为正（kernel 侧靠 len<=0 直接返回）
    ASSERT_TRUE(ComplexMulHost::ComputeComplexMulTilingCore(0, kAivNum, 0, tiling, plan));
    EXPECT_EQ(tiling.totalElems, 0);
    EXPECT_GT(tiling.tileFloats, 0);
}

// --------------------------------------------- kernel 侧核间切分的覆盖性不变量

TEST(ComplexMulSplit, CoresCoverEveryElementExactlyOnce)
{
    // 小规模穷举（大 total 逐元素校验太慢，另用汇总校验）
    for (int64_t total : {0, 1, 7, 8, 9, 64, 100, 320, 1000}) {
        for (int64_t coreNum : {1, 2, 8, 40}) {
            ExpectExactCoverage(total, coreNum);
        }
    }
}

TEST(ComplexMulSplit, LargeShapesAreFullyCovered)
{
    for (int64_t total : {65537, 1048576, 19660800}) {
        for (int64_t coreNum : {1, 20, 40, 48}) {
            int64_t covered = 0;
            int64_t prevEnd = 0;
            for (int64_t core = 0; core < coreNum; ++core) {
                int64_t offset = 0;
                int64_t len = 0;
                SplitByCore<ComplexMulHost::ALIGN_FLOATS>(total, coreNum, core, offset, len);
                if (len > 0) {
                    // 区间必须紧邻且递增，保证无空洞、无重叠
                    EXPECT_EQ(offset, prevEnd) << "total=" << total << " core=" << core;
                    prevEnd = offset + len;
                }
                covered += len;
            }
            EXPECT_EQ(covered, total) << "total=" << total << " coreNum=" << coreNum;
            EXPECT_EQ(prevEnd, total) << "total=" << total << " coreNum=" << coreNum;
        }
    }
}

TEST(ComplexMulSplit, PerCoreChunkIsAlignedExceptTail)
{
    // 除最后一个非空核外，每核长度都应是 32B 对齐的，保证 DataCopyPad 走整块搬运
    const int64_t total = 19660800;
    const int64_t coreNum = 40;
    int64_t lastNonEmpty = -1;
    for (int64_t core = 0; core < coreNum; ++core) {
        int64_t offset = 0;
        int64_t len = 0;
        SplitByCore<ComplexMulHost::ALIGN_FLOATS>(total, coreNum, core, offset, len);
        if (len > 0) {
            lastNonEmpty = core;
            EXPECT_EQ(offset % ComplexMulHost::ALIGN_FLOATS, 0) << "core=" << core;
        }
    }
    ASSERT_GE(lastNonEmpty, 0);
}
