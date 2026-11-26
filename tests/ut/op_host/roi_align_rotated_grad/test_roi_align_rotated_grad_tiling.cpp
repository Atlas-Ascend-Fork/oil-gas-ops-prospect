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
 * ROIAlignRotated 反向 ophost UT：grad_input 的 64B slab 独占切分。
 *
 * 覆盖：覆盖性、K=0 时仍覆盖全部 grad_input、aivNum=0、非法属性/溢出。
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "roi_align_rotated_grad_tiling_core.h"

namespace {

using RoiAlignRotatedGradHost::ComputeRoiAlignRotatedGradTilingCore;
using RoiAlignRotatedGradHost::ROI_ALIGN_ROTATED_GRAD_SLAB_FLOATS_CAP;

constexpr uint32_t kAivNum = 40; // 910B3

struct GradShape {
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    int64_t k;
    int64_t pooledH;
    int64_t pooledW;
    int64_t samplingRatio;
    float spatialScale;
};

const std::vector<GradShape>& RepresentativeShapes()
{
    static const std::vector<GradShape> kShapes = {
        {1, 256, 200, 200, 1000, 7, 7, 0, 1.0F}, {1, 256, 100, 100, 512, 14, 14, 2, 0.5F},
        {2, 128, 64, 80, 333, 7, 7, 1, 1.0F},    {1, 1, 5, 5, 1, 4, 4, 0, 1.0F},
        {1, 3, 1, 1, 2, 1, 1, 0, 1.0F},          {1, 8, 32, 32, 0, 7, 7, 0, 1.0F},
    };
    return kShapes;
}

void ExpectTilingCoversInput(const ROIAlignRotatedGradTilingData& tiling)
{
    if (tiling.totalIn == 0) {
        EXPECT_EQ(tiling.numSlabs, 0);
        EXPECT_EQ(tiling.blockDim, 1);
        return;
    }
    EXPECT_GT(tiling.slabFloats, 0);
    if (tiling.numSlabs > 1) {
        EXPECT_EQ(tiling.slabFloats % OgOpsCommon::INDEX_ALIGN_FLOATS, 0);
    }
    EXPECT_LE(tiling.slabFloats, ROI_ALIGN_ROTATED_GRAD_SLAB_FLOATS_CAP);
    EXPECT_EQ(tiling.numSlabs, (tiling.totalIn + tiling.slabFloats - 1) / tiling.slabFloats);
    EXPECT_GE(tiling.blockDim, 1);
    EXPECT_LE(tiling.blockDim, static_cast<int64_t>(kAivNum));
    EXPECT_EQ(tiling.baseSlabs, tiling.numSlabs / tiling.blockDim);
    EXPECT_EQ(tiling.remSlabs, tiling.numSlabs % tiling.blockDim);

    int64_t cursor = 0;
    for (int64_t slab = 0; slab < tiling.numSlabs; ++slab) {
        const int64_t start = slab * tiling.slabFloats;
        int64_t end = start + tiling.slabFloats;
        if (end > tiling.totalIn) {
            end = tiling.totalIn;
        }
        EXPECT_EQ(start, cursor);
        EXPECT_EQ(start % OgOpsCommon::INDEX_ALIGN_FLOATS, 0);
        EXPECT_GT(end, start);
        cursor = end;
    }
    EXPECT_EQ(cursor, tiling.totalIn);
}

} // namespace

TEST(RoiAlignRotatedGradTiling, SlabCapFitsUbBudget)
{
    EXPECT_GT(ROI_ALIGN_ROTATED_GRAD_SLAB_FLOATS_CAP, 0);
    EXPECT_EQ(ROI_ALIGN_ROTATED_GRAD_SLAB_FLOATS_CAP % OgOpsCommon::INDEX_ALIGN_FLOATS, 0);
    EXPECT_LE(ROI_ALIGN_ROTATED_GRAD_SLAB_FLOATS_CAP * static_cast<int64_t>(sizeof(float)),
              OgOpsCommon::INDEX_UB_BUDGET_BYTES);
}

TEST(RoiAlignRotatedGradTiling, PlanFilledForRepresentativeShapes)
{
    for (const GradShape& s : RepresentativeShapes()) {
        ROIAlignRotatedGradTilingData tiling{};
        RoiAlignRotatedGradHost::RoiAlignRotatedGradPlan plan;
        ASSERT_TRUE(ComputeRoiAlignRotatedGradTilingCore(s.n, s.c, s.h, s.w, s.k, s.pooledH, s.pooledW, s.samplingRatio,
                                                         s.spatialScale, kAivNum, tiling, plan))
            << "shape=" << s.n << "," << s.c << "," << s.h << "," << s.w << " k=" << s.k;

        EXPECT_EQ(tiling.n, s.n);
        EXPECT_EQ(tiling.c, s.c);
        EXPECT_EQ(tiling.h, s.h);
        EXPECT_EQ(tiling.w, s.w);
        EXPECT_EQ(tiling.k, s.k);
        EXPECT_EQ(tiling.pooledH, s.pooledH);
        EXPECT_EQ(tiling.pooledW, s.pooledW);
        EXPECT_EQ(tiling.samplingRatio, s.samplingRatio);
        EXPECT_FLOAT_EQ(tiling.spatialScale, s.spatialScale);
        EXPECT_EQ(tiling.totalIn, s.n * s.c * s.h * s.w);
        EXPECT_EQ(plan.tilingKey, ROI_ALIGN_ROTATED_GRAD_TILING_KEY);
        EXPECT_EQ(plan.blockDim, static_cast<uint32_t>(tiling.blockDim));
        EXPECT_EQ(plan.totalWorkspace, 0U);
        ExpectTilingCoversInput(tiling);
    }
}

TEST(RoiAlignRotatedGradTiling, ZeroRoisStillCoversWholeInput)
{
    // K=0 时 kernel 仍需把 grad_input 全零写出；切分必须覆盖全部输入元素。
    ROIAlignRotatedGradTilingData tiling{};
    RoiAlignRotatedGradHost::RoiAlignRotatedGradPlan plan;
    ASSERT_TRUE(ComputeRoiAlignRotatedGradTilingCore(1, 8, 32, 32, 0, 7, 7, 0, 1.0F, kAivNum, tiling, plan));
    EXPECT_EQ(tiling.totalIn, 8 * 32 * 32);
    EXPECT_GT(tiling.numSlabs, 0);
    ExpectTilingCoversInput(tiling);
}

TEST(RoiAlignRotatedGradTiling, EmptyInputYieldsEmptyPlan)
{
    ROIAlignRotatedGradTilingData tiling{};
    RoiAlignRotatedGradHost::RoiAlignRotatedGradPlan plan;
    ASSERT_TRUE(ComputeRoiAlignRotatedGradTilingCore(1, 0, 32, 32, 10, 7, 7, 0, 1.0F, kAivNum, tiling, plan));
    EXPECT_EQ(tiling.totalIn, 0);
    EXPECT_EQ(tiling.numSlabs, 0);
    EXPECT_EQ(tiling.blockDim, 1);
}

TEST(RoiAlignRotatedGradTiling, ZeroAivFallsBackToOneCore)
{
    ROIAlignRotatedGradTilingData tiling{};
    RoiAlignRotatedGradHost::RoiAlignRotatedGradPlan plan;
    ASSERT_TRUE(ComputeRoiAlignRotatedGradTilingCore(1, 8, 32, 32, 10, 7, 7, 0, 1.0F, 0U, tiling, plan));
    EXPECT_EQ(tiling.blockDim, 1);
    EXPECT_EQ(plan.blockDim, 1U);
}

TEST(RoiAlignRotatedGradTiling, RejectsInvalidShapesAndAttrs)
{
    ROIAlignRotatedGradTilingData tiling{};
    RoiAlignRotatedGradHost::RoiAlignRotatedGradPlan plan;

    EXPECT_FALSE(ComputeRoiAlignRotatedGradTilingCore(-1, 8, 32, 32, 10, 7, 7, 0, 1.0F, kAivNum, tiling, plan));
    EXPECT_FALSE(ComputeRoiAlignRotatedGradTilingCore(1, -1, 32, 32, 10, 7, 7, 0, 1.0F, kAivNum, tiling, plan));
    EXPECT_FALSE(ComputeRoiAlignRotatedGradTilingCore(1, 8, -1, 32, 10, 7, 7, 0, 1.0F, kAivNum, tiling, plan));
    EXPECT_FALSE(ComputeRoiAlignRotatedGradTilingCore(1, 8, 32, -1, 10, 7, 7, 0, 1.0F, kAivNum, tiling, plan));
    EXPECT_FALSE(ComputeRoiAlignRotatedGradTilingCore(1, 8, 32, 32, -1, 7, 7, 0, 1.0F, kAivNum, tiling, plan));
    EXPECT_FALSE(ComputeRoiAlignRotatedGradTilingCore(1, 8, 32, 32, 10, 0, 7, 0, 1.0F, kAivNum, tiling, plan));
    EXPECT_FALSE(ComputeRoiAlignRotatedGradTilingCore(1, 8, 32, 32, 10, 7, 0, 0, 1.0F, kAivNum, tiling, plan));
    EXPECT_FALSE(ComputeRoiAlignRotatedGradTilingCore(1, 8, 32, 32, 10, 7, 7, -1, 1.0F, kAivNum, tiling, plan));
    EXPECT_FALSE(ComputeRoiAlignRotatedGradTilingCore(1, 8, 32, 32, 10, 7, 7, 0, 0.0F, kAivNum, tiling, plan));
    EXPECT_FALSE(ComputeRoiAlignRotatedGradTilingCore(1, 8, 32, 32, 10, 7, 7, 0,
                                                      std::numeric_limits<float>::quiet_NaN(), kAivNum, tiling, plan));
}

TEST(RoiAlignRotatedGradTiling, RejectsOverflowingInputSize)
{
    ROIAlignRotatedGradTilingData tiling{};
    RoiAlignRotatedGradHost::RoiAlignRotatedGradPlan plan;
    const int64_t kMax = std::numeric_limits<int64_t>::max();
    EXPECT_FALSE(ComputeRoiAlignRotatedGradTilingCore(2, kMax, 32, 32, 10, 7, 7, 0, 1.0F, kAivNum, tiling, plan));
}
