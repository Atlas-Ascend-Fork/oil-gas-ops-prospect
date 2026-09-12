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
 * FusedSoftmaxGrad ophost UT：切分算法核心（fused_softmax_grad_tiling_core.h）。
 *
 * 与平台无关，无卡可跑。重点覆盖：MIX/VEC 路径分发（tiling key 0-3）、
 * UB 预算约束（rowsPerIter、V plane 驻留上限）、对齐计算、非法输入拒绝。
 * 期望值均为手算常数，不与实现共用计算路径。
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "fused_softmax_grad_tiling_core.h"

namespace {

constexpr uint64_t kUbSize = 196608; // 910B AIV UB

TEST(FusedSoftmaxGradCore, Bf16AlignedFitsVecPrefersVec)
{
    // bf16 [8,8,256,256] D64：S/D 对齐，但 V plane 65KB+余量仍进 UB → 走 VEC（key=2）
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        8, 8, 256, 256, 64, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));

    EXPECT_EQ(params.useMix, 0u);
    EXPECT_EQ(params.padS, 256u);
    EXPECT_EQ(params.padD, 64u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC);
    EXPECT_EQ(plan.gBytes, 0u);
}

TEST(FusedSoftmaxGradCore, Bf16AlignedGoesMixWhenVecOverflows)
{
    // bf16 [1,1,256,256] D256：V plane 256KB 超 UB → MIX（key=3）
    // rowsPerIter = (196608-65536)/(16*256) = 131072/4096 = 32
    // gBytes = 1*1*256*256*2 = 128KB
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 256, 256, 256, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));

    EXPECT_EQ(params.useMix, 1u);
    EXPECT_EQ(params.padS, 256u);
    EXPECT_EQ(params.padD, 256u);
    EXPECT_EQ(params.rowsPerIter, 32u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_BF16_MIX);
    EXPECT_EQ(plan.gBytes, 131072u);
}

TEST(FusedSoftmaxGradCore, Bf16NonAlignedDGoesVector)
{
    // bf16 D=20（非 16 对齐）→ 向量路径（key=2），padD = alignUp(20,16) = 32
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        2, 2, 128, 128, 20, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));

    EXPECT_EQ(params.useMix, 0u);
    EXPECT_EQ(params.padS, 128u);
    EXPECT_EQ(params.padD, 32u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC);
    EXPECT_EQ(plan.gBytes, 0u);
}

TEST(FusedSoftmaxGradCore, Fp32AlwaysVector)
{
    // fp32：即使 S/D 对齐也走向量（fixpipe C 16bit 结构上限），pad 按 8 对齐
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        2, 4, 64, 64, 64, FusedSoftmaxGradHost::FSG_KEY_FP32_VEC, kUbSize, params, plan));

    EXPECT_EQ(params.useMix, 0u);
    EXPECT_EQ(params.padS, 64u);
    EXPECT_EQ(params.padD, 64u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_FP32_VEC);
}

TEST(FusedSoftmaxGradCore, Fp16AlwaysVector)
{
    // fp16：契约 rtol=5e-4 与 half-g 量化误差过近，强制向量路径
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        2, 4, 64, 64, 32, FusedSoftmaxGradHost::FSG_KEY_FP16_VEC, kUbSize, params, plan));

    EXPECT_EQ(params.useMix, 0u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_FP16_VEC);
    EXPECT_EQ(params.padD, 32u);
}

TEST(FusedSoftmaxGradCore, VecPathRejectsVPlaneBeyondUb)
{
    // fp32 [1,1,256,256] D256：V plane 驻留 256*256*4 = 256KB > UB 预算 → 拒绝
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    EXPECT_FALSE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 256, 256, 256, FusedSoftmaxGradHost::FSG_KEY_FP32_VEC, kUbSize, params, plan));
}

TEST(FusedSoftmaxGradCore, MixRowsPerIterShrinksWithPadS)
{
    // MIX 仅在 VEC 预算溢出时启用。用两个超 UB 的对齐形状验证 rowsPerIter ∝ 1/padS。
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams wideParams;
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams tallParams;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    // S=128 D=256：V=128KB+余量仍进 UB，这里改用 S=256 D=256（V=256KB）
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 64, 256, 256, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, wideParams, plan));
    EXPECT_EQ(wideParams.useMix, 1u);
    EXPECT_EQ(wideParams.rowsPerIter, 32u); // 131072/(16*256)
    // S=128 D=512：V=256KB → MIX，padS=128 → rowsPerIter = 131072/(16*128) = 64
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 64, 128, 512, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, tallParams, plan));
    EXPECT_EQ(tallParams.useMix, 1u);
    EXPECT_EQ(tallParams.rowsPerIter, 64u);
}

TEST(FusedSoftmaxGradCore, RejectsInvalidInputs)
{
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;

    EXPECT_FALSE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 0, 64, 64, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan)); // Q=0
    EXPECT_FALSE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        0, 1, 64, 64, 64, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan)); // batch=0
    EXPECT_FALSE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 64, 64, 0, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan)); // D=0
    EXPECT_FALSE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 64, 64, 64, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, 0, params, plan)); // UB=0
}

TEST(FusedSoftmaxGradCore, QNotEqualSAllowed)
{
    // Q≠S 合法（openfold MSA 形态 [2,4,64,128]）：V plane 能进 UB，走 VEC
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        2, 4, 64, 128, 32, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));
    EXPECT_EQ(params.useMix, 0u);
    EXPECT_EQ(params.padS, 128u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC);
    EXPECT_EQ(plan.gBytes, 0u);
}

} // namespace
