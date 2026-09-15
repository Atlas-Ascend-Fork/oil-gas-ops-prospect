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
 * 对齐 ops-nn `<op>/tests/ut/op_host`。与平台无关，无卡可跑。重点覆盖：
 * MIX/VEC 路径分发（tiling key 0-3）、
 * UB 预算约束（rowsPerIter、V plane 驻留上限）、对齐计算、非法输入拒绝。
 * 期望值均为手算常数，不与实现共用计算路径。
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "fused_softmax_grad_tiling_core.h"

namespace {

constexpr uint64_t kUbSize = 196608; // 910B AIV UB

TEST(FusedSoftmaxGradCore, Bf16AlignedGoesMix)
{
    // bf16 [8,8,256,256] D64：S/D 均 16 对齐 → MIX（key=3），即使 V plane 能进 UB
    // rowsPerIter = (196608-65536)/(16*256) = 131072/4096 = 32
    // gBytes = 8*8*256*256*2 = 8MB
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        8, 8, 256, 256, 64, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));

    EXPECT_EQ(params.useMix, 1u);
    EXPECT_EQ(params.padS, 256u);
    EXPECT_EQ(params.padD, 64u);
    EXPECT_EQ(params.rowsPerIter, 32u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_BF16_MIX);
    EXPECT_EQ(plan.gBytes, 8388608u);
}

TEST(FusedSoftmaxGradCore, Bf16AlignedGoesMixWhenVecWouldFit)
{
    // bf16 [2,4,64,64] D64：M=N=K，padD 再垫 16 → 80（D-pad 路径）
    // rowsPerIter = 131072/(16*64) = 128
    // g = 8*64*64*2；A/B pad = 8*(64*80+64*80)*2
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        2, 4, 64, 64, 64, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));

    EXPECT_EQ(params.useMix, 1u);
    EXPECT_EQ(params.padS, 64u);
    EXPECT_EQ(params.padD, 80u);
    EXPECT_EQ(params.rowsPerIter, 128u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_BF16_MIX);
    EXPECT_EQ(plan.gBytes, 65536u + 163840u);
}

TEST(FusedSoftmaxGradCore, Bf16AlignedMixWhenPlatformUnderreportsUb)
{
    // platform 回报 64KB 时仍按 192KB 预算走 MIX，避免 avail=0 被拒
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        8, 8, 256, 256, 64, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, 65536, params, plan));

    EXPECT_EQ(params.useMix, 1u);
    EXPECT_EQ(params.rowsPerIter, 32u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_BF16_MIX);
}

TEST(FusedSoftmaxGradCore, Bf16AlignedWideGoesMix)
{
    // bf16 [1,1,256,256] D256：M=N=K，padD=272
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 256, 256, 256, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));

    EXPECT_EQ(params.useMix, 1u);
    EXPECT_EQ(params.padS, 256u);
    EXPECT_EQ(params.padD, 272u);
    EXPECT_EQ(params.rowsPerIter, 32u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_BF16_MIX);
    EXPECT_EQ(plan.gBytes, 131072u + 278528u);
}

TEST(FusedSoftmaxGradCore, Bf16NonAlignedDGoesMixWithPad)
{
    // bf16 D=20：pad 后走 MIX。g = 4*128*128*2；A/B pad = 4*(128*32+128*32)*2
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        2, 2, 128, 128, 20, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));

    EXPECT_EQ(params.useMix, 1u);
    EXPECT_EQ(params.padS, 128u);
    EXPECT_EQ(params.padD, 32u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_BF16_MIX);
    EXPECT_EQ(plan.gBytes, 131072u + 65536u);
}

TEST(FusedSoftmaxGradCore, Bf16NonAlignedSGoesMixWithPad)
{
    // bf16 S=20 D=16：B 按 padS=32 行。g = 16*32*2；A/B pad = (16*16+32*16)*2
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 16, 20, 16, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));

    EXPECT_EQ(params.useMix, 1u);
    EXPECT_EQ(params.padS, 32u);
    EXPECT_EQ(params.padD, 16u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_BF16_MIX);
    EXPECT_EQ(plan.gBytes, 1024u + 1536u);
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
    // MIX：rowsPerIter 与 padS 成反比（16B/行元素预算）
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams smallParams;
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams wideParams;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 16, 16, 16, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, smallParams, plan));
    EXPECT_EQ(smallParams.useMix, 1u);
    EXPECT_EQ(smallParams.padS, 16u);
    EXPECT_EQ(smallParams.padD, 32u);         // M=N=K 再垫 K
    EXPECT_EQ(smallParams.rowsPerIter, 512u); // 131072/(16*16)
    EXPECT_EQ(plan.gBytes, 512u + 2048u);

    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 64, 256, 256, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, wideParams, plan));
    EXPECT_EQ(wideParams.useMix, 1u);
    EXPECT_EQ(wideParams.rowsPerIter, 32u); // 131072/(16*256)
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
    // Q≠S 合法（openfold MSA 形态 [2,4,64,128]）：S/D 对齐 → MIX
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        2, 4, 64, 128, 32, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));
    EXPECT_EQ(params.useMix, 1u);
    EXPECT_EQ(params.padS, 128u);
    EXPECT_EQ(plan.tilingKey, FusedSoftmaxGradHost::FSG_KEY_BF16_MIX);
    EXPECT_EQ(plan.gBytes, 2u * 4 * 64 * 128 * 2);
}

} // namespace
