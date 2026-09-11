/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * FusedSoftmaxGradOg ophost UT：切分算法核心（fused_softmax_grad_tiling_core.h）。
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

TEST(FusedSoftmaxGradCore, Bf16AlignedGoesMix)
{
    // bf16 [8,8,256,256] D64：S/D 均 16 对齐 → MIX（key=3）
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
    // MIX：rowsPerIter 与 padS 成反比（16B/行元素预算）
    // [1,1,16,16] D16: padS=16 → rowsPerIter = 131072/(16*16) = 512
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams smallParams;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 16, 16, 16, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, smallParams, plan));
    EXPECT_EQ(smallParams.rowsPerIter, 512u);
    EXPECT_EQ(plan.gBytes, 512u);
}

TEST(FusedSoftmaxGradCore, RejectsInvalidInputs)
{
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;

    EXPECT_FALSE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 0, 64, 64, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));  // Q=0
    EXPECT_FALSE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        0, 1, 64, 64, 64, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));  // batch=0
    EXPECT_FALSE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 64, 64, 0, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));   // D=0
    EXPECT_FALSE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        1, 1, 64, 64, 64, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, 0, params, plan));        // UB=0
}

TEST(FusedSoftmaxGradCore, QNotEqualSAllowed)
{
    // Q≠S 合法（openfold MSA 形态 [2,4,64,128]）：MIX 路径正常分发
    FusedSoftmaxGradHost::FusedSoftmaxGradCoreParams params;
    FusedSoftmaxGradHost::FusedSoftmaxGradPlan plan;
    ASSERT_TRUE(FusedSoftmaxGradHost::ComputeFusedSoftmaxGradTilingCore(
        2, 4, 64, 128, 32, FusedSoftmaxGradHost::FSG_KEY_BF16_VEC, kUbSize, params, plan));
    EXPECT_EQ(params.useMix, 1u);
    EXPECT_EQ(params.padS, 128u);
    EXPECT_EQ(plan.gBytes, 2u * 4 * 64 * 128 * 2);
}

} // namespace
