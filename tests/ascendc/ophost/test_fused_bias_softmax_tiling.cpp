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
 * FusedBiasSoftmax ophost UT：切分算法核心（fused_bias_softmax_tiling_core.h）。
 *
 * 与平台无关，无卡可跑。期望值均为手算常数（推导见各用例注释），不与实现共用
 * 计算路径，避免同义反复。
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "fused_bias_softmax_tiling_core.h"

namespace {

constexpr uint64_t kUbSize = 196608; // 910B AIV UB
constexpr uint64_t kAivNum = 48;

TEST(FusedBiasSoftmaxCore, Bf16DualBiasBroadcastTypical)
{
    // [2,4,64,128] bf16 双 bias（bias1 广播）：openfold triangle attention 典型形态
    // padLineNum = 128（128*2B=256B 已 32B 对齐）
    // maxByteLine = 128*(2*2 + 4 + (2+4)*2) + 2*128*2 + 128*4 = 2560 + 1024 = 3584
    // rowsPerIter = (196608-32768)/3584 = 45（向下取整）
    // totalLine = 512 = 48*10 + 32 → 大核 32 个各 11 行，小核 16 个各 10 行
    FusedBiasSoftmaxHost::FusedBiasSoftmaxCoreParams params;
    FusedBiasSoftmaxHost::FusedBiasSoftmaxPlan plan;
    ASSERT_TRUE(FusedBiasSoftmaxHost::ComputeFusedBiasSoftmaxTilingCore(
        2, 4, 64, 128, 2, FusedBiasSoftmaxHost::FBS_KEY_BF16, 1, 1, 0, kUbSize, kAivNum, params, plan));

    EXPECT_EQ(params.paddingNum, 0u);
    EXPECT_EQ(params.padLineNum, 128u);
    EXPECT_EQ(params.coreNum, 48u);
    EXPECT_EQ(params.headCoreNum, 32u);
    EXPECT_EQ(params.lineHeadCore, 11u);
    EXPECT_EQ(params.iterHeadCore, 1u);
    EXPECT_EQ(params.lineHeadIter, 45u);
    EXPECT_EQ(params.lineLastHeadIter, 11u);
    EXPECT_EQ(params.lineTailCore, 10u);
    EXPECT_EQ(params.iterTailCore, 1u);
    EXPECT_EQ(params.lineLastTailIter, 10u);
    EXPECT_EQ(plan.blockDim, 48u);
    EXPECT_EQ(plan.tilingKey, FusedBiasSoftmaxHost::FBS_KEY_BF16);
}

TEST(FusedBiasSoftmaxCore, Fp32NoBias)
{
    // [2,4,64,64] fp32 无 bias：fp32 无 xF32 中转项，maxByteLine = 64*(2*4) = 512，
    // rowsPerIter = (196608-32768)/512 = 320
    FusedBiasSoftmaxHost::FusedBiasSoftmaxCoreParams params;
    FusedBiasSoftmaxHost::FusedBiasSoftmaxPlan plan;
    ASSERT_TRUE(FusedBiasSoftmaxHost::ComputeFusedBiasSoftmaxTilingCore(
        2, 4, 64, 64, 4, FusedBiasSoftmaxHost::FBS_KEY_FP32, 0, 0, 0, kUbSize, kAivNum, params, plan));

    EXPECT_EQ(params.padLineNum, 64u);
    EXPECT_EQ(params.paddingNum, 0u);
    EXPECT_EQ(params.headCoreNum, 32u);
    EXPECT_EQ(params.lineHeadIter, 320u);
    EXPECT_EQ(params.lineLastHeadIter, 11u);
    EXPECT_EQ(plan.tilingKey, FusedBiasSoftmaxHost::FBS_KEY_FP32);
}

TEST(FusedBiasSoftmaxCore, Fp32NonAlignedWidthPadsTo32B)
{
    // width=41 fp32：41%8=1 → padding 7，padLine 48（192B，32B 对齐）
    FusedBiasSoftmaxHost::FusedBiasSoftmaxCoreParams params;
    FusedBiasSoftmaxHost::FusedBiasSoftmaxPlan plan;
    ASSERT_TRUE(FusedBiasSoftmaxHost::ComputeFusedBiasSoftmaxTilingCore(
        1, 1, 41, 41, 4, FusedBiasSoftmaxHost::FBS_KEY_FP32, 0, 0, 0, kUbSize, kAivNum, params, plan));

    EXPECT_EQ(params.paddingNum, 7u);
    EXPECT_EQ(params.padLineNum, 48u);
}

TEST(FusedBiasSoftmaxCore, FullBias1HasNoBcastOverhead)
{
    // bias1 全尺寸 vs 广播：全尺寸不含 bcast 附加项，rowsPerIter 更大
    // 全尺寸: maxByteLine = 128*(4+4+12) = 2560 → rows = 163840/2560 = 64
    // 广播:   maxByteLine = 3584 → rows = 45
    FusedBiasSoftmaxHost::FusedBiasSoftmaxCoreParams fullParams, bcastParams;
    FusedBiasSoftmaxHost::FusedBiasSoftmaxPlan plan;
    ASSERT_TRUE(FusedBiasSoftmaxHost::ComputeFusedBiasSoftmaxTilingCore(
        2, 4, 64, 128, 2, FusedBiasSoftmaxHost::FBS_KEY_BF16, 1, 1, 1, kUbSize, kAivNum, fullParams, plan));
    ASSERT_TRUE(FusedBiasSoftmaxHost::ComputeFusedBiasSoftmaxTilingCore(
        2, 4, 64, 128, 2, FusedBiasSoftmaxHost::FBS_KEY_BF16, 1, 1, 0, kUbSize, kAivNum, bcastParams, plan));
    EXPECT_EQ(fullParams.lineHeadIter, 64u);
    EXPECT_EQ(bcastParams.lineHeadIter, 45u);
    EXPECT_GT(fullParams.lineHeadIter, bcastParams.lineHeadIter);
}

TEST(FusedBiasSoftmaxCore, RejectsInvalidInputs)
{
    FusedBiasSoftmaxHost::FusedBiasSoftmaxCoreParams params;
    FusedBiasSoftmaxHost::FusedBiasSoftmaxPlan plan;

    // width 超上限 4096
    EXPECT_FALSE(FusedBiasSoftmaxHost::ComputeFusedBiasSoftmaxTilingCore(
        1, 1, 8, 5000, 2, FusedBiasSoftmaxHost::FBS_KEY_BF16, 0, 0, 0, kUbSize, kAivNum, params, plan));
    // 核数为 0
    EXPECT_FALSE(FusedBiasSoftmaxHost::ComputeFusedBiasSoftmaxTilingCore(
        1, 1, 8, 128, 2, FusedBiasSoftmaxHost::FBS_KEY_BF16, 0, 0, 0, kUbSize, 0, params, plan));
    // height 为 0
    EXPECT_FALSE(FusedBiasSoftmaxHost::ComputeFusedBiasSoftmaxTilingCore(
        1, 1, 0, 128, 2, FusedBiasSoftmaxHost::FBS_KEY_BF16, 0, 0, 0, kUbSize, kAivNum, params, plan));
    // UB 过小（减去 SoftMax 32K 后不足一行）
    EXPECT_FALSE(FusedBiasSoftmaxHost::ComputeFusedBiasSoftmaxTilingCore(
        1, 1, 8, 128, 2, FusedBiasSoftmaxHost::FBS_KEY_BF16, 0, 0, 0, 33000, kAivNum, params, plan));
    // dtypeSize 为 0
    EXPECT_FALSE(FusedBiasSoftmaxHost::ComputeFusedBiasSoftmaxTilingCore(
        1, 1, 8, 128, 0, FusedBiasSoftmaxHost::FBS_KEY_BF16, 0, 0, 0, kUbSize, kAivNum, params, plan));
}

TEST(FusedBiasSoftmaxCore, Fp16UsesFp16Key)
{
    FusedBiasSoftmaxHost::FusedBiasSoftmaxCoreParams params;
    FusedBiasSoftmaxHost::FusedBiasSoftmaxPlan plan;
    ASSERT_TRUE(FusedBiasSoftmaxHost::ComputeFusedBiasSoftmaxTilingCore(
        2, 4, 64, 64, 2, FusedBiasSoftmaxHost::FBS_KEY_FP16, 1, 1, 0, kUbSize, kAivNum, params, plan));
    EXPECT_EQ(plan.tilingKey, FusedBiasSoftmaxHost::FBS_KEY_FP16);
}

} // namespace
