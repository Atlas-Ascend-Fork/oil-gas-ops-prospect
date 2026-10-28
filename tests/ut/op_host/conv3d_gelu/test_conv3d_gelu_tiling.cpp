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
 * Conv3DGelu ophost UT：路径选择（conv3d_gelu_route.h）。
 *
 * 完整 tiling 引擎依赖 PlatformAscendC，无卡 UT 编不进去。这里覆盖 host 与
 * aclnn 共用的两条 shape 条件：1×1 直写，以及小通道全分辨率 3×3 的 HW 切分。
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "conv3d_gelu_route.h"

namespace {

constexpr int64_t kZeroPad[6] = {0, 0, 0, 0, 0, 0};

bool Direct1x1(int64_t cin, int64_t cout, int64_t groups = 1, int64_t kd = 1, int64_t stride = 1, int64_t dilation = 1,
               const int64_t pads[6] = kZeroPad)
{
    return Conv3dGeluUseDirect1x1(cin, cout, groups, kd, kd, kd, stride, stride, stride, dilation, dilation, dilation,
                                  pads);
}

} // namespace

TEST(Conv3dGeluRoute, Direct1x1AcceptsMeasuredFastRange)
{
    EXPECT_TRUE(Direct1x1(8, 64));
    EXPECT_TRUE(Direct1x1(32, 32));
    EXPECT_TRUE(Direct1x1(128, 64));
}

TEST(Conv3dGeluRoute, Direct1x1RejectsSlowOrNonPointwise)
{
    EXPECT_FALSE(Direct1x1(1, 32));
    EXPECT_FALSE(Direct1x1(7, 16));
    EXPECT_FALSE(Direct1x1(32, 65));
    EXPECT_FALSE(Direct1x1(256, 512));
    EXPECT_FALSE(Direct1x1(16, 16, /*groups=*/2));
    EXPECT_FALSE(Direct1x1(16, 16, 1, /*kd=*/3));
    EXPECT_FALSE(Direct1x1(16, 16, 1, 1, /*stride=*/2));
    EXPECT_FALSE(Direct1x1(16, 16, 1, 1, 1, /*dilation=*/2));
    const int64_t padded[6] = {0, 0, 1, 1, 0, 0};
    EXPECT_FALSE(Direct1x1(16, 16, 1, 1, 1, 1, padded));
}

TEST(Conv3dGeluRoute, HwModeOnlyForSmallChannelFullRes3x3)
{
    EXPECT_TRUE(Conv3dGeluPreferHwMode(false, 1, 3, 3, 3, 7, 7, 160, 288));
    EXPECT_TRUE(Conv3dGeluPreferHwMode(false, 1, 3, 3, 3, 16, 16, 128, 128));
    EXPECT_FALSE(Conv3dGeluPreferHwMode(true, 1, 3, 3, 3, 8, 8, 160, 160));
    EXPECT_FALSE(Conv3dGeluPreferHwMode(false, 2, 3, 3, 3, 8, 8, 160, 160));
    EXPECT_FALSE(Conv3dGeluPreferHwMode(false, 1, 5, 5, 5, 8, 8, 160, 160));
    EXPECT_FALSE(Conv3dGeluPreferHwMode(false, 1, 3, 3, 3, 17, 8, 160, 160));
    EXPECT_FALSE(Conv3dGeluPreferHwMode(false, 1, 3, 3, 3, 8, 32, 160, 160));
    EXPECT_FALSE(Conv3dGeluPreferHwMode(false, 1, 3, 3, 3, 8, 8, 127, 160));
    EXPECT_FALSE(Conv3dGeluPreferHwMode(false, 1, 3, 3, 3, 8, 8, 160, 127));
}
