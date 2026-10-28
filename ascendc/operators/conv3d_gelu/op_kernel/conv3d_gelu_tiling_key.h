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
 * Conv3DGelu tiling key.
 *
 * 模板参数与 conv3d 完全一致（ConvL0PingPong / ConvBL1ByPass / GroupConvType /
 * OutputOrder），因为 cube 阶段直接复用 conv3d_v2 的 kernel 分派。TPL 标识符独立
 * 命名（Conv3DGelu），避免与 conv3d（Conv3DCustom）同时编译时冲突。
 */

#ifndef CONV3D_GELU_TILING_KEY_H
#define CONV3D_GELU_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace Conv3dGeluTilingKey {

#define L0_PINGPONG_ALL_CLOSE 0
#define L0_PINGPONG_L0A_OPEN 1
#define L0_PINGPONG_L0B_OPEN 2
#define L0_PINGPONG_ALL_OPEN 3

#define L1_BYPASS_OFF 0
#define L1_BYPASS_ON 1

#define NOGROUP_CONV 0
#define GROUPCONV_WEIGHT_GFZ 1

#define M_MODE 0
#define HW_MODE 1

ASCENDC_TPL_ARGS_DECL(
    Conv3DGelu,
    ASCENDC_TPL_UINT_DECL(ConvL0PingPong, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, L0_PINGPONG_ALL_CLOSE,
                          L0_PINGPONG_L0A_OPEN, L0_PINGPONG_L0B_OPEN, L0_PINGPONG_ALL_OPEN),
    ASCENDC_TPL_UINT_DECL(ConvBL1ByPass, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, L1_BYPASS_OFF, L1_BYPASS_ON),
    ASCENDC_TPL_UINT_DECL(GroupConvType, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, NOGROUP_CONV, GROUPCONV_WEIGHT_GFZ),
    ASCENDC_TPL_UINT_DECL(OutputOrder, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, M_MODE, HW_MODE));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(ConvL0PingPong, ASCENDC_TPL_UI_LIST, L0_PINGPONG_ALL_CLOSE,
                                              L0_PINGPONG_L0A_OPEN, L0_PINGPONG_L0B_OPEN, L0_PINGPONG_ALL_OPEN),
                         ASCENDC_TPL_UINT_SEL(ConvBL1ByPass, ASCENDC_TPL_UI_LIST, L1_BYPASS_OFF, L1_BYPASS_ON),
                         ASCENDC_TPL_UINT_SEL(GroupConvType, ASCENDC_TPL_UI_LIST, NOGROUP_CONV, GROUPCONV_WEIGHT_GFZ),
                         ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST, M_MODE)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(ConvL0PingPong, ASCENDC_TPL_UI_LIST, L0_PINGPONG_ALL_CLOSE,
                                              L0_PINGPONG_L0A_OPEN, L0_PINGPONG_L0B_OPEN, L0_PINGPONG_ALL_OPEN),
                         ASCENDC_TPL_UINT_SEL(ConvBL1ByPass, ASCENDC_TPL_UI_LIST, L1_BYPASS_OFF, L1_BYPASS_ON),
                         ASCENDC_TPL_UINT_SEL(GroupConvType, ASCENDC_TPL_UI_LIST, NOGROUP_CONV),
                         ASCENDC_TPL_UINT_SEL(OutputOrder, ASCENDC_TPL_UI_LIST, HW_MODE)));

} // namespace Conv3dGeluTilingKey

#endif // CONV3D_GELU_TILING_KEY_H
