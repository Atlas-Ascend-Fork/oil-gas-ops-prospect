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
 * Conv3DGelu kernel entry — conv3d + GELU 融合（cube + vector）。
 *
 * 融合策略（清晰区分 cube / vector，思路同 FFN 的 matmul→激活）：
 *   1) Cube 阶段：复用 conv3d_v2 的高性能 cube kernel，conv.IterateAll 把卷积结果
 *      写入输出 y（910b / float32，x/y 为 NDC1HWC0，filter 为 FRACTAL_Z_3D）。
 *   2) SyncAll<false>()：同步所有 AIC + AIV 核，确保 conv 结果全部写完且跨核可见。
 *   3) Vector 阶段：AIV 核对 y 做原地逐元素 tanh 近似 GELU。
 *
 * conv kernel 自身零改动；gelu 为逐元素运算，原地写回安全。
 */

#include "conv3dv2.h"
#include "lib/matmul_intf.h"
#include "conv3d_gelu_tiling_key.h"
#include "conv3d_v2_kernel_dispatch.h"
#include "conv3d_gelu_tiling_data.h"
#include "conv3d_gelu_vec.h"

using namespace AscendC;

#if defined(FORMAT_X) && FORMAT_X == FORMAT_NCDHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_NCDHW && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCDHW
constexpr ConvFormat aFormat = ConvFormat::NCDHW;
constexpr ConvFormat bFormat = ConvFormat::NCDHW;
constexpr ConvFormat cFormat = ConvFormat::NCDHW;
constexpr ConvBL1ByPass bL1ByPassFlag = ConvBL1ByPass::BYPASS_OFF;
constexpr QuantType quantType = QuantType::NO_QUANT;
#elif defined(ORIG_DTYPE_X) && ORIG_DTYPE_X != DT_INT8
constexpr ConvFormat aFormat = ConvFormat::NDC1HWC0;
constexpr ConvFormat bFormat = ConvFormat::FRACTAL_Z_3D;
constexpr ConvFormat cFormat = ConvFormat::NDC1HWC0;
constexpr ConvBL1ByPass bL1ByPassFlag = ConvBL1ByPass::BYPASS_ON;
constexpr QuantType quantType = QuantType::NO_QUANT;
#else
constexpr ConvFormat aFormat = ConvFormat::NDC1HWC0;
constexpr ConvFormat bFormat = ConvFormat::FRACTAL_Z_3D;
constexpr ConvFormat cFormat = ConvFormat::NCDHW;
constexpr ConvBL1ByPass bL1ByPassFlag = ConvBL1ByPass::BYPASS_ON;
constexpr QuantType quantType = QuantType::PER_CHANNEL_NO_OFFSET;
#endif

template <uint8_t ConvL0PingPongT, uint8_t BL1ByPassFlagT, uint8_t GroupConvTypeT, uint8_t OutputOrderT>
__global__ __aicore__ void conv3d_gelu(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                       GM_ADDR tiling)
{
    ASC_OP_LOGD("[Conv3DGelu] Begin to process conv3d_gelu.\n");
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR user_workspace = GetUserWorkspace(workspace);
    REGISTER_TILING_DEFAULT(Conv3DGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(Conv3DGeluTilingData, tilingData, tiling);

    using A_TYPE = ConvType<TPosition::GM, aFormat, DTYPE_X>;
    using B_TYPE = ConvType<TPosition::GM, bFormat, DTYPE_FILTER>;
    using C_TYPE = ConvType<TPosition::GM, cFormat, DTYPE_Y>;

#if defined(DTYPE_BIAS)
    using BIAS_TYPE = ConvType<TPosition::GM, ConvFormat::ND, DTYPE_BIAS>;
#else
    using BIAS_TYPE = ConvType<TPosition::GM, ConvFormat::ND, float>;
#endif

    // ---- Cube 阶段：conv3d 写入 y（pipe 作用域内创建/销毁，便于随后 gelu 重建 TPipe）----
    {
        DispatchConv3DV2Kernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, static_cast<ConvL0PingPong>(ConvL0PingPongT),
                               static_cast<ConvBL1ByPass>(BL1ByPassFlagT), static_cast<GroupConvType>(GroupConvTypeT),
                               static_cast<OutputOrder>(OutputOrderT), quantType>(
            x, filter, bias, /*scale=*/nullptr, /*offset=*/nullptr, y, user_workspace, &tilingData.convTiling);
    }

    // ---- 同步所有 AIC + AIV 核：确保 conv 结果全部落盘且跨核可见 ----
    SyncAll<false>();

    // ---- Vector 阶段：AIV 核对 y 原地做 GELU ----
    if ASCEND_IS_AIV {
        Conv3dGeluVec::GeluInplace<DTYPE_Y> op;
        op.Init(y, tilingData.geluTotalNum, tilingData.geluBlockFactor, tilingData.geluUbFactor);
        op.Process();
    }
    ASC_OP_LOGD("[Conv3DGelu] process conv3d_gelu success.\n");
}
