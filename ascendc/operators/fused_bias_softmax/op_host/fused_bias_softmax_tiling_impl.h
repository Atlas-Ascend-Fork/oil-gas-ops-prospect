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
 * FusedBiasSoftmax 切分：platform 读取与 SoftMax 高阶 api tiling 在此桥接，
 * 平台无关的切分算法在 fused_bias_softmax_tiling_core.h（ophost UT 覆盖）。
 */

#ifndef FUSED_BIAS_SOFTMAX_TILING_IMPL_H
#define FUSED_BIAS_SOFTMAX_TILING_IMPL_H

#include <cstdint>

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "fused_bias_softmax_tiling_core.h"
#include "fused_bias_softmax_tiling_data_def.h"

namespace FusedBiasSoftmaxHost {

struct FusedBiasSoftmaxShape {
    uint64_t batch = 0;
    uint64_t channel = 0;
    uint64_t height = 0;
    uint64_t width = 0;
    uint64_t dtypeSize = 0;
    uint64_t dtypeKey = 0; // 0=fp32, 1=fp16, 2=bf16
    uint64_t hasB1 = 0;
    uint64_t hasB2 = 0;
    uint64_t b1Full = 0;
};

/** platform 读取并委托 core；结果写入 tiling 数据结构。 */
inline bool ComputeFusedBiasSoftmaxTiling(const FusedBiasSoftmaxShape& shape,
                                          platform_ascendc::PlatformAscendC& platform,
                                          optiling::FusedBiasSoftmaxTilingData& tilingData, FusedBiasSoftmaxPlan& plan)
{
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        return false;
    }
    const uint64_t aivNum = platform.GetCoreNumAiv();
    if (aivNum == 0) {
        return false;
    }

    FusedBiasSoftmaxCoreParams params;
    if (!ComputeFusedBiasSoftmaxTilingCore(shape.batch, shape.channel, shape.height, shape.width, shape.dtypeSize,
                                           shape.dtypeKey, shape.hasB1, shape.hasB2, shape.b1Full, ubSize, aivNum,
                                           params, plan)) {
        return false;
    }

    tilingData.set_coreNum(params.coreNum);
    tilingData.set_batch(shape.batch);
    tilingData.set_channel(shape.channel);
    tilingData.set_height(shape.height);
    tilingData.set_width(shape.width);
    tilingData.set_hasB1(shape.hasB1);
    tilingData.set_hasB2(shape.hasB2);
    tilingData.set_b1Full(shape.b1Full);
    tilingData.set_qPerPlane(shape.height);
    tilingData.set_paddingNum(params.paddingNum);
    tilingData.set_padLineNum(params.padLineNum);
    tilingData.set_headCoreNum(params.headCoreNum);
    tilingData.set_lineHeadCore(params.lineHeadCore);
    tilingData.set_iterHeadCore(params.iterHeadCore);
    tilingData.set_lineHeadIter(params.lineHeadIter);
    tilingData.set_lineLastHeadIter(params.lineLastHeadIter);
    tilingData.set_lineTailCore(params.lineTailCore);
    tilingData.set_iterTailCore(params.iterTailCore);
    tilingData.set_lineTailIter(params.lineTailIter);
    tilingData.set_lineLastTailIter(params.lineLastTailIter);

    // SoftMax 高阶 api tiling（host 侧 API）
    const ge::Shape softmaxShape({static_cast<int64_t>(params.lineHeadIter), static_cast<int64_t>(params.padLineNum)});
    uint64_t softmaxTmpSize = AscendC::GetSoftMaxMaxTmpSize(softmaxShape, FBS_FP32_SIZE, false);
    if (softmaxTmpSize > FBS_SOFTMAX_BUF_SIZE) {
        softmaxTmpSize = FBS_SOFTMAX_BUF_SIZE;
    }
    AscendC::SoftMaxTilingFunc(softmaxShape, FBS_FP32_SIZE, softmaxTmpSize, tilingData.softmaxTilingData);
    return true;
}

} // namespace FusedBiasSoftmaxHost

#endif // FUSED_BIAS_SOFTMAX_TILING_IMPL_H
