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
 * FusedBiasSoftmax kernel 入口（fused_bias_softmax）。
 */

/*!
 * \file fused_bias_softmax.cpp
 * \brief
 */

#include "FusedBiasSoftmax.h"

extern "C" __global__ __aicore__ void fused_bias_softmax(GM_ADDR x, GM_ADDR bias1, GM_ADDR bias2, GM_ADDR y,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        AscendC::FusedBiasSoftmax<float> op;
        op.Init(x, bias1, bias2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        AscendC::FusedBiasSoftmax<half> op;
        op.Init(x, bias1, bias2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        AscendC::FusedBiasSoftmax<bfloat16_t> op;
        op.Init(x, bias1, bias2, y, tilingData);
        op.Process();
    }
}
