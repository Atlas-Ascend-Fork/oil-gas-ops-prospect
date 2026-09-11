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
 * FusedBiasSoftmaxOg kernel 入口（fused_bias_softmax）。
 */

/*!
 * \file fused_bias_softmax.cpp
 * \brief
 */

#include "FusedBiasSoftmax.h"

extern "C" __global__ __aicore__ void fused_bias_softmax(GM_ADDR x, GM_ADDR bias1, GM_ADDR bias2, GM_ADDR y,
                                                         GM_ADDR workspace, GM_ADDR tiling) {
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
