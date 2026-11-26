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
 * ROIAlignRotatedGrad kernel 入口（AIV-only）。
 */

#include "kernel_operator.h"
#include "og_index_kernel.h"
#include "roi_align_rotated_grad_tiling.h"
#include "roi_align_rotated_grad_impl.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void roi_align_rotated_grad(GM_ADDR grad_output, GM_ADDR rois, GM_ADDR grad_input,
                                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;

    ROIAlignRotatedGradTilingData tilingData;
    OgIndex::CopyTilingData(&tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        TPipe pipe;
        RoiAlignRotatedGrad::RoiAlignRotatedGradKernel op;
        op.Init(grad_output, rois, grad_input, tilingData, &pipe);
        op.Process();
    }
}
