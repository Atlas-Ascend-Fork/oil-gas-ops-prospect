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
 * Conv3DGelu 自定义 L2 aclnn 接口（对齐 native conv3d 的做法）。
 *
 * 与自动生成的 aclnnConv3DGelu 不同：本接口**直接接收 NCDHW 的 x/filter/y**，
 * 在同一个 executor 内用 l0op::TransData 完成
 *   x:    NCDHW       -> NDC1HWC0
 *   filter: NCDHW     -> FRACTAL_Z_3D
 *   y:    NDC1HWC0    -> NCDHW
 * 并在中间 launch Conv3DGelu（cube conv + vector gelu）。所有格式转换的中间
 * buffer 都由 executor 统一管理/复用，调用侧（torch_npu）无需任何手动 transdata。
 */

#ifndef ACLNN_CONV3D_GELU_CUSTOM_H
#define ACLNN_CONV3D_GELU_CUSTOM_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnConv3dGeluCustomGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* biasOptional, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, int64_t groups, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnConv3dGeluCustomV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* biasOptional, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, int64_t groups, bool enableHf32, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnConv3dGeluCustom(void* workspace, uint64_t workspaceSize,
                                                                         aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_CONV3D_GELU_CUSTOM_H
