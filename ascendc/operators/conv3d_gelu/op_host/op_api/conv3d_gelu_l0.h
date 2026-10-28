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
 * Conv3DGelu L0 接口：在给定 executor 内向 launcher 列表追加一次 Conv3DGelu
 * cube+vector kernel 调用。输入/输出均为已转好的私有格式（x/y=NDC1HWC0,
 * filter=FRACTAL_Z_3D），格式转换由上层 L2 (aclnn) 通过 l0op::TransData 完成。
 */

#ifndef OP_API_CONV3D_GELU_L0_H
#define OP_API_CONV3D_GELU_L0_H

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* Conv3DGelu(const aclTensor* x, const aclTensor* filter, const aclTensor* bias,
                            const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations,
                            int64_t groups, bool enableHf32, aclOpExecutor* executor);

// 把 GELU 结果直接写入调用方给出的 out（NCDHW）。1x1 卷积走这条路径，避免再分配一份同尺寸私有格式缓冲。
const aclTensor* Conv3DGeluInto(const aclTensor* x, const aclTensor* filter, const aclTensor* bias,
                                const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations,
                                int64_t groups, bool enableHf32, aclTensor* out, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_CONV3D_GELU_L0_H
