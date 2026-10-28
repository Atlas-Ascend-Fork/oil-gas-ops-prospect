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

#ifndef OP_API_CONV3D_GELU_GRAD_L0_H
#define OP_API_CONV3D_GELU_GRAD_L0_H

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* Conv3DGeluGrad(const aclTensor* z, const aclTensor* gradY, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_CONV3D_GELU_GRAD_L0_H
