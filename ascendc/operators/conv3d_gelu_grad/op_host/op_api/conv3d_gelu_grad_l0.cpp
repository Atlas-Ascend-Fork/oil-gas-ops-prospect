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

#include "conv3d_gelu_grad_l0.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Conv3DGeluGrad);

const aclTensor* Conv3DGeluGrad(const aclTensor* z, const aclTensor* gradY, aclOpExecutor* executor)
{
    L0_DFX(Conv3DGeluGrad, z, gradY);
    auto dz = executor->AllocTensor(z->GetDataType(), z->GetStorageFormat(), z->GetOriginalFormat());
    if (dz == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Conv3DGeluGrad alloc output failed.");
        return nullptr;
    }
    auto inferRet = INFER_SHAPE(Conv3DGeluGrad, OP_INPUT(z, gradY), OP_OUTPUT(dz));
    if (inferRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "Conv3DGeluGrad InferShape failed.");
        return nullptr;
    }
    auto launchRet = ADD_TO_LAUNCHER_LIST_AICORE(Conv3DGeluGrad, OP_INPUT(z, gradY), OP_OUTPUT(dz));
    if (launchRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Conv3DGeluGrad launch kernel failed.");
        return nullptr;
    }
    return dz;
}

} // namespace l0op
