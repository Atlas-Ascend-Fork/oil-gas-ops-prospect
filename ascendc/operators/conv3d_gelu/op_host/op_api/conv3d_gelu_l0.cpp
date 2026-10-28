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
 * Conv3DGelu L0 实现。
 *
 * 参考官方 conv3d_v2 的 Conv3dv2WithFlag：先 INFER_SHAPE 填好输出 shape，再
 * ADD_TO_LAUNCHER_LIST_AICORE 把 kernel 加入当前 executor 的 launch 列表。
 * 输出张量沿用输入 x 的私有 storage 格式（NDC1HWC0）与原始格式（NCDHW）。
 */

#include "conv3d_gelu_l0.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Conv3DGelu);

const aclTensor* Conv3DGelu(const aclTensor* x, const aclTensor* filter, const aclTensor* bias,
                            const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations,
                            int64_t groups, bool enableHf32, aclOpExecutor* executor)
{
    L0_DFX(Conv3DGelu, x, filter, bias, strides, pads, dilations, groups, enableHf32);

    // 输出与输入同私有格式（NDC1HWC0 storage / NCDHW origin），shape 由 InferShape 计算。
    auto out = executor->AllocTensor(x->GetDataType(), x->GetStorageFormat(), x->GetOriginalFormat());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Conv3DGelu alloc output tensor failed.");
        return nullptr;
    }

    auto inferRet = INFER_SHAPE(Conv3DGelu, OP_INPUT(x, filter, bias), OP_OUTPUT(out),
                                OP_ATTR(strides, pads, dilations, groups, enableHf32));
    if (inferRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "Conv3DGelu InferShape failed.");
        return nullptr;
    }

    auto launchRet = ADD_TO_LAUNCHER_LIST_AICORE(Conv3DGelu, OP_INPUT(x, filter, bias), OP_OUTPUT(out),
                                                 OP_ATTR(strides, pads, dilations, groups, enableHf32));
    if (launchRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Conv3DGelu launch kernel failed.");
        return nullptr;
    }
    return out;
}

const aclTensor* Conv3DGeluInto(const aclTensor* x, const aclTensor* filter, const aclTensor* bias,
                                const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations,
                                int64_t groups, bool enableHf32, aclTensor* out, aclOpExecutor* executor)
{
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Conv3DGeluInto out is nullptr.");
        return nullptr;
    }

    auto inferRet = INFER_SHAPE(Conv3DGelu, OP_INPUT(x, filter, bias), OP_OUTPUT(out),
                                OP_ATTR(strides, pads, dilations, groups, enableHf32));
    if (inferRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "Conv3DGeluInto InferShape failed.");
        return nullptr;
    }

    auto launchRet = ADD_TO_LAUNCHER_LIST_AICORE(Conv3DGelu, OP_INPUT(x, filter, bias), OP_OUTPUT(out),
                                                 OP_ATTR(strides, pads, dilations, groups, enableHf32));
    if (launchRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Conv3DGeluInto launch kernel failed.");
        return nullptr;
    }
    return out;
}

} // namespace l0op
