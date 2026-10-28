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

#include "aclnn_conv3d_gelu_grad.h"

#include "conv3d_gelu_grad_l0.h"

#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static aclnnStatus CopyOut(const aclTensor* src, aclTensor* dst, aclOpExecutor* executor)
{
    if (src == nullptr || dst == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto copied = l0op::ViewCopy(src, dst, executor);
    return copied == nullptr ? ACLNN_ERR_INNER_NULLPTR : ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnConv3dGeluGradGetWorkspaceSize(const aclTensor* z, const aclTensor* gradY, aclTensor* gradZ,
                                                uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnConv3dGeluGrad, DFX_IN(z, gradY), DFX_OUT(gradZ));
    CHECK_COND(z != nullptr, ACLNN_ERR_PARAM_NULLPTR, "z must not be nullptr.");
    CHECK_COND(gradY != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gradY must not be nullptr.");
    CHECK_COND(gradZ != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gradZ must not be nullptr.");
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto zCont = l0op::Contiguous(z, uniqueExecutor.get());
    CHECK_RET(zCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dy = l0op::Contiguous(gradY, uniqueExecutor.get());
    CHECK_RET(dy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dz = l0op::Conv3DGeluGrad(zCont, dy, uniqueExecutor.get());
    CHECK_RET(CopyOut(dz, gradZ, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConv3dGeluGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConv3dGeluGrad);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnConv3dGeluGrad launch failed.");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
