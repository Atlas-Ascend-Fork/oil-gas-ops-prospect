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
 * Conv3DGelu 自定义 L2 aclnn 实现，对齐官方 aclnnConvolution 的格式处理：
 * Contiguous -> TransData(私有格式) -> Conv3DGelu -> TransData(NCDHW) -> ViewCopy，
 * 全程共用一个 executor，中间 buffer 由 executor 复用。
 */

#include "aclnn_conv3d_gelu_custom.h"

#include "conv3d_gelu_l0.h"
#include "../conv3d_gelu_route.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t kPad6 = 6;
constexpr int64_t kSpatial3 = 3;

// padding 支持 3（每维对称）或 6（head/tail）；统一展开成 6 元 [d0,d1,h0,h1,w0,w1]。
static const aclIntArray* NormalizePads(const aclIntArray* padding, aclOpExecutor* executor)
{
    if (padding == nullptr) {
        int64_t zeros[kPad6] = {0, 0, 0, 0, 0, 0};
        return executor->AllocIntArray(zeros, kPad6);
    }
    if (padding->Size() == kPad6) {
        return padding;
    }
    if (padding->Size() == kSpatial3) {
        int64_t p6[kPad6] = {(*padding)[0], (*padding)[0], (*padding)[1], (*padding)[1], (*padding)[2], (*padding)[2]};
        return executor->AllocIntArray(p6, kPad6);
    }
    return nullptr;
}

static bool AllEqual(const aclIntArray* arr, int64_t expect)
{
    if (arr == nullptr) {
        return true;
    }
    for (uint64_t i = 0; i < arr->Size(); ++i) {
        if ((*arr)[i] != expect) {
            return false;
        }
    }
    return true;
}

// 1x1、stride=1、无 padding 的 NCDHW 卷积可以直接把 GELU 写进用户输出，
// 不再做 NDC1HWC0 往返，从而去掉与输出同尺寸的 workspace。
static bool UseDirectNcdhw(const aclTensor* x, const aclTensor* weight, const aclTensor* out, const aclIntArray* stride,
                           const aclIntArray* padding, const aclIntArray* dilation, int64_t groups)
{
    if (groups != 1 || x == nullptr || weight == nullptr || out == nullptr) {
        return false;
    }
    if (x->GetStorageFormat() != Format::FORMAT_NCDHW || weight->GetStorageFormat() != Format::FORMAT_NCDHW ||
        out->GetStorageFormat() != Format::FORMAT_NCDHW) {
        return false;
    }
    const op::Shape& wShape = weight->GetViewShape();
    if (wShape.GetDimNum() != 5 || wShape.GetDim(2) != 1 || wShape.GetDim(3) != 1 || wShape.GetDim(4) != 1) {
        return false;
    }
    if (!AllEqual(stride, 1) || !AllEqual(padding, 0) || !AllEqual(dilation, 1)) {
        return false;
    }
    const int64_t cin = x->GetViewShape().GetDim(1);
    const int64_t cout = wShape.GetDim(0);
    int64_t pads[6] = {0, 0, 0, 0, 0, 0};
    return Conv3dGeluUseDirect1x1(cin, cout, groups, 1, 1, 1, 1, 1, 1, 1, 1, 1, pads);
}

aclnnStatus Conv3dGeluCustomGetWorkspaceSizeImpl(const aclTensor* x, const aclTensor* weight,
                                                 const aclTensor* biasOptional, const aclIntArray* stride,
                                                 const aclIntArray* padding, const aclIntArray* dilation,
                                                 int64_t groups, bool enableHf32, aclTensor* out,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR, "weight must not be nullptr.");
    CHECK_COND(out != nullptr, ACLNN_ERR_PARAM_NULLPTR, "out must not be nullptr.");
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
    CHECK_COND(groups == 1, ACLNN_ERR_PARAM_INVALID, "Conv3DGelu only supports groups=1.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 1) 连续化（torch 张量一般已连续，这里保证 storage 与 view 一致）。
    auto xCont = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto weightCont = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_RET(weightCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* biasCont = nullptr;
    if (biasOptional != nullptr) {
        biasCont = l0op::Contiguous(biasOptional, uniqueExecutor.get());
        CHECK_RET(biasCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclIntArray* pads = NormalizePads(padding, uniqueExecutor.get());
    CHECK_RET(pads != nullptr, ACLNN_ERR_PARAM_INVALID);

    // 1×1 且通道落在直写更快的区间：cube 结果原地 GELU 写进用户 out。
    if (UseDirectNcdhw(x, weight, out, stride, padding, dilation, groups)) {
        auto written = l0op::Conv3DGeluInto(xCont, weightCont, biasCont, stride, pads, dilation, groups, enableHf32,
                                            out, uniqueExecutor.get());
        CHECK_RET(written != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 2) NCDHW -> 私有格式（与 native conv3d 一致，buffer 由 executor 管理）。
    auto xNd = l0op::TransData(xCont, Format::FORMAT_NDC1HWC0, groups, uniqueExecutor.get());
    CHECK_RET(xNd != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto weightFz = l0op::TransData(weightCont, Format::FORMAT_FRACTAL_Z_3D, groups, uniqueExecutor.get());
    CHECK_RET(weightFz != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 3) cube conv + vector gelu（私有格式输出 NDC1HWC0）。
    auto convOut = l0op::Conv3DGelu(xNd, weightFz, biasCont, stride, pads, dilation, groups, enableHf32,
                                    uniqueExecutor.get());
    CHECK_RET(convOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 4) NDC1HWC0 -> NCDHW，并拷回用户输出张量。
    auto outNcdhw = l0op::TransData(convOut, out->GetStorageFormat(), groups, uniqueExecutor.get());
    CHECK_RET(outNcdhw != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(outNcdhw, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnConv3dGeluCustomGetWorkspaceSize(const aclTensor* x, const aclTensor* weight,
                                                  const aclTensor* biasOptional, const aclIntArray* stride,
                                                  const aclIntArray* padding, const aclIntArray* dilation,
                                                  int64_t groups, aclTensor* out, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnConv3dGeluCustom, DFX_IN(x, weight, biasOptional, stride, padding, dilation, groups),
                   DFX_OUT(out));
    return Conv3dGeluCustomGetWorkspaceSizeImpl(x, weight, biasOptional, stride, padding, dilation, groups, false, out,
                                                workspaceSize, executor);
}

aclnnStatus aclnnConv3dGeluCustomV2GetWorkspaceSize(const aclTensor* x, const aclTensor* weight,
                                                    const aclTensor* biasOptional, const aclIntArray* stride,
                                                    const aclIntArray* padding, const aclIntArray* dilation,
                                                    int64_t groups, bool enableHf32, aclTensor* out,
                                                    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnConv3dGeluCustomV2,
                   DFX_IN(x, weight, biasOptional, stride, padding, dilation, groups, enableHf32), DFX_OUT(out));
    return Conv3dGeluCustomGetWorkspaceSizeImpl(x, weight, biasOptional, stride, padding, dilation, groups, enableHf32,
                                                out, workspaceSize, executor);
}

aclnnStatus aclnnConv3dGeluCustom(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConv3dGeluCustom);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "aclnnConv3dGeluCustom launch aicore failed.");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
