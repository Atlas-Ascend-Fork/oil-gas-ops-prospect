/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include <torch/extension.h>
#include <ATen/ATen.h>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" {
aclnnStatus aclnnComplexMulOgGetWorkspaceSize(
    const aclTensor *ur, const aclTensor *ui, const aclTensor *vr, const aclTensor *vi,
    const aclTensor *wrOut, const aclTensor *wiOut, uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnComplexMulOg(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnComplexMulGradOgGetWorkspaceSize(
    const aclTensor *gwr, const aclTensor *gwi, const aclTensor *ur, const aclTensor *ui, const aclTensor *vr,
    const aclTensor *vi, const aclTensor *gurOut, const aclTensor *guiOut, const aclTensor *gvrOut,
    const aclTensor *gviOut, uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnComplexMulGradOg(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
}

namespace {

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

aclTensor *CreateNdTensor(const std::vector<int64_t> &shape, void *deviceData)
{
    auto strides = ContiguousStrides(shape);
    return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), deviceData);
}

std::vector<int64_t> ShapeOf(const at::Tensor &t)
{
    return std::vector<int64_t>(t.sizes().begin(), t.sizes().end());
}

void Destroy(aclTensor *t)
{
    if (t != nullptr) {
        aclDestroyTensor(t);
    }
}

} // namespace

std::vector<at::Tensor> complex_mul_forward(const at::Tensor &ur, const at::Tensor &ui, const at::Tensor &vr,
                                            const at::Tensor &vi)
{
    TORCH_CHECK(ur.is_contiguous() && ui.is_contiguous() && vr.is_contiguous() && vi.is_contiguous(),
                "complex_mul inputs must be contiguous");
    TORCH_CHECK(ur.scalar_type() == at::kFloat, "complex_mul requires float32");
    TORCH_CHECK(ur.sizes() == ui.sizes() && ur.sizes() == vr.sizes() && ur.sizes() == vi.sizes(),
                "complex_mul shape mismatch");

    at::Tensor wr = at::empty_like(ur);
    at::Tensor wi = at::empty_like(ur);
    auto shape = ShapeOf(ur);

    aclTensor *urT = CreateNdTensor(shape, ur.data_ptr());
    aclTensor *uiT = CreateNdTensor(shape, ui.data_ptr());
    aclTensor *vrT = CreateNdTensor(shape, vr.data_ptr());
    aclTensor *viT = CreateNdTensor(shape, vi.data_ptr());
    aclTensor *wrT = CreateNdTensor(shape, wr.data_ptr());
    aclTensor *wiT = CreateNdTensor(shape, wi.data_ptr());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnComplexMulOgGetWorkspaceSize(urT, uiT, vrT, viT, wrT, wiT, &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnComplexMulOgGetWorkspaceSize failed, ret=", static_cast<int>(ret));

    void *workspace = nullptr;
    at::Tensor workspaceTensor;
    if (workspaceSize > 0) {
        workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)}, ur.options().dtype(at::kByte));
        workspace = workspaceTensor.data_ptr();
    }
    ret = aclnnComplexMulOg(workspace, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnComplexMulOg failed, ret=", static_cast<int>(ret));

    Destroy(urT);
    Destroy(uiT);
    Destroy(vrT);
    Destroy(viT);
    Destroy(wrT);
    Destroy(wiT);
    return {wr, wi};
}

std::vector<at::Tensor> complex_mul_backward(const at::Tensor &gwr, const at::Tensor &gwi, const at::Tensor &ur,
                                             const at::Tensor &ui, const at::Tensor &vr, const at::Tensor &vi)
{
    TORCH_CHECK(gwr.is_contiguous() && gwi.is_contiguous() && ur.is_contiguous() && ui.is_contiguous() &&
                    vr.is_contiguous() && vi.is_contiguous(),
                "complex_mul_grad inputs must be contiguous");
    TORCH_CHECK(gwr.scalar_type() == at::kFloat, "complex_mul_grad requires float32");

    at::Tensor gur = at::empty_like(ur);
    at::Tensor gui = at::empty_like(ur);
    at::Tensor gvr = at::empty_like(ur);
    at::Tensor gvi = at::empty_like(ur);
    auto shape = ShapeOf(ur);

    aclTensor *gwrT = CreateNdTensor(shape, gwr.data_ptr());
    aclTensor *gwiT = CreateNdTensor(shape, gwi.data_ptr());
    aclTensor *urT = CreateNdTensor(shape, ur.data_ptr());
    aclTensor *uiT = CreateNdTensor(shape, ui.data_ptr());
    aclTensor *vrT = CreateNdTensor(shape, vr.data_ptr());
    aclTensor *viT = CreateNdTensor(shape, vi.data_ptr());
    aclTensor *gurT = CreateNdTensor(shape, gur.data_ptr());
    aclTensor *guiT = CreateNdTensor(shape, gui.data_ptr());
    aclTensor *gvrT = CreateNdTensor(shape, gvr.data_ptr());
    aclTensor *gviT = CreateNdTensor(shape, gvi.data_ptr());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnComplexMulGradOgGetWorkspaceSize(gwrT, gwiT, urT, uiT, vrT, viT, gurT, guiT, gvrT, gviT,
                                                     &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnComplexMulGradOgGetWorkspaceSize failed, ret=", static_cast<int>(ret));

    void *workspace = nullptr;
    at::Tensor workspaceTensor;
    if (workspaceSize > 0) {
        workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)}, ur.options().dtype(at::kByte));
        workspace = workspaceTensor.data_ptr();
    }
    ret = aclnnComplexMulGradOg(workspace, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnComplexMulGradOg failed, ret=", static_cast<int>(ret));

    Destroy(gwrT);
    Destroy(gwiT);
    Destroy(urT);
    Destroy(uiT);
    Destroy(vrT);
    Destroy(viT);
    Destroy(gurT);
    Destroy(guiT);
    Destroy(gvrT);
    Destroy(gviT);
    return {gur, gui, gvr, gvi};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("complex_mul_forward", &complex_mul_forward, "ComplexMulOg forward");
    m.def("complex_mul_backward", &complex_mul_backward, "ComplexMulGradOg backward");
}
