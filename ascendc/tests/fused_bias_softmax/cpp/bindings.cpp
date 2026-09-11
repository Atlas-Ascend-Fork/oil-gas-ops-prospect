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
 * FusedBiasSoftmaxOg / FusedSoftmaxGradOg 的 torch_npu pybind 绑定（fused_bias_softmax_npu）。
 *
 * 直接链接 opbuild autogen 的 aclnn 两段式入口，语义（均为原地写回）：
 *   forward:  y = softmax(x + bias1(广播/全尺寸, 可选) + bias2(可选), -1)，y 别名 x
 *   backward: g = go @ V^T; probs ← probs ⊙ (g − rowsum(probs⊙g))，out 别名 probs
 */

#include <cstdint>
#include <optional>
#include <vector>

#include <torch/extension.h>
#include <ATen/ATen.h>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" {
aclnnStatus aclnnFusedBiasSoftmaxOgGetWorkspaceSize(
    const aclTensor *x, const aclTensor *bias1, const aclTensor *bias2, const aclTensor *y,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnFusedBiasSoftmaxOg(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

aclnnStatus aclnnFusedSoftmaxGradOgGetWorkspaceSize(
    const aclTensor *probs, const aclTensor *gradOutput, const aclTensor *values, const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnFusedSoftmaxGradOg(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);
}

namespace {

aclDataType ToAclDtype(at::ScalarType t)
{
    switch (t) {
        case at::kFloat:
            return ACL_FLOAT;
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        default:
            TORCH_CHECK(false, "fused_bias_softmax: unsupported dtype, need fp32/fp16/bf16");
    }
    return ACL_FLOAT;
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

aclTensor *CreateNdTensor(const std::vector<int64_t> &shape, at::ScalarType dtype, void *deviceData)
{
    // storage 描述符用 1-D 全量 numel（与稳定版 openfold binding 一致；4-D storage
    // 描述符在执行器缓存复用时存在输出指针未重绑定的行为差异）
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    int64_t numel = 1;
    for (auto d : shape) {
        numel *= d;
    }
    std::vector<int64_t> storageDims = {numel};
    return aclCreateTensor(shape.data(), shape.size(), ToAclDtype(dtype), strides.data(), 0, ACL_FORMAT_ND,
        storageDims.data(), 1, deviceData);
}

} // namespace

at::Tensor fused_bias_softmax_forward(const at::Tensor &x, std::optional<at::Tensor> bias1,
    std::optional<at::Tensor> bias2)
{
    TORCH_CHECK(x.is_contiguous(), "fused_bias_softmax: x must be contiguous");
    if (bias1.has_value()) {
        TORCH_CHECK(bias1->is_contiguous(), "fused_bias_softmax: bias1 must be contiguous");
    }
    if (bias2.has_value()) {
        TORCH_CHECK(bias2->is_contiguous(), "fused_bias_softmax: bias2 must be contiguous");
    }

    // y 别名 x（原地语义）：aclnn 输出指向同一块 device 内存
    at::Tensor y = x;
    auto xT = CreateNdTensor(ShapeOf(x), x.scalar_type(), x.data_ptr());
    aclTensor *b1T = bias1.has_value() ? CreateNdTensor(ShapeOf(*bias1), bias1->scalar_type(), bias1->data_ptr())
                                       : nullptr;
    aclTensor *b2T = bias2.has_value() ? CreateNdTensor(ShapeOf(*bias2), bias2->scalar_type(), bias2->data_ptr())
                                       : nullptr;
    aclTensor *yT = CreateNdTensor(ShapeOf(y), y.scalar_type(), y.data_ptr());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnFusedBiasSoftmaxOgGetWorkspaceSize(xT, b1T, b2T, yT, &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnFusedBiasSoftmaxOgGetWorkspaceSize failed, ret=", static_cast<int>(ret));

    void *workspace = nullptr;
    at::Tensor workspaceTensor;
    if (workspaceSize > 0) {
        workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)}, x.options().dtype(at::kByte));
        workspace = workspaceTensor.data_ptr();
    }
    ret = aclnnFusedBiasSoftmaxOg(workspace, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnFusedBiasSoftmaxOg failed, ret=", static_cast<int>(ret));

    Destroy(xT);
    Destroy(b1T);
    Destroy(b2T);
    Destroy(yT);
    return y;
}

at::Tensor fused_softmax_grad_backward(const at::Tensor &probs, const at::Tensor &gradOutput,
    const at::Tensor &values)
{
    TORCH_CHECK(probs.is_contiguous() && gradOutput.is_contiguous() && values.is_contiguous(),
        "fused_softmax_grad: probs/gradOutput/values must be contiguous");

    // out 别名 probs（原地覆写 probs → dL/dlogits）
    // 输入/输出复用同一个 aclTensor 对象（与已验证稳定的 openfold binding 一致）
    at::Tensor out = probs;
    auto probsT = CreateNdTensor(ShapeOf(probs), probs.scalar_type(), probs.data_ptr());
    auto goT = CreateNdTensor(ShapeOf(gradOutput), gradOutput.scalar_type(), gradOutput.data_ptr());
    auto vT = CreateNdTensor(ShapeOf(values), values.scalar_type(), values.data_ptr());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnFusedSoftmaxGradOgGetWorkspaceSize(probsT, goT, vT, probsT, &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnFusedSoftmaxGradOgGetWorkspaceSize failed, ret=", static_cast<int>(ret));

    void *workspace = nullptr;
    at::Tensor workspaceTensor;
    if (workspaceSize > 0) {
        workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)}, probs.options().dtype(at::kByte));
        workspace = workspaceTensor.data_ptr();
    }
    ret = aclnnFusedSoftmaxGradOg(workspace, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnFusedSoftmaxGradOg failed, ret=", static_cast<int>(ret));

    Destroy(probsT);
    Destroy(goT);
    Destroy(vT);
    return out;
}


PYBIND11_MODULE(fused_bias_softmax_npu, m)
{
    m.def("fused_bias_softmax_forward", &fused_bias_softmax_forward,
        "y = softmax(x + bias1 + bias2, -1), in-place; args: x, bias1|None, bias2|None",
        py::arg("x"), py::arg("bias1") = std::nullopt, py::arg("bias2") = std::nullopt);
    m.def("fused_softmax_grad_backward", &fused_softmax_grad_backward,
        "g = go @ V^T; probs <- probs*(g - rowsum(probs*g)), in-place",
        py::arg("probs"), py::arg("grad_output"), py::arg("values"));
}
