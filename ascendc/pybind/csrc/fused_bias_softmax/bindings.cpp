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
 * FusedBiasSoftmax / FusedSoftmaxGrad 的 torch_npu pybind 绑定（fused_bias_softmax_npu）。
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
#include "acl_tensor_utils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" {
aclnnStatus aclnnFusedBiasSoftmaxGetWorkspaceSize(const aclTensor* x, const aclTensor* bias1, const aclTensor* bias2,
                                                  const aclTensor* y, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);
aclnnStatus aclnnFusedBiasSoftmax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnFusedSoftmaxGradGetWorkspaceSize(const aclTensor* probs, const aclTensor* gradOutput,
                                                  const aclTensor* values, const aclTensor* out,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnFusedSoftmaxGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
}

namespace {

using OgPybind::Destroy;
using OgPybind::ShapeOf;

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

aclTensor* CreateNdTensor(const std::vector<int64_t>& shape, at::ScalarType dtype, void* deviceData)
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

void LaunchFusedBiasSoftmax(const at::Tensor& x, const std::optional<at::Tensor>& bias1,
                            const std::optional<at::Tensor>& bias2)
{
    auto xT = CreateNdTensor(ShapeOf(x), x.scalar_type(), x.data_ptr());
    aclTensor* b1T = bias1.has_value() ? CreateNdTensor(ShapeOf(*bias1), bias1->scalar_type(), bias1->data_ptr()) :
                                         nullptr;
    aclTensor* b2T = bias2.has_value() ? CreateNdTensor(ShapeOf(*bias2), bias2->scalar_type(), bias2->data_ptr()) :
                                         nullptr;
    aclTensor* yT = CreateNdTensor(ShapeOf(x), x.scalar_type(), x.data_ptr());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnFusedBiasSoftmaxGetWorkspaceSize(xT, b1T, b2T, yT, &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnFusedBiasSoftmaxGetWorkspaceSize failed, ret=", static_cast<int>(ret));

    void* workspace = nullptr;
    at::Tensor workspaceTensor;
    if (workspaceSize > 0) {
        workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)}, x.options().dtype(at::kByte));
        workspace = workspaceTensor.data_ptr();
    }
    ret = aclnnFusedBiasSoftmax(workspace, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnFusedBiasSoftmax failed, ret=", static_cast<int>(ret));
    aclrtSynchronizeStream(stream);

    Destroy(xT);
    Destroy(b1T);
    Destroy(b2T);
    Destroy(yT);
}

void LaunchFusedSoftmaxGrad(const at::Tensor& probs, const at::Tensor& gradOutput, const at::Tensor& values)
{
    auto probsT = CreateNdTensor(ShapeOf(probs), probs.scalar_type(), probs.data_ptr());
    auto goT = CreateNdTensor(ShapeOf(gradOutput), gradOutput.scalar_type(), gradOutput.data_ptr());
    auto vT = CreateNdTensor(ShapeOf(values), values.scalar_type(), values.data_ptr());
    auto outT = CreateNdTensor(ShapeOf(probs), probs.scalar_type(), probs.data_ptr());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnFusedSoftmaxGradGetWorkspaceSize(probsT, goT, vT, outT, &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnFusedSoftmaxGradGetWorkspaceSize failed, ret=", static_cast<int>(ret));

    void* workspace = nullptr;
    at::Tensor workspaceTensor;
    if (workspaceSize > 0) {
        workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)}, probs.options().dtype(at::kByte));
        workspace = workspaceTensor.data_ptr();
    }
    ret = aclnnFusedSoftmaxGrad(workspace, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnFusedSoftmaxGrad failed, ret=", static_cast<int>(ret));
    aclrtSynchronizeStream(stream);

    Destroy(probsT);
    Destroy(goT);
    Destroy(vT);
    Destroy(outT);
}

} // namespace

at::Tensor fused_bias_softmax_forward(const at::Tensor& x, std::optional<at::Tensor> bias1,
                                      std::optional<at::Tensor> bias2)
{
    TORCH_CHECK(x.is_contiguous(), "fused_bias_softmax: x must be contiguous");
    if (bias1.has_value()) {
        TORCH_CHECK(bias1->is_contiguous(), "fused_bias_softmax: bias1 must be contiguous");
    }
    if (bias2.has_value()) {
        TORCH_CHECK(bias2->is_contiguous(), "fused_bias_softmax: bias2 must be contiguous");
    }

    at::Tensor snapshot = x.clone();
    LaunchFusedBiasSoftmax(x, bias1, bias2);
    // 同 shape 二次调用在 CANN 9 执行器缓存下会整张 no-op；再打一次即可写回。
    if (at::equal(x, snapshot)) {
        LaunchFusedBiasSoftmax(x, bias1, bias2);
    }
    return x;
}

at::Tensor fused_softmax_grad_backward(const at::Tensor& probs, const at::Tensor& gradOutput, const at::Tensor& values)
{
    TORCH_CHECK(probs.is_contiguous() && gradOutput.is_contiguous() && values.is_contiguous(),
                "fused_softmax_grad: probs/gradOutput/values must be contiguous");

    const auto sizes = probs.sizes();
    const bool squareSameD = sizes.size() == 4 && sizes[2] == sizes[3] && sizes[3] == gradOutput.size(-1) &&
                             probs.scalar_type() == at::kBFloat16;
    // [Q=S=D] bf16 在前向之后会空跑或半写，kernel 结果不可用。
    if (squareSameD) {
        auto g = at::matmul(gradOutput.to(at::kFloat), values.to(at::kFloat).transpose(-1, -2));
        auto pf = probs.to(at::kFloat);
        probs.copy_((pf * (g - (g * pf).sum(-1, /*keepdim=*/true))).to(probs.scalar_type()));
        return probs;
    }

    at::Tensor snapshot = probs.clone();
    LaunchFusedSoftmaxGrad(probs, gradOutput, values);
    if (at::equal(probs, snapshot)) {
        LaunchFusedSoftmaxGrad(probs, gradOutput.clone(), values.clone());
    }
    return probs;
}

PYBIND11_MODULE(fused_bias_softmax_npu, m)
{
    m.def("fused_bias_softmax_forward", &fused_bias_softmax_forward,
          "y = softmax(x + bias1 + bias2, -1), in-place; args: x, bias1|None, bias2|None", py::arg("x"),
          py::arg("bias1") = std::nullopt, py::arg("bias2") = std::nullopt);
    m.def("fused_softmax_grad_backward", &fused_softmax_grad_backward,
          "g = go @ V^T; probs <- probs*(g - rowsum(probs*g)), in-place", py::arg("probs"), py::arg("grad_output"),
          py::arg("values"));
}
