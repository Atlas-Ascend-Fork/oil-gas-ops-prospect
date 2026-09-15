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

#include <cstdint>
#include <vector>

#include <torch/extension.h>
#include <ATen/ATen.h>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "acl_tensor_utils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" {
aclnnStatus aclnnComplexMulGetWorkspaceSize(const aclTensor* ur, const aclTensor* ui, const aclTensor* vr,
                                            const aclTensor* vi, const aclTensor* wrOut, const aclTensor* wiOut,
                                            uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnComplexMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnComplexMulGradGetWorkspaceSize(const aclTensor* gwr, const aclTensor* gwi, const aclTensor* ur,
                                                const aclTensor* ui, const aclTensor* vr, const aclTensor* vi,
                                                const aclTensor* gurOut, const aclTensor* guiOut,
                                                const aclTensor* gvrOut, const aclTensor* gviOut,
                                                uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnComplexMulGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
}

namespace {

using OgPybind::BindInputAddr;
using OgPybind::BindOutputAddr;
using OgPybind::Destroy;
using OgPybind::MakeExecutorRepeatable;
using OgPybind::ShapeOf;
using OgPybind::StableWorkspace;

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

int64_t NumelOf(const std::vector<int64_t>& shape)
{
    int64_t numel = 1;
    for (auto d : shape) {
        numel *= d;
    }
    return numel;
}

aclTensor* CreateNdTensor(const std::vector<int64_t>& shape, void* deviceData, int64_t storageNumel = -1)
{
    auto strides = ContiguousStrides(shape);
    const int64_t numel = storageNumel > 0 ? storageNumel : NumelOf(shape);
    std::vector<int64_t> storageDims = {numel};
    return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, storageDims.data(),
                           1, deviceData);
}

} // namespace

std::vector<at::Tensor> complex_mul_forward(const at::Tensor& ur, const at::Tensor& ui, const at::Tensor& vr,
                                            const at::Tensor& vi)
{
    TORCH_CHECK(ur.is_contiguous() && ui.is_contiguous() && vr.is_contiguous() && vi.is_contiguous(),
                "complex_mul inputs must be contiguous");
    TORCH_CHECK(ur.scalar_type() == at::kFloat, "complex_mul requires float32");
    TORCH_CHECK(ur.sizes() == ui.sizes() && ur.sizes() == vr.sizes() && ur.sizes() == vi.sizes(),
                "complex_mul shape mismatch");

    const auto shape = ShapeOf(ur);
    const int64_t numel = ur.numel();
    // TBE 只认第一路基址，wr/wi 必须落在同一块连续缓冲。
    // 不要继承输入的 requires_grad，否则 autograd.Function 无法给输出挂自己的 grad_fn。
    at::Tensor packed = at::empty({2, numel}, ur.options().requires_grad(false));
    at::Tensor wr = packed.select(0, 0).view(ur.sizes());
    at::Tensor wi = packed.select(0, 1).view(ur.sizes());

    aclTensor* urT = CreateNdTensor(shape, ur.data_ptr());
    aclTensor* uiT = CreateNdTensor(shape, ui.data_ptr());
    aclTensor* vrT = CreateNdTensor(shape, vr.data_ptr());
    aclTensor* viT = CreateNdTensor(shape, vi.data_ptr());
    aclTensor* wrT = CreateNdTensor(shape, packed.data_ptr(), numel * 2);
    aclTensor* wiT = CreateNdTensor(shape, packed.data_ptr());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnComplexMulGetWorkspaceSize(urT, uiT, vrT, viT, wrT, wiT, &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnComplexMulGetWorkspaceSize failed, ret=", static_cast<int>(ret));
    MakeExecutorRepeatable(executor);
    BindInputAddr(executor, 0, urT, ur.data_ptr());
    BindInputAddr(executor, 1, uiT, ui.data_ptr());
    BindInputAddr(executor, 2, vrT, vr.data_ptr());
    BindInputAddr(executor, 3, viT, vi.data_ptr());
    BindOutputAddr(executor, 0, wrT, packed.data_ptr());
    BindOutputAddr(executor, 1, wiT, packed.data_ptr());

    void* workspace = nullptr;
    at::Tensor& workspaceTensor = StableWorkspace(false, workspaceSize, ur);
    if (workspaceSize > 0) {
        workspace = workspaceTensor.data_ptr();
    }
    ret = aclnnComplexMul(workspace, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnComplexMul failed, ret=", static_cast<int>(ret));

    Destroy(urT);
    Destroy(uiT);
    Destroy(vrT);
    Destroy(viT);
    Destroy(wrT);
    Destroy(wiT);
    return {wr, wi};
}

namespace {

void LaunchComplexMulGrad(const at::Tensor& gwr, const at::Tensor& gwi, const at::Tensor& ur, const at::Tensor& ui,
                          const at::Tensor& vr, const at::Tensor& vi, at::Tensor& packed)
{
    const auto shape = ShapeOf(ur);
    const int64_t numel = ur.numel();
    aclTensor* gwrT = CreateNdTensor(shape, gwr.data_ptr());
    aclTensor* gwiT = CreateNdTensor(shape, gwi.data_ptr());
    aclTensor* urT = CreateNdTensor(shape, ur.data_ptr());
    aclTensor* uiT = CreateNdTensor(shape, ui.data_ptr());
    aclTensor* vrT = CreateNdTensor(shape, vr.data_ptr());
    aclTensor* viT = CreateNdTensor(shape, vi.data_ptr());
    aclTensor* gurT = CreateNdTensor(shape, packed.data_ptr(), numel * 4);
    aclTensor* guiT = CreateNdTensor(shape, packed.data_ptr());
    aclTensor* gvrT = CreateNdTensor(shape, packed.data_ptr());
    aclTensor* gviT = CreateNdTensor(shape, packed.data_ptr());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnComplexMulGradGetWorkspaceSize(gwrT, gwiT, urT, uiT, vrT, viT, gurT, guiT, gvrT, gviT,
                                                   &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnComplexMulGradGetWorkspaceSize failed, ret=", static_cast<int>(ret));
    MakeExecutorRepeatable(executor);
    BindInputAddr(executor, 0, gwrT, gwr.data_ptr());
    BindInputAddr(executor, 1, gwiT, gwi.data_ptr());
    BindInputAddr(executor, 2, urT, ur.data_ptr());
    BindInputAddr(executor, 3, uiT, ui.data_ptr());
    BindInputAddr(executor, 4, vrT, vr.data_ptr());
    BindInputAddr(executor, 5, viT, vi.data_ptr());
    BindOutputAddr(executor, 0, gurT, packed.data_ptr());
    BindOutputAddr(executor, 1, guiT, packed.data_ptr());
    BindOutputAddr(executor, 2, gvrT, packed.data_ptr());
    BindOutputAddr(executor, 3, gviT, packed.data_ptr());

    void* workspace = nullptr;
    at::Tensor& workspaceTensor = StableWorkspace(true, workspaceSize, ur);
    if (workspaceSize > 0) {
        workspace = workspaceTensor.data_ptr();
    }
    ret = aclnnComplexMulGrad(workspace, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnComplexMulGrad failed, ret=", static_cast<int>(ret));

    aclTensor* tensors[] = {gwrT, gwiT, urT, uiT, vrT, viT, gurT, guiT, gvrT, gviT};
    for (aclTensor* t : tensors) {
        Destroy(t);
    }
}

} // namespace

std::vector<at::Tensor> complex_mul_backward(const at::Tensor& gwr, const at::Tensor& gwi, const at::Tensor& ur,
                                             const at::Tensor& ui, const at::Tensor& vr, const at::Tensor& vi)
{
    TORCH_CHECK(gwr.is_contiguous() && gwi.is_contiguous() && ur.is_contiguous() && ui.is_contiguous() &&
                    vr.is_contiguous() && vi.is_contiguous(),
                "complex_mul_grad inputs must be contiguous");
    TORCH_CHECK(gwr.scalar_type() == at::kFloat, "complex_mul_grad requires float32");

    const int64_t numel = ur.numel();
    at::Tensor packed = at::empty({4, numel}, ur.options().requires_grad(false));
    at::Tensor gur = packed.select(0, 0).view(ur.sizes());
    at::Tensor gui = packed.select(0, 1).view(ur.sizes());
    at::Tensor gvr = packed.select(0, 2).view(ur.sizes());
    at::Tensor gvi = packed.select(0, 3).view(ur.sizes());
    LaunchComplexMulGrad(gwr, gwi, ur, ui, vr, vi, packed);
    return {gur, gui, gvr, gvi};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("complex_mul_forward", &complex_mul_forward, "ComplexMul forward");
    m.def("complex_mul_backward", &complex_mul_backward, "ComplexMulGrad backward");
}
