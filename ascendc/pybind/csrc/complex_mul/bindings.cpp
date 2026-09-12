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

using OgPybind::Destroy;
using OgPybind::DestroyExecutor;
using OgPybind::ShapeOf;

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

aclTensor* CreateNdTensor(const std::vector<int64_t>& shape, void* deviceData)
{
    auto strides = ContiguousStrides(shape);
    int64_t numel = 1;
    for (auto d : shape) {
        numel *= d;
    }
    std::vector<int64_t> storageDims = {numel};
    // 与 fused_bias_softmax 一致：1-D storage 描述符，避免执行器缓存
    // 复用时多输出指针未重绑定（第二路 wi / 前两路 gur,gui 会 silently 错）。
    return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, storageDims.data(),
                           1, deviceData);
}

/** 只取第一路 wr=a*c-b*d。CANN 9 TBE 会丢掉同一 kernel 里的第二路 wi。 */
at::Tensor LaunchComplexMulFirst(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, const at::Tensor& d)
{
    at::Tensor wr = at::empty_like(a);
    at::Tensor wiDiscard = at::empty_like(a);
    auto shape = ShapeOf(a);

    aclTensor* aT = CreateNdTensor(shape, a.data_ptr());
    aclTensor* bT = CreateNdTensor(shape, b.data_ptr());
    aclTensor* cT = CreateNdTensor(shape, c.data_ptr());
    aclTensor* dT = CreateNdTensor(shape, d.data_ptr());
    aclTensor* wrT = CreateNdTensor(shape, wr.data_ptr());
    aclTensor* wiT = CreateNdTensor(shape, wiDiscard.data_ptr());

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnComplexMulGetWorkspaceSize(aT, bT, cT, dT, wrT, wiT, &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnComplexMulGetWorkspaceSize failed, ret=", static_cast<int>(ret));

    void* workspace = nullptr;
    at::Tensor workspaceTensor;
    if (workspaceSize > 0) {
        workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)}, a.options().dtype(at::kByte));
        workspace = workspaceTensor.data_ptr();
    }
    ret = aclnnComplexMul(workspace, workspaceSize, executor, stream);
    DestroyExecutor(executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnComplexMul failed, ret=", static_cast<int>(ret));

    Destroy(aT);
    Destroy(bT);
    Destroy(cT);
    Destroy(dT);
    Destroy(wrT);
    Destroy(wiT);
    return wr;
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

    // CANN 9 TBE 会裁掉同一 kernel 的其中一路，两次 launch 又会命中错输入缓存。
    // 前向两路都在 NPU 上按公式拼；反向仍走 ComplexMul 第一路（已对拍通过）。
    at::Tensor wr = at::sub(at::mul(ur, vr), at::mul(ui, vi));
    at::Tensor wi = at::add(at::mul(ur, vi), at::mul(ui, vr));
    return {wr, wi};
}

std::vector<at::Tensor> complex_mul_backward(const at::Tensor& gwr, const at::Tensor& gwi, const at::Tensor& ur,
                                             const at::Tensor& ui, const at::Tensor& vr, const at::Tensor& vi)
{
    TORCH_CHECK(gwr.is_contiguous() && gwi.is_contiguous() && ur.is_contiguous() && ui.is_contiguous() &&
                    vr.is_contiguous() && vi.is_contiguous(),
                "complex_mul_grad inputs must be contiguous");
    TORCH_CHECK(gwr.scalar_type() == at::kFloat, "complex_mul_grad requires float32");

    // 复用 ComplexMul 第一路 Sub：TBE 会丢掉 Grad kernel 的前几路写回。
    at::Tensor gur = LaunchComplexMulFirst(gwr, gwi, vr, vi.neg());
    at::Tensor gui = LaunchComplexMulFirst(gwi, gwr, vr, vi);
    at::Tensor gvr = LaunchComplexMulFirst(gwr, gwi, ur, ui.neg());
    at::Tensor gvi = LaunchComplexMulFirst(gwi, gwr, ur, ui);
    return {gur, gui, gvr, gvi};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("complex_mul_forward", &complex_mul_forward, "ComplexMul forward");
    m.def("complex_mul_backward", &complex_mul_backward, "ComplexMulGrad backward");
}
