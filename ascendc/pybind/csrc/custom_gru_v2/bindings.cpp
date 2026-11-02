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

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "aclnn_custom_gruv2.h"

namespace custom_gru {

static aclTensor* make_acl_tensor(const at::Tensor& t)
{
    auto c = t.contiguous();
    auto ndim = (uint64_t)c.dim();
    std::vector<int64_t> shape(c.sizes().begin(), c.sizes().end());
    std::vector<int64_t> strides(c.strides().begin(), c.strides().end());
    return aclCreateTensor(shape.data(), ndim, ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, nullptr, 0, c.data_ptr());
}

std::tuple<at::Tensor, at::Tensor> custom_gru_v2(const at::Tensor& x, const at::Tensor& weight_ih,
                                                 const at::Tensor& weight_hh, const at::Tensor& bias_ih,
                                                 const at::Tensor& bias_hh, int64_t hidden, bool bidir)
{
    int64_t B = x.size(0), T = x.size(1);
    int64_t num_dirs = bidir ? 2 : 1;
    auto opts = x.options();
    auto y = at::empty({num_dirs, B, T, hidden}, opts);
    auto output_h = at::empty({num_dirs, B, hidden}, opts);

    aclTensor* x_acl = make_acl_tensor(x);
    aclTensor* wi_acl = make_acl_tensor(weight_ih);
    aclTensor* wh_acl = make_acl_tensor(weight_hh);
    bool has_bias = bias_ih.defined() && bias_ih.numel() > 0 && bias_hh.defined() && bias_hh.numel() > 0;
    aclTensor* bi_acl = has_bias ? make_acl_tensor(bias_ih) : nullptr;
    aclTensor* bh_acl = has_bias ? make_acl_tensor(bias_hh) : nullptr;
    aclTensor* y_acl = make_acl_tensor(y);
    aclTensor* oh_acl = make_acl_tensor(output_h);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCustomGRUV2GetWorkspaceSize(x_acl, wi_acl, wh_acl, bi_acl, bh_acl, bidir, y_acl, oh_acl, &ws_size,
                                                &executor);
    TORCH_CHECK(ret == 0, "aclnnCustomGRUV2GetWorkspaceSize failed: ", ret);

    auto ws_tensor = (ws_size > 0) ? at::empty({(int64_t)((ws_size + 7) / 8 * 8)}, opts.dtype(at::kByte)) :
                                     at::empty({0}, opts.dtype(at::kByte));
    void* workspace = ws_size > 0 ? ws_tensor.data_ptr() : nullptr;
    auto stream = c10_npu::getCurrentNPUStream().stream();
    ret = aclnnCustomGRUV2(workspace, ws_size, executor, stream);
    aclrtSynchronizeStream(stream);
    TORCH_CHECK(ret == 0, "aclnnCustomGRUV2 failed: ", ret);

    aclDestroyTensor(x_acl);
    aclDestroyTensor(wi_acl);
    aclDestroyTensor(wh_acl);
    if (bi_acl)
        aclDestroyTensor(bi_acl);
    if (bh_acl)
        aclDestroyTensor(bh_acl);
    aclDestroyTensor(y_acl);
    aclDestroyTensor(oh_acl);
    return {y, output_h};
}

} // namespace custom_gru

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("custom_gru_v2", &custom_gru::custom_gru_v2); }
