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
 * pybind 绑定共用的 aclTensor 辅助函数（消除各 bindings.cpp 间的重复代码）。
 */

#ifndef OG_PYBIND_ACL_TENSOR_UTILS_H
#define OG_PYBIND_ACL_TENSOR_UTILS_H

#include <cstdint>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

namespace OgPybind {

inline std::vector<int64_t> ShapeOf(const at::Tensor& t)
{
    return std::vector<int64_t>(t.sizes().begin(), t.sizes().end());
}

inline void Destroy(aclTensor* t)
{
    if (t != nullptr) {
        aclDestroyTensor(t);
    }
}

inline void DestroyExecutor(aclOpExecutor* executor)
{
    if (executor != nullptr) {
        aclDestroyAclOpExecutor(executor);
    }
}

/** 命中 aclnn 缓存后按 GetWorkspaceSize 参数顺序重绑 GM，否则会写到上一跳缓冲。 */
inline void MakeExecutorRepeatable(aclOpExecutor* executor)
{
    if (executor != nullptr) {
        aclSetAclOpExecutorRepeatable(executor);
    }
}

inline void BindTensorAddr(aclOpExecutor* executor, size_t index, aclTensor* tensor, void* addr)
{
    if (executor == nullptr || tensor == nullptr || addr == nullptr) {
        return;
    }
    aclSetRawTensorAddr(tensor, addr);
    auto ret = aclSetTensorAddr(executor, index, tensor, addr);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclSetTensorAddr failed, index=", index, " ret=", static_cast<int>(ret));
}

inline void BindInputAddr(aclOpExecutor* executor, size_t index, aclTensor* tensor, void* addr)
{
    if (executor == nullptr || tensor == nullptr || addr == nullptr) {
        return;
    }
    aclSetRawTensorAddr(tensor, addr);
    auto ret = aclSetInputTensorAddr(executor, index, tensor, addr);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclSetInputTensorAddr failed, index=", index, " ret=", static_cast<int>(ret));
}

inline void BindOutputAddr(aclOpExecutor* executor, size_t index, aclTensor* tensor, void* addr)
{
    if (executor == nullptr || tensor == nullptr || addr == nullptr) {
        return;
    }
    aclSetRawTensorAddr(tensor, addr);
    auto ret = aclSetOutputTensorAddr(executor, index, tensor, addr);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclSetOutputTensorAddr failed, index=", index, " ret=", static_cast<int>(ret));
}

/** 工作区必须活过下一次 launch：局部 Tensor 出函数即释放，缓存 executor 仍握着旧指针。 */
inline at::Tensor& StableWorkspace(bool backward, uint64_t workspaceSize, const at::Tensor& like)
{
    thread_local at::Tensor fwd;
    thread_local at::Tensor bwd;
    at::Tensor& ws = backward ? bwd : fwd;
    if (workspaceSize == 0) {
        return ws;
    }
    const int64_t n = static_cast<int64_t>(workspaceSize);
    if (!ws.defined() || ws.numel() < n || ws.device() != like.device()) {
        ws = at::empty({n}, like.options().dtype(at::kByte));
    }
    return ws;
}

} // namespace OgPybind

#endif
