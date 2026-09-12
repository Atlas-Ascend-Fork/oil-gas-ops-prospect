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

} // namespace OgPybind

#endif
