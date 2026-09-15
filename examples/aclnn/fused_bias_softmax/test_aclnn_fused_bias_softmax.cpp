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
 * 两段式 aclnnFusedBiasSoftmax 小样例：无 bias 的 softmax(x, -1)，y 原地写回 x。
 */

#include <algorithm>
#include <cmath>

#include "aclnn_example.h"
#include "og_aclnn_decls.h"

namespace {

constexpr int64_t kRows = 2;
constexpr int64_t kWidth = 8;
constexpr size_t kNumel = static_cast<size_t>(kRows * kWidth);

const std::vector<int64_t> kShape = {1, 1, kRows, kWidth};

std::vector<float> MakeInput()
{
    std::vector<float> x(kNumel);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(i % 8) * 0.1f - 0.3f;
    }
    return x;
}

struct Tensors {
    std::vector<float> x = MakeInput();
    void* xDev = nullptr;
    aclTensor* xT = nullptr;
    aclTensor* yT = nullptr;
};

int SetupTensors(Tensors& t)
{
    auto ret = CreateAclTensor(t.x, kShape, &t.xDev, aclDataType::ACL_FLOAT, &t.xT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 原地写回：y 复用 x 的 device 缓冲
    return CreateAclTensorView(t.xDev, kShape, aclDataType::ACL_FLOAT, &t.yT);
}

int LaunchOp(const Tensors& t, aclrtStream stream, void** workspaceAddr)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnFusedBiasSoftmaxGetWorkspaceSize(t.xT, nullptr, nullptr, t.yT, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedBiasSoftmaxGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    ret = MallocWorkspace(workspaceSize, workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnFusedBiasSoftmax(*workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedBiasSoftmax failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

std::vector<float> Expected(const Tensors& t)
{
    const std::vector<float>& x = t.x;
    std::vector<float> expected(x.size(), 0.0f);
    for (int64_t row = 0; row < kRows; ++row) {
        const size_t base = static_cast<size_t>(row * kWidth);
        float maxv = x[base];
        for (int64_t c = 1; c < kWidth; ++c) {
            maxv = std::max(maxv, x[base + static_cast<size_t>(c)]);
        }
        float sum = 0.0f;
        for (int64_t c = 0; c < kWidth; ++c) {
            expected[base + static_cast<size_t>(c)] = std::exp(x[base + static_cast<size_t>(c)] - maxv);
            sum += expected[base + static_cast<size_t>(c)];
        }
        for (int64_t c = 0; c < kWidth; ++c) {
            expected[base + static_cast<size_t>(c)] /= sum;
        }
    }
    return expected;
}

void Release(Tensors& t)
{
    DestroyTensors(&t.xT, 1);
    DestroyTensors(&t.yT, 1);
    FreeDeviceBuffers(&t.xDev, 1);
}

} // namespace

int main()
{
    const AclnnExampleOps<Tensors> ops{SetupTensors, LaunchOp, &Tensors::xDev, Expected, Release};
    return RunAclnnExample(ops, kNumel);
}
