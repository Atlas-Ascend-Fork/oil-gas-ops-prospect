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
 * 对齐 ops-nn addmv/examples：内联小 tensor，两段式 aclnnComplexMul。
 * wr/wi 共用一块 packed 缓冲（kernel 忽略独立 wi 指针）。
 */

#include "aclnn_example.h"
#include "og_aclnn_decls.h"

namespace {

constexpr size_t kNumel = 4;
constexpr size_t kPackedNumel = kNumel * 2;
constexpr size_t kInputCount = 4;

const std::vector<int64_t> kShape = {2, 2};
const std::vector<float> kUr = {1, 2, 3, 4};
const std::vector<float> kUi = {0, 1, -1, 0.5f};
const std::vector<float> kVr = {0.5f, 1, 1, 2};
const std::vector<float> kVi = {1, 0, 0.5f, -1};

struct Tensors {
    void* inDev[kInputCount] = {};
    aclTensor* inT[kInputCount] = {};
    void* packedDev = nullptr;
    aclTensor* wrT = nullptr;
    aclTensor* wiT = nullptr;
};

int SetupTensors(Tensors& t)
{
    const std::vector<float>* hosts[kInputCount] = {&kUr, &kUi, &kVr, &kVi};
    for (size_t i = 0; i < kInputCount; ++i) {
        auto ret = CreateAclTensor(*hosts[i], kShape, &t.inDev[i], aclDataType::ACL_FLOAT, &t.inT[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    const std::vector<float> packed(kPackedNumel, 0.0f);
    auto ret = CreateAclTensor(packed, kShape, &t.packedDev, aclDataType::ACL_FLOAT, &t.wrT, kPackedNumel);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return CreateAclTensorView(t.packedDev, kShape, aclDataType::ACL_FLOAT, &t.wiT);
}

int LaunchOp(const Tensors& t, aclrtStream stream, void** workspaceAddr)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnComplexMulGetWorkspaceSize(t.inT[0], t.inT[1], t.inT[2], t.inT[3], t.wrT, t.wiT, &workspaceSize,
                                               &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnComplexMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    ret = MallocWorkspace(workspaceSize, workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnComplexMul(*workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnComplexMul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

std::vector<float> Expected(const Tensors&)
{
    std::vector<float> expected(kPackedNumel, 0.0f);
    for (size_t i = 0; i < kNumel; ++i) {
        expected[i] = kUr[i] * kVr[i] - kUi[i] * kVi[i];
        expected[i + kNumel] = kUr[i] * kVi[i] + kUi[i] * kVr[i];
    }
    return expected;
}

void Release(Tensors& t)
{
    DestroyTensors(t.inT, kInputCount);
    DestroyTensors(&t.wrT, 1);
    DestroyTensors(&t.wiT, 1);
    FreePackedBuffers(t.inDev, kInputCount, &t.packedDev);
}

} // namespace

int main()
{
    const AclnnExampleOps<Tensors> ops{SetupTensors, LaunchOp, &Tensors::packedDev, Expected, Release};
    return RunAclnnExample(ops, kPackedNumel);
}
