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
 * 两段式 aclnnComplexMulGrad 小样例。
 * gur/gui/gvr/gvi 共用一块 packed 缓冲（kernel 忽略后三个独立指针）。
 */

#include "aclnn_example.h"
#include "og_aclnn_decls.h"

namespace {

constexpr size_t kNumel = 4;
constexpr size_t kInputCount = 6;
constexpr size_t kOutputCount = 4;
constexpr size_t kPackedNumel = kNumel * kOutputCount;

const std::vector<int64_t> kShape = {2, 2};
const std::vector<float> kGwr = {1, 0, 0.5f, -1};
const std::vector<float> kGwi = {0, 1, -0.5f, 0.25f};
const std::vector<float> kUr = {1, 2, 3, 4};
const std::vector<float> kUi = {0, 1, -1, 0.5f};
const std::vector<float> kVr = {0.5f, 1, 1, 2};
const std::vector<float> kVi = {1, 0, 0.5f, -1};

struct Tensors {
    void* inDev[kInputCount] = {};
    aclTensor* inT[kInputCount] = {};
    void* packedDev = nullptr;
    aclTensor* outT[kOutputCount] = {};
};

int SetupTensors(Tensors& t)
{
    const std::vector<float>* hosts[kInputCount] = {&kGwr, &kGwi, &kUr, &kUi, &kVr, &kVi};
    for (size_t i = 0; i < kInputCount; ++i) {
        auto ret = CreateAclTensor(*hosts[i], kShape, &t.inDev[i], aclDataType::ACL_FLOAT, &t.inT[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    const std::vector<float> packed(kPackedNumel, 0.0f);
    auto ret = CreateAclTensor(packed, kShape, &t.packedDev, aclDataType::ACL_FLOAT, &t.outT[0], kPackedNumel);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    for (size_t i = 1; i < kOutputCount; ++i) {
        ret = CreateAclTensorView(t.packedDev, kShape, aclDataType::ACL_FLOAT, &t.outT[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    return ACL_SUCCESS;
}

int LaunchOp(const Tensors& t, aclrtStream stream, void** workspaceAddr)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnComplexMulGradGetWorkspaceSize(t.inT[0], t.inT[1], t.inT[2], t.inT[3], t.inT[4], t.inT[5],
                                                   t.outT[0], t.outT[1], t.outT[2], t.outT[3], &workspaceSize,
                                                   &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnComplexMulGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    ret = MallocWorkspace(workspaceSize, workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnComplexMulGrad(*workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnComplexMulGrad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

std::vector<float> Expected(const Tensors&)
{
    std::vector<float> expected(kPackedNumel, 0.0f);
    for (size_t i = 0; i < kNumel; ++i) {
        expected[i] = kGwr[i] * kVr[i] + kGwi[i] * kVi[i];
        expected[i + kNumel] = -kGwr[i] * kVi[i] + kGwi[i] * kVr[i];
        expected[i + kNumel * 2] = kGwr[i] * kUr[i] + kGwi[i] * kUi[i];
        expected[i + kNumel * 3] = -kGwr[i] * kUi[i] + kGwi[i] * kUr[i];
    }
    return expected;
}

void Release(Tensors& t)
{
    DestroyTensors(t.inT, kInputCount);
    DestroyTensors(t.outT, kOutputCount);
    FreePackedBuffers(t.inDev, kInputCount, &t.packedDev);
}

} // namespace

int main()
{
    const AclnnExampleOps<Tensors> ops{SetupTensors, LaunchOp, &Tensors::packedDev, Expected, Release};
    return RunAclnnExample(ops, kPackedNumel);
}
