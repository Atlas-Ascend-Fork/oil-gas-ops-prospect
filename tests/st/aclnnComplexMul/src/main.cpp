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

#include <vector>

#include "common.h"
#include "og_aclnn_decls.h"

namespace {

constexpr int kDeviceId = 0;
constexpr size_t kInputCount = 4;
constexpr int64_t kPackedFactor = 2; // wr / wi 打包在同一块缓冲里

int deviceId = kDeviceId;
aclrtStream stream = nullptr;
bool gIsDevice = false;
void* hostIn[kInputCount] = {};
void* devIn[kInputCount] = {};
void* hostPacked = nullptr;
void* devPacked = nullptr;
aclTensor* inT[kInputCount] = {};
aclTensor* wrT = nullptr;
aclTensor* wiT = nullptr;
size_t oneBytes = 0;
std::vector<int64_t> gShape{8, 64};

const char* kInFiles[kInputCount] = {"../input/input_ur.bin", "../input/input_ui.bin", "../input/input_vr.bin",
                                     "../input/input_vi.bin"};

void ReleaseTensors()
{
    DestroyTensors(inT, kInputCount);
    DestroyTensors(&wrT, 1);
    DestroyTensors(&wiT, 1);
    ReleaseStBuffers(hostIn, devIn, kInputCount, &hostPacked, &devPacked, gIsDevice);
}

bool ProcessInput()
{
    if (!PrepareStInputs(kInFiles, kInputCount, gShape, &oneBytes, gIsDevice, hostIn, devIn, inT, kPackedFactor,
                         &hostPacked, &devPacked)) {
        return false;
    }
    wrT = CreateNdTensor(gShape, ACL_FLOAT, devPacked, ShapeNumel(gShape) * kPackedFactor);
    wiT = CreateNdTensor(gShape, ACL_FLOAT, devPacked);
    INFO_LOG("Set input success.");
    return wrT != nullptr && wiT != nullptr;
}

bool ExecuteOp()
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnComplexMulGetWorkspaceSize(inT[0], inT[1], inT[2], inT[3], wrT, wiT, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclnnComplexMulGetWorkspaceSize failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("GetWorkspaceSize success, workspace size %lu", static_cast<unsigned long>(workspaceSize));
    return RunAclnnOp(aclnnComplexMul, workspaceSize, executor, stream, "aclnnComplexMul");
}

bool GetResult()
{
    if (!DeviceToHost(hostPacked, devPacked, oneBytes * kPackedFactor, gIsDevice)) {
        return false;
    }
    auto* packed = static_cast<char*>(hostPacked);
    return WriteFile("../output/output_wr.bin", packed, oneBytes) &&
           WriteFile("../output/output_wi.bin", packed + oneBytes, oneBytes);
}

} // namespace

int main() { return RunStMain(deviceId, &stream, &gIsDevice, {ProcessInput, ExecuteOp, GetResult, ReleaseTensors}); }
