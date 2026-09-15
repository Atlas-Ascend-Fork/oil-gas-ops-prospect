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
constexpr size_t kInputCount = 6;
constexpr size_t kOutputCount = 4; // gur / gui / gvr / gvi 打包在同一块缓冲里

int deviceId = kDeviceId;
aclrtStream stream = nullptr;
bool gIsDevice = false;
void* hostIn[kInputCount] = {};
void* devIn[kInputCount] = {};
void* hostPacked = nullptr;
void* devPacked = nullptr;
aclTensor* inT[kInputCount] = {};
aclTensor* outT[kOutputCount] = {};
size_t oneBytes = 0;
std::vector<int64_t> gShape{8, 64};

const char* kInFiles[kInputCount] = {"../input/input_gwr.bin", "../input/input_gwi.bin", "../input/input_ur.bin",
                                     "../input/input_ui.bin",  "../input/input_vr.bin",  "../input/input_vi.bin"};
const char* kOutFiles[kOutputCount] = {"../output/output_gur.bin", "../output/output_gui.bin",
                                       "../output/output_gvr.bin", "../output/output_gvi.bin"};

void ReleaseTensors()
{
    DestroyTensors(inT, kInputCount);
    DestroyTensors(outT, kOutputCount);
    ReleaseStBuffers(hostIn, devIn, kInputCount, &hostPacked, &devPacked, gIsDevice);
}

bool ProcessInput()
{
    if (!PrepareStInputs(kInFiles, kInputCount, gShape, &oneBytes, gIsDevice, hostIn, devIn, inT, kOutputCount,
                         &hostPacked, &devPacked)) {
        return false;
    }
    const int64_t packedNumel = ShapeNumel(gShape) * static_cast<int64_t>(kOutputCount);
    outT[0] = CreateNdTensor(gShape, ACL_FLOAT, devPacked, packedNumel);
    for (size_t i = 1; i < kOutputCount; ++i) {
        outT[i] = CreateNdTensor(gShape, ACL_FLOAT, devPacked);
    }
    INFO_LOG("Set input success.");
    return outT[0] != nullptr;
}

bool ExecuteOp()
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnComplexMulGradGetWorkspaceSize(inT[0], inT[1], inT[2], inT[3], inT[4], inT[5], outT[0], outT[1],
                                                   outT[2], outT[3], &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclnnComplexMulGradGetWorkspaceSize failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("GetWorkspaceSize success, workspace size %lu", static_cast<unsigned long>(workspaceSize));
    return RunAclnnOp(aclnnComplexMulGrad, workspaceSize, executor, stream, "aclnnComplexMulGrad");
}

bool GetResult()
{
    if (!DeviceToHost(hostPacked, devPacked, oneBytes * kOutputCount, gIsDevice)) {
        return false;
    }
    auto* packed = static_cast<char*>(hostPacked);
    bool ok = true;
    for (size_t i = 0; i < kOutputCount; ++i) {
        ok = WriteFile(kOutFiles[i], packed + i * oneBytes, oneBytes) && ok;
    }
    return ok;
}

} // namespace

int main() { return RunStMain(deviceId, &stream, &gIsDevice, {ProcessInput, ExecuteOp, GetResult, ReleaseTensors}); }
