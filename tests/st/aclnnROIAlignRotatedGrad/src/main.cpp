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
 * aclnnROIAlignRotatedGrad 两段式 ST：
 *   grad_output [K,C,PH,PW] + rois [K,6] -> grad_input [N,C,H,W]
 * 真值由 scripts/gen_data.py 用纯 torch 参考实现反向生成。
 */

#include <fstream>
#include <vector>

#include "common.h"
#include "og_aclnn_decls.h"

namespace {

constexpr int kDeviceId = 0;

bool ReadAttrs(const char* path, int64_t& ph, int64_t& pw, int64_t& sr, double& scale)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        ERROR_LOG("open attrs failed: %s", path);
        return false;
    }
    file >> ph >> pw >> sr >> scale;
    return static_cast<bool>(file);
}

} // namespace

int main()
{
    aclrtStream stream = nullptr;
    bool isDevice = false;
    if (!InitResource(kDeviceId, &stream, &isDevice)) {
        return 1;
    }

    std::vector<int64_t> inputShape;
    std::vector<int64_t> gradShape;
    std::vector<int64_t> roisShape;
    int64_t ph = 0;
    int64_t pw = 0;
    int64_t sr = 0;
    double scale = 1.0;
    if (!ReadShapeFile("../input/shape.txt", inputShape) || !ReadShapeFile("../input/grad_shape.txt", gradShape) ||
        !ReadShapeFile("../input/rois_shape.txt", roisShape) || !ReadAttrs("../input/attrs.txt", ph, pw, sr, scale)) {
        DestroyResource(kDeviceId, stream);
        return 1;
    }
    if (inputShape.size() != 4 || gradShape.size() != 4 || roisShape.size() != 2 || roisShape[1] != 6) {
        ERROR_LOG("invalid shape ranks: input=%zu grad=%zu rois=%zu", inputShape.size(), gradShape.size(),
                  roisShape.size());
        DestroyResource(kDeviceId, stream);
        return 1;
    }

    const size_t gradBytes = TensorBytes(gradShape, ACL_FLOAT);
    const size_t roisBytes = TensorBytes(roisShape, ACL_FLOAT);
    const size_t outBytes = TensorBytes(inputShape, ACL_FLOAT);

    void* gradHost = nullptr;
    void* gradDev = nullptr;
    void* roisHost = nullptr;
    void* roisDev = nullptr;
    void* outHost = nullptr;
    void* outDev = nullptr;
    aclTensor* gradT = nullptr;
    aclTensor* roisT = nullptr;
    aclTensor* outT = nullptr;
    void* workspace = nullptr;

    auto release = [&]() {
        DestroyTensors(&gradT, 1);
        DestroyTensors(&roisT, 1);
        DestroyTensors(&outT, 1);
        FreeHostDevice(gradHost, gradDev, isDevice);
        FreeHostDevice(roisHost, roisDev, isDevice);
        FreeHostDevice(outHost, outDev, isDevice);
        if (workspace != nullptr) {
            (void)aclrtFree(workspace);
            workspace = nullptr;
        }
        DestroyResource(kDeviceId, stream);
    };

    if (!MallocHostDevice(gradBytes, isDevice, &gradHost, &gradDev) ||
        !MallocHostDevice(roisBytes, isDevice, &roisHost, &roisDev) ||
        !MallocHostDevice(outBytes, isDevice, &outHost, &outDev)) {
        release();
        return 1;
    }
    if (!ReadFile("../input/grad_output.bin", gradHost, gradBytes) ||
        !ReadFile("../input/rois.bin", roisHost, roisBytes)) {
        release();
        return 1;
    }
    if (!HostToDevice(gradDev, gradHost, gradBytes, isDevice) ||
        !HostToDevice(roisDev, roisHost, roisBytes, isDevice)) {
        release();
        return 1;
    }

    gradT = CreateNdTensor(gradShape, ACL_FLOAT, gradDev);
    roisT = CreateNdTensor(roisShape, ACL_FLOAT, roisDev);
    outT = CreateNdTensor(inputShape, ACL_FLOAT, outDev);
    if (gradT == nullptr || roisT == nullptr || outT == nullptr) {
        ERROR_LOG("create tensor failed");
        release();
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnROIAlignRotatedGradGetWorkspaceSize(gradT, roisT, scale, ph, pw, sr, inputShape[0], inputShape[1],
                                                        inputShape[2], inputShape[3], outT, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclnnROIAlignRotatedGradGetWorkspaceSize failed: %d", static_cast<int>(ret));
        release();
        return 1;
    }
    if (workspaceSize > 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        ERROR_LOG("allocate workspace failed");
        release();
        return 1;
    }
    if (!RunAclnnOp(aclnnROIAlignRotatedGrad, workspaceSize, executor, stream, "aclnnROIAlignRotatedGrad")) {
        release();
        return 1;
    }
    if (!DeviceToHost(outHost, outDev, outBytes, isDevice) || !WriteFile("../output/output.bin", outHost, outBytes)) {
        release();
        return 1;
    }

    release();
    INFO_LOG("ROIAlignRotatedGrad ST done: out shape=[%ld,%ld,%ld,%ld]", static_cast<long>(inputShape[0]),
             static_cast<long>(inputShape[1]), static_cast<long>(inputShape[2]), static_cast<long>(inputShape[3]));
    return 0;
}
