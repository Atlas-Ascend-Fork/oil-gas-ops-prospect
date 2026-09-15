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
 * 对齐 ops-nn matmul/addmv/examples/test_aclnn_addmv.cpp 的两段式 aclnn 小样例脚手架。
 * 各算子 examples/test_aclnn_*.cpp 内联构造小 tensor，host 侧核对结果。
 */

#ifndef OG_ACLNN_EXAMPLE_H
#define OG_ACLNN_EXAMPLE_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

inline int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

inline std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

inline int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

inline aclTensor* MakeNdTensor(const std::vector<int64_t>& shape, aclDataType dataType, void* deviceAddr,
                               int64_t storageNumel = -1)
{
    const auto strides = ContiguousStrides(shape);
    const int64_t numel = storageNumel > 0 ? storageNumel : GetShapeSize(shape);
    std::vector<int64_t> storageDims = {numel};
    return aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                           storageDims.data(), 1, deviceAddr);
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor, int64_t storageNumel = -1)
{
    const int64_t logical = GetShapeSize(shape);
    const int64_t storage = storageNumel > 0 ? storageNumel : static_cast<int64_t>(hostData.size());
    CHECK_RET(storage >= logical, LOG_PRINT("hostData smaller than logical shape\n"); return 1);
    const size_t bytes = static_cast<size_t>(storage) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    *tensor = MakeNdTensor(shape, dataType, *deviceAddr, storage);
    return 0;
}

inline int CreateAclTensorView(void* deviceAddr, const std::vector<int64_t>& shape, aclDataType dataType,
                               aclTensor** tensor, int64_t storageNumel = -1)
{
    *tensor = MakeNdTensor(shape, dataType, deviceAddr, storageNumel);
    return *tensor == nullptr ? 1 : 0;
}

inline bool HostClose(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1e-5f,
                      float rtol = 1e-5f)
{
    if (actual.size() != expected.size()) {
        LOG_PRINT("size mismatch: actual=%zu expected=%zu\n", actual.size(), expected.size());
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const float exp = expected[i];
        const float got = actual[i];
        const float tol = atol + rtol * std::fabs(exp);
        if (std::fabs(got - exp) > tol) {
            LOG_PRINT("mismatch[%zu] expected=%f actual=%f\n", i, exp, got);
            return false;
        }
    }
    return true;
}

inline int CopyOut(void* deviceAddr, std::vector<float>& host, size_t count)
{
    host.resize(count);
    const size_t bytes = count * sizeof(float);
    auto ret = aclrtMemcpy(host.data(), bytes, deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    return 0;
}

/** workspaceSize 为 0 时不分配，*workspaceAddr 保持 nullptr。 */
inline int MallocWorkspace(uint64_t workspaceSize, void** workspaceAddr)
{
    if (workspaceSize == 0) {
        return ACL_SUCCESS;
    }
    auto ret = aclrtMalloc(workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

inline void PrintResult(const std::vector<float>& result)
{
    for (size_t i = 0; i < result.size(); ++i) {
        LOG_PRINT("result[%zu] is: %f\n", i, result[i]);
    }
}

inline void DestroyTensors(aclTensor** tensors, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (tensors[i] != nullptr) {
            (void)aclDestroyTensor(tensors[i]);
            tensors[i] = nullptr;
        }
    }
}

inline void FreeDeviceBuffers(void** buffers, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (buffers[i] != nullptr) {
            (void)aclrtFree(buffers[i]);
            buffers[i] = nullptr;
        }
    }
}

/** 释放 count 个输入 device 缓冲，外加一块打包输出缓冲。 */
inline void FreePackedBuffers(void** inputs, size_t count, void** packed)
{
    FreeDeviceBuffers(inputs, count);
    FreeDeviceBuffers(packed, 1);
}

/** 样例收尾：释放 workspace、销毁 stream、复位 device、去初始化 acl。 */
inline void Finalize(int32_t deviceId, aclrtStream stream, void* workspaceAddr)
{
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
}

/**
 * 各样例的算子相关回调。TensorsT 由样例自定义，持有 host 输入、device 缓冲与 aclTensor。
 * setup 建 tensor，launch 下发两段式接口，outputAddr 指向回读用的 device 缓冲成员，
 * expected 给出 host 侧期望值，release 回收资源。
 */
template <typename TensorsT>
struct AclnnExampleOps {
    int (*setup)(TensorsT&);
    int (*launch)(const TensorsT&, aclrtStream, void**);
    void* TensorsT::*outputAddr;
    std::vector<float> (*expected)(const TensorsT&);
    void (*release)(TensorsT&);
};

/** 样例主流程：Init -> setup -> launch -> 回读 -> 对拍 -> 打印 -> 释放 -> 收尾。对拍不过返回 1。 */
template <typename TensorsT>
int RunAclnnExample(const AclnnExampleOps<TensorsT>& ops, size_t outNumel, float atol = 1e-5f, float rtol = 1e-5f)
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    TensorsT tensors;
    void* workspaceAddr = nullptr;
    ret = ops.setup(tensors);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = ops.launch(tensors, stream, &workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> result;
    ret = CopyOut(tensors.*ops.outputAddr, result, outNumel);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const bool close = HostClose(result, ops.expected(tensors), atol, rtol);
    if (close) {
        PrintResult(result);
    }
    ops.release(tensors);
    Finalize(deviceId, stream, workspaceAddr);
    return close ? 0 : 1;
}

#endif
