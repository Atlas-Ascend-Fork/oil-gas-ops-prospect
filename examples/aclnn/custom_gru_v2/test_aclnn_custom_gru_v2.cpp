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
 * 两段式 aclnnCustomGRUV2 小样例：B=2、T=4、F_in=4、H=5，带 bias，单向。
 * gate 行序与 PyTorch nn.GRU 一致（weight_ih 行 [r; z; n]），tanh 用
 * 2·sigmoid(2x)−1 数值稳定形式。对拍基准为本文件内的 CPU float32 参考实现。
 */

#include <cmath>
#include <vector>

#include "aclnn_example.h"
#include "og_aclnn_decls.h"

namespace {

constexpr int64_t kB = 2;
constexpr int64_t kT = 4;
constexpr int64_t kF = 4;
constexpr int64_t kH = 5;
constexpr size_t kYNumel = static_cast<size_t>(kB * kT * kH);
constexpr size_t kHNumel = static_cast<size_t>(kB * kH);
constexpr size_t kXNumel = static_cast<size_t>(kB * kT * kF);
constexpr size_t kWiNumel = static_cast<size_t>(3 * kH * kF);
constexpr size_t kWhNumel = static_cast<size_t>(3 * kH * kH);
constexpr size_t kBiasNumel = static_cast<size_t>(3 * kH);

const std::vector<int64_t> kXShape = {kB, kT, kF};
const std::vector<int64_t> kWiShape = {3 * kH, kF};
const std::vector<int64_t> kWhShape = {3 * kH, kH};
const std::vector<int64_t> kBiasShape = {3 * kH};
const std::vector<int64_t> kYShape = {1, kB, kT, kH};
const std::vector<int64_t> kHShape = {1, kB, kH};

struct Tensors {
    std::vector<float> x;
    std::vector<float> wi;
    std::vector<float> wh;
    std::vector<float> bi;
    std::vector<float> bh;
    void* xDev = nullptr;
    void* wiDev = nullptr;
    void* whDev = nullptr;
    void* biDev = nullptr;
    void* bhDev = nullptr;
    void* yDev = nullptr;
    void* ohDev = nullptr;
    aclTensor* xT = nullptr;
    aclTensor* wiT = nullptr;
    aclTensor* whT = nullptr;
    aclTensor* biT = nullptr;
    aclTensor* bhT = nullptr;
    aclTensor* yT = nullptr;
    aclTensor* ohT = nullptr;

    Tensors() : x(kXNumel), wi(kWiNumel), wh(kWhNumel), bi(kBiasNumel), bh(kBiasNumel)
    {
        // 小尺寸确定性输入，便于 CPU 参考可复现。
        for (size_t i = 0; i < kXNumel; ++i) {
            x[i] = 0.125f * static_cast<float>(i % 7) - 0.375f;
        }
        for (size_t i = 0; i < kWiNumel; ++i) {
            wi[i] = 0.0625f * static_cast<float>(i % 5) - 0.125f;
        }
        for (size_t i = 0; i < kWhNumel; ++i) {
            wh[i] = 0.0625f * static_cast<float>(i % 3) - 0.0625f;
        }
        for (size_t i = 0; i < kBiasNumel; ++i) {
            bi[i] = 0.03125f * static_cast<float>(i % 3);
            bh[i] = -0.03125f * static_cast<float>(i % 2);
        }
    }
};

int SetupTensors(Tensors& t)
{
    struct {
        const std::vector<float>* host;
        const std::vector<int64_t>* shape;
        void** dev;
        aclTensor** tensor;
    } specs[] = {
        {&t.x, &kXShape, &t.xDev, &t.xT},       {&t.wi, &kWiShape, &t.wiDev, &t.wiT},
        {&t.wh, &kWhShape, &t.whDev, &t.whT},   {&t.bi, &kBiasShape, &t.biDev, &t.biT},
        {&t.bh, &kBiasShape, &t.bhDev, &t.bhT},
    };
    for (auto& spec : specs) {
        auto ret = CreateAclTensor(*spec.host, *spec.shape, spec.dev, aclDataType::ACL_FLOAT, spec.tensor);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    const std::vector<float> y(kYNumel, 0.0f);
    auto ret = CreateAclTensor(y, kYShape, &t.yDev, aclDataType::ACL_FLOAT, &t.yT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    const std::vector<float> oh(kHNumel, 0.0f);
    ret = CreateAclTensor(oh, kHShape, &t.ohDev, aclDataType::ACL_FLOAT, &t.ohT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

int LaunchOp(const Tensors& t, aclrtStream stream, void** workspaceAddr)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnCustomGRUV2GetWorkspaceSize(t.xT, t.wiT, t.whT, t.biT, t.bhT, false, t.yT, t.ohT, &workspaceSize,
                                                &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCustomGRUV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    ret = MallocWorkspace(workspaceSize, workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnCustomGRUV2(*workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCustomGRUV2 failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

float TanhStable(float v)
{
    // 与 kernel 一致的稳定 tanh：2·sigmoid(2x)−1。
    return 2.0f / (1.0f + std::exp(-2.0f * v)) - 1.0f;
}

std::vector<float> Expected(const Tensors& t)
{
    // CPU float32 参考：单层 GRU，gate 行序 [r; z; n]，与 PyTorch nn.GRU 一致。
    std::vector<float> y(kYNumel, 0.0f);
    std::vector<float> oh(kHNumel, 0.0f);
    std::vector<float> h(static_cast<size_t>(kH), 0.0f);
    for (int64_t b = 0; b < kB; ++b) {
        std::fill(h.begin(), h.end(), 0.0f);
        for (int64_t ts = 0; ts < kT; ++ts) {
            std::vector<float> gi(static_cast<size_t>(3 * kH), 0.0f);
            std::vector<float> gh(static_cast<size_t>(3 * kH), 0.0f);
            for (int64_t row = 0; row < 3 * kH; ++row) {
                gi[static_cast<size_t>(row)] = t.bi[static_cast<size_t>(row)];
                gh[static_cast<size_t>(row)] = t.bh[static_cast<size_t>(row)];
                for (int64_t k = 0; k < kF; ++k) {
                    gi[static_cast<size_t>(row)] += t.wi[static_cast<size_t>(row * kF + k)] *
                                                    t.x[static_cast<size_t>(b * kT * kF + ts * kF + k)];
                }
                for (int64_t k = 0; k < kH; ++k) {
                    gh[static_cast<size_t>(row)] += t.wh[static_cast<size_t>(row * kH + k)] * h[static_cast<size_t>(k)];
                }
            }
            std::vector<float> r(static_cast<size_t>(kH)), z(static_cast<size_t>(kH)), n(static_cast<size_t>(kH));
            for (int64_t d = 0; d < kH; ++d) {
                const size_t ud = static_cast<size_t>(d);
                r[ud] = 1.0f / (1.0f + std::exp(-(gi[ud] + gh[ud])));
                z[ud] = 1.0f /
                        (1.0f + std::exp(-(gi[static_cast<size_t>(kH) + ud] + gh[static_cast<size_t>(kH) + ud])));
                n[ud] = TanhStable(gi[static_cast<size_t>(2 * kH) + ud] + r[ud] * gh[static_cast<size_t>(2 * kH) + ud]);
                h[ud] = (1.0f - z[ud]) * n[ud] + z[ud] * h[ud];
                y[static_cast<size_t>(b * kT * kH + ts * kH) + ud] = h[ud];
            }
        }
        for (int64_t d = 0; d < kH; ++d) {
            oh[static_cast<size_t>(b * kH) + static_cast<size_t>(d)] = h[static_cast<size_t>(d)];
        }
    }
    // 样例只对拍 y；output_h 追加在尾部一并比对，保证两个输出都被验证。
    y.insert(y.end(), oh.begin(), oh.end());
    return y;
}

void Release(Tensors& t)
{
    DestroyTensors(&t.xT, 5);
    DestroyTensors(&t.yT, 2);
    void* inBufs[] = {t.xDev, t.wiDev, t.whDev, t.biDev, t.bhDev};
    FreeDeviceBuffers(inBufs, 5);
    void* outBufs[] = {t.yDev, t.ohDev};
    FreeDeviceBuffers(outBufs, 2);
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    Tensors tensors;
    void* workspaceAddr = nullptr;
    ret = SetupTensors(tensors);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = LaunchOp(tensors, stream, &workspaceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> yHost;
    ret = CopyOut(tensors.yDev, yHost, kYNumel);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<float> ohHost;
    ret = CopyOut(tensors.ohDev, ohHost, kHNumel);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 单一缓冲比对：y 在前、output_h 在后（与 Expected 的拼接顺序一致）。
    yHost.insert(yHost.end(), ohHost.begin(), ohHost.end());
    const bool close = HostClose(yHost, Expected(tensors), 3e-3f, 3e-3f);
    if (close) {
        PrintResult(yHost);
    } else {
        LOG_PRINT("aclnnCustomGRUV2 output mismatch against CPU reference\n");
    }
    Release(tensors);
    Finalize(deviceId, stream, workspaceAddr);
    return close ? 0 : 1;
}
