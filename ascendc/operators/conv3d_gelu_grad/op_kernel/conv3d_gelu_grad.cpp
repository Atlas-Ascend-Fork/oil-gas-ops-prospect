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
 * Conv3DGeluGrad 的向量核：dz = dy * gelu'(z)。
 *
 * gelu 与前向同一条 tanh 近似：
 *   gelu(z) = z / (1 + exp(-k * (z + c * z^3)))
 *   k = 1.5957691216057308, c = 0.044715
 * 导数：
 *   s = sigmoid(k * (z + c * z^3))
 *   gelu'(z) = s + z * s * (1 - s) * k * (1 + 3 * c * z^2)
 */

#include "kernel_operator.h"
#include "conv3d_gelu_grad_tiling.h"

using namespace AscendC;

namespace {

constexpr int32_t kBufferNum = 2;
constexpr float kGeluK = 1.5957691216057308f;
constexpr float kGeluC = 0.044715f;
constexpr float kGelu3C = 3.0f * kGeluC;

template <typename T>
__aicore__ inline void CopyTiling(T* dst, GM_ADDR tilingGm)
{
    auto* out = reinterpret_cast<uint32_t*>(dst);
    auto* src = reinterpret_cast<__gm__ uint32_t*>(tilingGm);
    for (uint32_t i = 0; i < sizeof(T) / sizeof(uint32_t); ++i) {
        out[i] = src[i];
    }
}

template <typename T>
class GeluBackward {
public:
    __aicore__ inline GeluBackward() {}

    __aicore__ inline void Init(GM_ADDR z, GM_ADDR dy, GM_ADDR dz, int64_t totalNum, int64_t blockFactor,
                                int64_t ubFactor)
    {
        int64_t offset = blockFactor * static_cast<int64_t>(GetBlockIdx());
        int64_t remainder = totalNum - offset;
        blockLength_ = (remainder > blockFactor) ? blockFactor : remainder;
        if (blockLength_ < 0) {
            blockLength_ = 0;
        }
        ubFactor_ = ubFactor;
        zGm_.SetGlobalBuffer((__gm__ T*)z + offset, blockLength_);
        dyGm_.SetGlobalBuffer((__gm__ T*)dy + offset, blockLength_);
        dzGm_.SetGlobalBuffer((__gm__ T*)dz + offset, blockLength_);
        pipe_.InitBuffer(zQueue_, kBufferNum, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(dyQueue_, kBufferNum, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(dzQueue_, kBufferNum, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(tmpBuf_, 4 * ubFactor_ * sizeof(T));
    }

    __aicore__ inline int64_t TileLen(int64_t index, int64_t loopCount) const
    {
        return (index == loopCount - 1) ? (blockLength_ - ubFactor_ * index) : ubFactor_;
    }

    __aicore__ inline void Process()
    {
        if (ubFactor_ <= 0 || blockLength_ <= 0) {
            return;
        }
        int64_t loopCount = (blockLength_ + ubFactor_ - 1) / ubFactor_;
        CopyIn(0, TileLen(0, loopCount));
        for (int64_t i = 0; i < loopCount; ++i) {
            if (i + 1 < loopCount) {
                CopyIn(i + 1, TileLen(i + 1, loopCount));
            }
            Compute(TileLen(i, loopCount));
            CopyOut(i, TileLen(i, loopCount));
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum)
    {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LocalTensor<T> zLocal = zQueue_.AllocTensor<T>();
        DataCopyPad(zLocal, zGm_[progress * ubFactor_], p, padParams);
        zQueue_.EnQue(zLocal);

        LocalTensor<T> dyLocal = dyQueue_.AllocTensor<T>();
        DataCopyPad(dyLocal, dyGm_[progress * ubFactor_], p, padParams);
        dyQueue_.EnQue(dyLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum)
    {
        LocalTensor<T> zLocal = zQueue_.template DeQue<T>();
        LocalTensor<T> dyLocal = dyQueue_.template DeQue<T>();
        LocalTensor<T> dzLocal = dzQueue_.template AllocTensor<T>();
        LocalTensor<T> tmp = tmpBuf_.template Get<T>();
        LocalTensor<T> z2 = tmp[0];
        LocalTensor<T> inner = tmp[ubFactor_];
        LocalTensor<T> sigmoid = tmp[ubFactor_ * 2];
        LocalTensor<T> acc = tmp[ubFactor_ * 3];

        int32_t n = static_cast<int32_t>(currentNum);
        Mul(z2, zLocal, zLocal, n);
        Muls(inner, z2, (T)kGeluC, n);
        Mul(inner, inner, zLocal, n);
        Add(inner, inner, zLocal, n);
        Muls(z2, z2, (T)kGelu3C, n);
        Adds(z2, z2, (T)1.0, n);
        Muls(z2, z2, (T)kGeluK, n);
        Muls(inner, inner, (T)kGeluK, n);
        Muls(inner, inner, (T)-1.0, n);
        Exp(inner, inner, n);
        Adds(inner, inner, (T)1.0, n);
        Duplicate(acc, (T)1.0, n);
        Div(sigmoid, acc, inner, n);
        Sub(acc, acc, sigmoid, n);
        Mul(acc, acc, sigmoid, n);
        Mul(acc, acc, z2, n);
        Mul(acc, acc, zLocal, n);
        Add(acc, acc, sigmoid, n);
        Mul(dzLocal, dyLocal, acc, n);

        dzQueue_.template EnQue<T>(dzLocal);
        zQueue_.FreeTensor(zLocal);
        dyQueue_.FreeTensor(dyLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum)
    {
        LocalTensor<T> dzLocal = dzQueue_.template DeQue<T>();
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPad(dzGm_[progress * ubFactor_], dzLocal, p);
        dzQueue_.FreeTensor(dzLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> zQueue_;
    TQue<QuePosition::VECIN, kBufferNum> dyQueue_;
    TQue<QuePosition::VECOUT, kBufferNum> dzQueue_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    GlobalTensor<T> zGm_;
    GlobalTensor<T> dyGm_;
    GlobalTensor<T> dzGm_;
    int64_t blockLength_ = 0;
    int64_t ubFactor_ = 0;
};

} // namespace

extern "C" __global__ __aicore__ void conv3d_gelu_grad(GM_ADDR z, GM_ADDR grad_y, GM_ADDR grad_z, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    if (TILING_KEY_IS(0)) {
        Conv3DGeluGradTilingData tilingData;
        CopyTiling(&tilingData, tiling);
        GeluBackward<float> op;
        op.Init(z, grad_y, grad_z, tilingData.totalNum, tilingData.blockFactor, tilingData.ubFactor);
        op.Process();
    }
}
