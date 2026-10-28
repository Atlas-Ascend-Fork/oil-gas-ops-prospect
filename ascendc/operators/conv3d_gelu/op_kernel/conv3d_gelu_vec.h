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
 * Conv3DGelu 向量（AIV）阶段：对 conv3d 的输出做逐元素 tanh 近似 GELU。
 *
 * 与 oil_gas_ops/operators/gelu 算法一致（torch.nn.functional.gelu(approximate='tanh')
 * 的有理等价形式）：
 *   gelu(x) = x / (1 + e^(-1.5957691216057308 * (x + 0.044715 * x^3)))
 *
 * 输入即 cube 阶段写入的 conv 结果（GM），原地写回同一缓冲（gelu 为逐元素运算，
 * 原地安全）。每个 AIV 核处理 [blockIdx*blockFactor, +blockLength) 的连续元素区间。
 */

#ifndef CONV3D_GELU_VEC_H
#define CONV3D_GELU_VEC_H

#include "kernel_operator.h"

namespace Conv3dGeluVec {

using namespace AscendC;

constexpr int32_t GELU_BUFFER_NUM = 2;
constexpr float GELU_CUBE_COEFF = 0.044715f;
constexpr float GELU_NEG_SQRT_2_OVER_PI_X2 = -1.5957691216057308f;

template <typename T>
class GeluInplace {
public:
    __aicore__ inline GeluInplace() {}

    __aicore__ inline void Init(GM_ADDR y, int64_t totalNum, int64_t blockFactor, int64_t ubFactor)
    {
        int64_t offset = blockFactor * static_cast<int64_t>(GetBlockIdx());
        int64_t remainder = totalNum - offset;
        blockLength_ = (remainder > blockFactor) ? blockFactor : remainder;
        if (blockLength_ < 0) {
            blockLength_ = 0;
        }
        ubFactor_ = ubFactor;
        gm_.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
        pipe_.InitBuffer(inQueue_, GELU_BUFFER_NUM, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(outQueue_, GELU_BUFFER_NUM, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(tmpBuf_, ubFactor_ * sizeof(T));
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
        // 双缓冲：下一 tile 的 GM->UB 与当前 tile 的 GELU 重叠。
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
        LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, gm_[progress * ubFactor_], p, padParams);
        inQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum)
    {
        LocalTensor<T> xLocal = inQueue_.template DeQue<T>();
        LocalTensor<T> yLocal = outQueue_.template AllocTensor<T>();
        LocalTensor<T> tmp = tmpBuf_.template Get<T>();

        int32_t n = static_cast<int32_t>(currentNum);
        Mul(tmp, xLocal, xLocal, n);                      // x^2
        Mul(tmp, tmp, xLocal, n);                         // x^3
        Muls(tmp, tmp, (T)GELU_CUBE_COEFF, n);            // 0.044715 * x^3
        Add(tmp, tmp, xLocal, n);                         // x + 0.044715 * x^3
        Muls(tmp, tmp, (T)GELU_NEG_SQRT_2_OVER_PI_X2, n); // -1.5957691 * (...)
        Exp(tmp, tmp, n);                                 // e^(...)
        Adds(tmp, tmp, (T)1.0, n);                        // 1 + e^(...)
        Div(yLocal, xLocal, tmp, n);                      // x / (1 + e^(...))

        outQueue_.template EnQue<T>(yLocal);
        inQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum)
    {
        LocalTensor<T> yLocal = outQueue_.template DeQue<T>();
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        p.srcStride = 0;
        p.dstStride = 0;
        DataCopyPad(gm_[progress * ubFactor_], yLocal, p);
        outQueue_.FreeTensor(yLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, GELU_BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, GELU_BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    GlobalTensor<T> gm_;
    int64_t blockLength_ = 0;
    int64_t ubFactor_ = 0;
};

} // namespace Conv3dGeluVec

#endif // CONV3D_GELU_VEC_H
