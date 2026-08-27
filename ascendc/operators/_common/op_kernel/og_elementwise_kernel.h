/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 逐元素 AIV 算子的公共 kernel 基建（方案 C：多路拼接单槽 + double buffer）。
 *
 * complex_mul / complex_mul_grad 的 kernel 只在"搬哪几路、算什么"上不同，
 * 核间切分、UB 队列布局、DataCopyPad 样板、tile 循环骨架完全一致，抽到这里。
 *
 * 用 CRTP 而不是虚函数：AIV 上没有虚表，模板在编译期展开为直调。派生类需提供
 *   LocalTensor<float> LoadTile(int64_t gmOff, int32_t cur)                 搬入一拍
 *   void ComputeTile(LocalTensor<float> in, LocalTensor<float> out,
 *                    LocalTensor<float> tmp, int32_t cur)                   算一拍
 *   void StoreTile(int64_t gmOff, LocalTensor<float> out, int32_t cur)      搬出一拍
 *
 * 本文件由 scripts/merge_operators.py 从 operators/_common/op_kernel/ 拷进每个算子
 * 合并后的 op_kernel/<op>/ 目录，故用无路径 include。
 */

#ifndef OG_ELEMENTWISE_KERNEL_H
#define OG_ELEMENTWISE_KERNEL_H

#include "kernel_operator.h"

namespace OgOpsKernel {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
/** 32B / sizeof(float)：核间切分粒度，保证每核起始地址 32B 对齐。 */
constexpr int64_t ALIGN_FLOATS = 8;

/** 向上对齐到 kAlign 的倍数（除数是编译期正常量）。 */
template <int64_t kAlign>
__aicore__ inline int64_t AlignUp(int64_t v)
{
    static_assert(kAlign > 0, "kAlign must be positive");
    return (v + kAlign - 1) / kAlign * kAlign;
}

/**
 * 核间切分：per = ceil(total / cores) 上取整到 ALIGN_FLOATS，start = blockIdx * per。
 * 越界核拿到 len = 0，由调用方直接返回。
 *
 * 除数先用三元表达式钳到 >= 1 再参与除法（host 侧本就把 aivNum==0 兜底成 1，
 * 这里是第二道保险）。钳位与除法在同一表达式链上，静态检查不必做路径分析
 * 即可确定除数为正。
 */
__aicore__ inline void SplitByCore(int64_t total, int64_t coreNum, int64_t &offset, int64_t &len)
{
    offset = 0;
    len = 0;
    if (total <= 0) {
        return;
    }
    const int64_t cores = coreNum > 0 ? coreNum : 1;
    const int64_t per = AlignUp<ALIGN_FLOATS>((total + cores - 1) / cores);
    const int64_t start = static_cast<int64_t>(GetBlockIdx()) * per;
    if (start >= total) {
        return;
    }
    offset = start;
    len = total - start < per ? total - start : per;
}

/** 把 GM 上的 tiling POD 按 uint32 逐字读进片上结构体。 */
template <typename TilingT>
__aicore__ inline void CopyTilingData(TilingT *dst, GM_ADDR tilingGm)
{
    auto *out = reinterpret_cast<uint32_t *>(dst);
    auto *src = reinterpret_cast<__gm__ uint32_t *>(tilingGm);
    for (uint32_t i = 0; i < sizeof(TilingT) / sizeof(uint32_t); ++i) {
        out[i] = src[i];
    }
}

template <typename Derived>
class ElementwiseTileKernel {
public:
    __aicore__ inline ElementwiseTileKernel() {}

    /** tile 循环骨架：搬入 → 计算 → 搬出，三步由派生类实现。 */
    __aicore__ inline void Process()
    {
        if (len_ <= 0 || tileFloats_ <= 0) {
            return;
        }
        Derived *self = static_cast<Derived *>(this);
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        for (int64_t done = 0; done < len_; done += tileFloats_) {
            const int32_t cur = TileLen(done);
            const int64_t gmOff = offset_ + done;

            LocalTensor<float> in = self->LoadTile(gmOff, cur);
            LocalTensor<float> out = outQue_.AllocTensor<float>();
            self->ComputeTile(in, out, tmp, cur);
            inQue_.FreeTensor(in);
            outQue_.EnQue(out);

            out = outQue_.DeQue<float>();
            self->StoreTile(gmOff, out, cur);
            outQue_.FreeTensor(out);
        }
    }

protected:
    /**
     * 记录切分并申请 UB：inQue 单槽拼 inSlots 路、outQue 单槽拼 outSlots 路，
     * 外加一路 tmp。份数必须与 host 侧 UB_SLOT_FACTOR 一致
     * （BUFFER_NUM * (inSlots + outSlots) + 1）。
     */
    __aicore__ inline void InitTiles(int64_t totalElems, int64_t tileFloats, int64_t coreNum, int32_t inSlots,
                                     int32_t outSlots, TPipe *pipe)
    {
        totalElems_ = totalElems;
        tileFloats_ = tileFloats;
        coreNum_ = coreNum;
        SplitByCore(totalElems_, coreNum_, offset_, len_);

        const uint32_t tileBytes = static_cast<uint32_t>(tileFloats_) * sizeof(float);
        pipe->InitBuffer(inQue_, BUFFER_NUM, static_cast<uint32_t>(inSlots) * tileBytes);
        pipe->InitBuffer(outQue_, BUFFER_NUM, static_cast<uint32_t>(outSlots) * tileBytes);
        pipe->InitBuffer(tmpBuf_, tileBytes);
    }

    /** 每路在单槽内的元素步长（第 k 路为 tensor[k * SlotStride()]）。 */
    __aicore__ inline uint32_t SlotStride() const
    {
        return static_cast<uint32_t>(tileFloats_);
    }

    /** 本拍实际元素数（尾拍可能短于 tileFloats_）。 */
    __aicore__ inline int32_t TileLen(int64_t done) const
    {
        const int64_t rest = len_ - done;
        return static_cast<int32_t>(rest < tileFloats_ ? rest : tileFloats_);
    }

    __aicore__ inline void CopyInPad(LocalTensor<float> dst, GlobalTensor<float> src, int32_t cur)
    {
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(cur) * sizeof(float);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(dst, src, params, pad);
    }

    __aicore__ inline void CopyOutPad(GlobalTensor<float> dst, LocalTensor<float> src, int32_t cur)
    {
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(cur) * sizeof(float);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPad(dst, src, params);
    }

    TQue<TPosition::VECIN, BUFFER_NUM> inQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQue_;
    TBuf<TPosition::VECCALC> tmpBuf_;

private:
    int64_t totalElems_ = 0;
    int64_t tileFloats_ = 0;
    int64_t coreNum_ = 0;
    int64_t offset_ = 0;
    int64_t len_ = 0;
};

} // namespace OgOpsKernel

#endif // OG_ELEMENTWISE_KERNEL_H
