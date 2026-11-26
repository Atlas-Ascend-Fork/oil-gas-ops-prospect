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
 * ROIAlignRotated 前向 AIV kernel（正确性基线）。
 *
 * 输出 [K,C,PH,PW] 按 64B slab 切分，每个核独占连续 slab：
 *   - slab 在 UB 内清零；
 *   - 按 ROI 分块把 rois/angle 搬入 UB，向量 Cos/Sin 预计算；
 *   - 对 slab 覆盖到的 (k,c,ph,pw) 做双线性采样并写 UB；
 *   - 最后 DataCopyPad 整片回写。
 *
 * 基线用标量 GM 读 + UB 标量写，优先保证逐元素语义与 CUDA 一致；
 * 性能优化（平面驻留 + Gather）在 M5 阶段进行。
 */

#ifndef ROI_ALIGN_ROTATED_IMPL_H
#define ROI_ALIGN_ROTATED_IMPL_H

#include "kernel_operator.h"
#include "og_index_kernel.h"

#include "roi_align_rotated_tiling.h"

namespace RoiAlignRotated {

using namespace AscendC;

constexpr int64_t kRoiChunk = 512;
constexpr int64_t kMinBufferFloats = 64; // 向量指令最小 repeat 粒度

class RoiAlignRotatedKernel {
public:
    __aicore__ inline RoiAlignRotatedKernel() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR rois, GM_ADDR out, const ROIAlignRotatedTilingData& tiling,
                                TPipe* pipe)
    {
        tiling_ = tiling;
        inputGm_.SetGlobalBuffer((__gm__ float*)input, tiling_.n * tiling_.c * tiling_.h * tiling_.w);
        roisGm_.SetGlobalBuffer((__gm__ float*)rois, tiling_.k * 6);
        outGm_.SetGlobalBuffer((__gm__ float*)out, tiling_.totalOut);

        int64_t slabFloats = tiling_.slabFloats;
        if (slabFloats < kMinBufferFloats) {
            slabFloats = kMinBufferFloats;
        }
        const int64_t slabBytes = slabFloats * static_cast<int64_t>(sizeof(float));
        pipe->InitBuffer(slabBuf_, static_cast<uint32_t>(slabBytes));
        pipe->InitBuffer(angleBuf_, static_cast<uint32_t>(kRoiChunk * sizeof(float)));
        pipe->InitBuffer(cosBuf_, static_cast<uint32_t>(kRoiChunk * sizeof(float)));
        pipe->InitBuffer(sinBuf_, static_cast<uint32_t>(kRoiChunk * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        if (tiling_.numSlabs <= 0 || tiling_.slabFloats <= 0) {
            return;
        }
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= tiling_.blockDim) {
            return;
        }

        const int64_t extra = blockIdx < tiling_.remSlabs ? blockIdx : tiling_.remSlabs;
        const int64_t startSlab = blockIdx * tiling_.baseSlabs + extra;
        const int64_t endSlab = startSlab + tiling_.baseSlabs + (blockIdx < tiling_.remSlabs ? 1 : 0);
        for (int64_t slabIdx = startSlab; slabIdx < endSlab; ++slabIdx) {
            ProcessSlab(slabIdx);
        }
    }

private:
    __aicore__ inline void ProcessSlab(int64_t slabIdx)
    {
        int64_t startElem = slabIdx * tiling_.slabFloats;
        int64_t endElem = startElem + tiling_.slabFloats;
        if (endElem > tiling_.totalOut) {
            endElem = tiling_.totalOut;
        }
        const int64_t slabLen = endElem - startElem;
        if (slabLen <= 0) {
            return;
        }

        LocalTensor<float> slab = slabBuf_.Get<float>();
        Duplicate(slab, 0.0F, static_cast<uint32_t>(slabLen));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        const int64_t tile = tiling_.pooledH * tiling_.pooledW;
        const int64_t perK = tiling_.c * tile;
        if (perK <= 0) {
            return;
        }
        const int64_t kStart = startElem / perK;
        const int64_t kEnd = (endElem - 1) / perK;

        for (int64_t chunkStart = (kStart / kRoiChunk) * kRoiChunk; chunkStart <= kEnd; chunkStart += kRoiChunk) {
            int64_t chunkCount = tiling_.k - chunkStart;
            if (chunkCount > kRoiChunk) {
                chunkCount = kRoiChunk;
            }
            if (chunkCount <= 0) {
                break;
            }
            if (chunkStart + chunkCount <= kStart) {
                continue;
            }
            LoadRoiChunk(chunkStart, chunkCount);
            const int64_t from = OgIndex::MaxInt64(kStart, chunkStart) - chunkStart;
            const int64_t to = OgIndex::MinInt64(kEnd, chunkStart + chunkCount - 1) - chunkStart;
            for (int64_t kk = from; kk <= to; ++kk) {
                ProcessRoi(chunkStart + kk, kk, slab, startElem, endElem);
            }
        }

        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams params{1, static_cast<uint32_t>(slabLen * sizeof(float)), 0, 0, 0};
        DataCopyPad(outGm_[startElem], slab, params);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void LoadRoiChunk(int64_t chunkStart, int64_t chunkCount)
    {
        // 基线：ROI 字段用标量 GM 读，angle 逐元素写 UB 后向量 Cos/Sin。
        // DataCopyPad 的多 block 步长语义在不同 CANN 小版本上不完全一致，
        // 这里先规避，保证语义正确；M5 再做批量搬运优化。
        LocalTensor<float> angle = angleBuf_.Get<float>();
        for (int64_t i = 0; i < chunkCount; ++i) {
            const float angleDeg = roisGm_.GetValue((chunkStart + i) * 6 + 5);
            angle.SetValue(static_cast<uint32_t>(i), angleDeg);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        Muls(angle, angle, OgIndex::kPi / 180.0F, static_cast<uint32_t>(chunkCount));
        LocalTensor<float> cosT = cosBuf_.Get<float>();
        LocalTensor<float> sinT = sinBuf_.Get<float>();
        Cos(cosT, angle, static_cast<uint32_t>(chunkCount));
        Sin(sinT, angle, static_cast<uint32_t>(chunkCount));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline void ProcessRoi(int64_t k, int64_t kk, LocalTensor<float>& slab, int64_t startElem,
                                      int64_t endElem)
    {
        const int64_t roiBase = k * 6;
        const float batch = roisGm_.GetValue(roiBase + 0);
        const float centerX = roisGm_.GetValue(roiBase + 1);
        const float centerY = roisGm_.GetValue(roiBase + 2);
        const float roiW = roisGm_.GetValue(roiBase + 3);
        const float roiH = roisGm_.GetValue(roiBase + 4);
        const float cosT = cosBuf_.Get<float>().GetValue(static_cast<uint32_t>(kk));
        const float sinT = sinBuf_.Get<float>().GetValue(static_cast<uint32_t>(kk));

        const int64_t n = static_cast<int64_t>(batch);
        if (n < 0 || n >= tiling_.n) {
            return;
        }
        const float offset = 0.5F;
        const float roiCenterW = centerX * tiling_.spatialScale - offset;
        const float roiCenterH = centerY * tiling_.spatialScale - offset;
        const float scaledW = roiW * tiling_.spatialScale;
        const float scaledH = roiH * tiling_.spatialScale;
        if (scaledW < 0.0F || scaledH < 0.0F) {
            return;
        }

        const float binW = scaledW / static_cast<float>(tiling_.pooledW);
        const float binH = scaledH / static_cast<float>(tiling_.pooledH);
        int64_t gridH = tiling_.samplingRatio > 0 ? tiling_.samplingRatio :
                                                    OgIndex::CeilToInt(scaledH / static_cast<float>(tiling_.pooledH));
        int64_t gridW = tiling_.samplingRatio > 0 ? tiling_.samplingRatio :
                                                    OgIndex::CeilToInt(scaledW / static_cast<float>(tiling_.pooledW));
        if (gridH <= 0 || gridW <= 0) {
            return;
        }
        const int64_t count = gridH * gridW;
        const float invCount = 1.0F / static_cast<float>(count);

        const float roiStartH = -scaledH / 2.0F;
        const float roiStartW = -scaledW / 2.0F;

        const int64_t tile = tiling_.pooledH * tiling_.pooledW;
        const int64_t perK = tiling_.c * tile;
        const int64_t kBase = k * perK;

        int64_t cStart = (startElem > kBase) ? (startElem - kBase) / tile : 0;
        int64_t cEnd = (endElem - 1 >= kBase) ? (endElem - 1 - kBase) / tile : -1;
        if (cEnd < 0) {
            return;
        }
        if (cStart < 0) {
            cStart = 0;
        }
        if (cEnd >= tiling_.c) {
            cEnd = tiling_.c - 1;
        }
        if (cStart > cEnd) {
            return;
        }

        for (int64_t c = cStart; c <= cEnd; ++c) {
            const int64_t tileStart = kBase + c * tile;
            const int64_t lo = OgIndex::MaxInt64(startElem, tileStart);
            const int64_t hi = OgIndex::MinInt64(endElem, tileStart + tile);
            if (lo >= hi) {
                continue;
            }
            const int64_t inputBase = (n * tiling_.c + c) * tiling_.h * tiling_.w;
            for (int64_t flat = lo; flat < hi; ++flat) {
                const int64_t b = flat - tileStart;
                const int64_t ph = b / tiling_.pooledW;
                const int64_t pw = b % tiling_.pooledW;
                float output = 0.0F;
                for (int64_t iy = 0; iy < gridH; ++iy) {
                    const float yy = roiStartH + static_cast<float>(ph) * binH +
                                     (static_cast<float>(iy) + 0.5F) * binH / static_cast<float>(gridH);
                    for (int64_t ix = 0; ix < gridW; ++ix) {
                        const float xx = roiStartW + static_cast<float>(pw) * binW +
                                         (static_cast<float>(ix) + 0.5F) * binW / static_cast<float>(gridW);
                        const float y = yy * cosT - xx * sinT + roiCenterH;
                        const float x = yy * sinT + xx * cosT + roiCenterW;
                        output += OgIndex::Bilinear(inputGm_, inputBase, tiling_.h, tiling_.w, y, x);
                    }
                }
                slab.SetValue(static_cast<uint32_t>(flat - startElem), output * invCount);
            }
        }
    }

    TBuf<TPosition::VECCALC> slabBuf_;
    TBuf<TPosition::VECCALC> angleBuf_;
    TBuf<TPosition::VECCALC> cosBuf_;
    TBuf<TPosition::VECCALC> sinBuf_;

    GlobalTensor<float> inputGm_;
    GlobalTensor<float> roisGm_;
    GlobalTensor<float> outGm_;
    ROIAlignRotatedTilingData tiling_{};
};

} // namespace RoiAlignRotated

#endif // ROI_ALIGN_ROTATED_IMPL_H
