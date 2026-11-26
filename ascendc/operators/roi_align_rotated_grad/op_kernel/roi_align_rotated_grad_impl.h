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
 * ROIAlignRotated 反向 AIV kernel（正确性基线）。
 *
 * grad_input 按 64B slab 独占切分：每核在 UB 内清零、把命中本 slab 的
 * grad_output 贡献累加进来，最后整片 DataCopyPad 回写。没有 atomicAdd。
 *
 * 一个 slab 可能跨多个 (n,c) 平面；kernel 从 flat [start,end) 反推 n 范围，
 * 再按 c 范围只处理与 slab 相交的通道，避免每个 sample 扫全部 C。
 */

#ifndef ROI_ALIGN_ROTATED_GRAD_IMPL_H
#define ROI_ALIGN_ROTATED_GRAD_IMPL_H

#include "kernel_operator.h"
#include "og_index_kernel.h"

#include "roi_align_rotated_grad_tiling.h"

namespace RoiAlignRotatedGrad {

using namespace AscendC;

constexpr int64_t kRoiChunk = 512;
constexpr int64_t kMinBufferFloats = 64; // 向量指令最小 repeat 粒度

class RoiAlignRotatedGradKernel {
public:
    __aicore__ inline RoiAlignRotatedGradKernel() {}

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR rois, GM_ADDR gradInput,
                                const ROIAlignRotatedGradTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        gradOutGm_.SetGlobalBuffer((__gm__ float*)gradOutput,
                                   tiling_.k * tiling_.c * tiling_.pooledH * tiling_.pooledW);
        roisGm_.SetGlobalBuffer((__gm__ float*)rois, tiling_.k * 6);
        gradInputGm_.SetGlobalBuffer((__gm__ float*)gradInput, tiling_.totalIn);

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
        if (endElem > tiling_.totalIn) {
            endElem = tiling_.totalIn;
        }
        const int64_t slabLen = endElem - startElem;
        if (slabLen <= 0) {
            return;
        }

        LocalTensor<float> slab = slabBuf_.Get<float>();
        Duplicate(slab, 0.0F, static_cast<uint32_t>(slabLen));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);

        const int64_t planeSize = tiling_.h * tiling_.w;
        const int64_t perN = tiling_.c * planeSize;
        if (perN <= 0 || tiling_.k <= 0) {
            // K=0 或空输入：仍需把零写回。
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyExtParams params{1, static_cast<uint32_t>(slabLen * sizeof(float)), 0, 0, 0};
            DataCopyPad(gradInputGm_[startElem], slab, params);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            return;
        }
        const int64_t nStart = startElem / perN;
        const int64_t nEnd = (endElem - 1) / perN;

        for (int64_t chunkStart = 0; chunkStart < tiling_.k; chunkStart += kRoiChunk) {
            int64_t chunkCount = tiling_.k - chunkStart;
            if (chunkCount > kRoiChunk) {
                chunkCount = kRoiChunk;
            }
            LoadRoiChunk(chunkStart, chunkCount);
            for (int64_t kk = 0; kk < chunkCount; ++kk) {
                ProcessRoi(chunkStart + kk, kk, slab, startElem, endElem, nStart, nEnd);
            }
        }

        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyExtParams params{1, static_cast<uint32_t>(slabLen * sizeof(float)), 0, 0, 0};
        DataCopyPad(gradInputGm_[startElem], slab, params);
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
        Cos(cosBuf_.Get<float>(), angle, static_cast<uint32_t>(chunkCount));
        Sin(sinBuf_.Get<float>(), angle, static_cast<uint32_t>(chunkCount));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline void ProcessRoi(int64_t k, int64_t kk, LocalTensor<float>& slab, int64_t startElem,
                                      int64_t endElem, int64_t nStart, int64_t nEnd)
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
        if (n < nStart || n > nEnd || n < 0 || n >= tiling_.n) {
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

        const int64_t planeSize = tiling_.h * tiling_.w;
        const int64_t perN = tiling_.c * planeSize;
        const int64_t nBase = n * perN;

        int64_t cStart = (startElem > nBase) ? (startElem - nBase) / planeSize : 0;
        int64_t cEnd = (endElem - 1 >= nBase) ? (endElem - 1 - nBase) / planeSize : -1;
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

        const int64_t pooled = tiling_.pooledH * tiling_.pooledW;
        for (int64_t ph = 0; ph < tiling_.pooledH; ++ph) {
            for (int64_t pw = 0; pw < tiling_.pooledW; ++pw) {
                for (int64_t iy = 0; iy < gridH; ++iy) {
                    const float yy = roiStartH + static_cast<float>(ph) * binH +
                                     (static_cast<float>(iy) + 0.5F) * binH / static_cast<float>(gridH);
                    for (int64_t ix = 0; ix < gridW; ++ix) {
                        const float xx = roiStartW + static_cast<float>(pw) * binW +
                                         (static_cast<float>(ix) + 0.5F) * binW / static_cast<float>(gridW);
                        const float y = yy * cosT - xx * sinT + roiCenterH;
                        const float x = yy * sinT + xx * cosT + roiCenterW;
                        const OgIndex::BilinearGrad grad = OgIndex::ComputeBilinearGrad(tiling_.h, tiling_.w, y, x);
                        if (!grad.valid) {
                            continue;
                        }
                        const int64_t outOffset = ph * tiling_.pooledW + pw;
                        for (int64_t c = cStart; c <= cEnd; ++c) {
                            const float gradOut = gradOutGm_.GetValue((k * tiling_.c + c) * pooled + outOffset);
                            if (gradOut == 0.0F) {
                                continue;
                            }
                            const int64_t planeBase = nBase + c * planeSize;
                            AddOne(slab, startElem, endElem, planeBase + grad.yLow * tiling_.w + grad.xLow,
                                   gradOut * grad.w1 * invCount);
                            AddOne(slab, startElem, endElem, planeBase + grad.yLow * tiling_.w + grad.xHigh,
                                   gradOut * grad.w2 * invCount);
                            AddOne(slab, startElem, endElem, planeBase + grad.yHigh * tiling_.w + grad.xLow,
                                   gradOut * grad.w3 * invCount);
                            AddOne(slab, startElem, endElem, planeBase + grad.yHigh * tiling_.w + grad.xHigh,
                                   gradOut * grad.w4 * invCount);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void AddOne(LocalTensor<float>& slab, int64_t startElem, int64_t endElem, int64_t index,
                                  float value)
    {
        if (index < startElem || index >= endElem) {
            return;
        }
        const uint32_t offset = static_cast<uint32_t>(index - startElem);
        slab.SetValue(offset, slab.GetValue(offset) + value);
    }

    TBuf<TPosition::VECCALC> slabBuf_;
    TBuf<TPosition::VECCALC> angleBuf_;
    TBuf<TPosition::VECCALC> cosBuf_;
    TBuf<TPosition::VECCALC> sinBuf_;

    GlobalTensor<float> gradOutGm_;
    GlobalTensor<float> roisGm_;
    GlobalTensor<float> gradInputGm_;
    ROIAlignRotatedGradTilingData tiling_{};
};

} // namespace RoiAlignRotatedGrad

#endif // ROI_ALIGN_ROTATED_GRAD_IMPL_H
