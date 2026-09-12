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
 * FusedBiasSoftmax 切分算法核心（不依赖 platform_ascendc / gert / CANN 头）。
 *
 * 抽出目的：tiling 入口依赖 gert::TilingContext，平台读取依赖 PlatformAscendC，
 * 两者在 CANN 二进制包下都无法在 UT 里构造；把与平台无关的切分算法隔离到本文件
 * 后，ophost UT（tests/ascendc/ophost/）直接驱动真实算法（无卡可跑）。
 *
 * UB 预算模型（每行字节数，padLineNum = width 对齐到 32B 的元素数）：
 *   x 载入 + y 写出 + xF32 中转(非 fp32) + 各 bias 载入 + 各 bias F32 中转(非 fp32)
 * 高阶 SoftMax 预留 32K 共享 buffer；广播 bias1 附加 distinct 行缓存。
 */

#ifndef FUSED_BIAS_SOFTMAX_TILING_CORE_H
#define FUSED_BIAS_SOFTMAX_TILING_CORE_H

#include <cstdint>

namespace FusedBiasSoftmaxHost {

constexpr uint64_t FBS_ALIGNED_BYTES = 32;
constexpr uint64_t FBS_FP32_SIZE = 4;
constexpr uint64_t FBS_SOFTMAX_BUF_SIZE = 32 * 1024;
constexpr uint64_t FBS_XY_PARAMS = 2;

// tiling key：dtype 维度（与 kernel 侧 TILING_KEY_IS 一致）
constexpr uint64_t FBS_KEY_FP32 = 0;
constexpr uint64_t FBS_KEY_FP16 = 1;
constexpr uint64_t FBS_KEY_BF16 = 2;

/** 切分结果（plain POD，UT 可直接断言）。 */
struct FusedBiasSoftmaxCoreParams {
    uint64_t paddingNum = 0;       // 对齐需要补充的数据个数
    uint64_t padLineNum = 0;       // 对齐后的一行长度（元素）
    uint64_t coreNum = 0;          // AIV 核数
    uint64_t headCoreNum = 0;      // 大核数量
    uint64_t lineHeadCore = 0;     // 大核处理的行数
    uint64_t iterHeadCore = 0;     // 大核循环数
    uint64_t lineHeadIter = 0;     // 大核每次循环行数
    uint64_t lineLastHeadIter = 0; // 大核尾循环行数
    uint64_t lineTailCore = 0;     // 小核处理的行数
    uint64_t iterTailCore = 0;     // 小核循环数
    uint64_t lineTailIter = 0;     // 小核每次循环行数
    uint64_t lineLastTailIter = 0; // 小核尾循环行数
};

/** blockDim 与 tilingKey（dtype key 由 impl 侧按 ge dtype 计算后传入）。 */
struct FusedBiasSoftmaxPlan {
    uint64_t blockDim = 0;
    uint64_t tilingKey = 0;
};

inline uint64_t FbsCeilDiv(uint64_t a, uint64_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }

/** width 对齐到 32B 的元素数与 padding 个数。 */
inline void ComputePadding(uint64_t width, uint64_t dtypeSize, uint64_t& paddingNum, uint64_t& padLineNum)
{
    if (dtypeSize == 0) { // G.EXP.22 防御：调用方已校验，此处兜底
        paddingNum = 0;
        padLineNum = width;
        return;
    }
    const uint64_t alignedBlock = FBS_ALIGNED_BYTES / dtypeSize;
    const uint64_t left = width % alignedBlock;
    paddingNum = left > 0 ? alignedBlock - left : 0u;
    padLineNum = width + paddingNum;
}

/**
 * 每行 UB 字节数（UB 预算模型，见文件头）。
 * bcastBias1 为真时附加广播 bias1 的 distinct 行缓存（≤2 行）与单行 fp32 转换暂存。
 */
inline uint64_t ComputeMaxByteLine(uint64_t padLineNum, uint64_t dtypeSize, bool isFp32, uint64_t hasB1, uint64_t hasB2,
                                   bool b1Full)
{
    uint64_t biasPerElem = 0;
    if (hasB2 != 0) {
        biasPerElem += dtypeSize + (isFp32 ? 0 : FBS_FP32_SIZE);
    }
    if (hasB1 != 0) {
        biasPerElem += dtypeSize + (isFp32 ? 0 : FBS_FP32_SIZE);
    }
    uint64_t maxByteLine = padLineNum * (FBS_XY_PARAMS * dtypeSize + (isFp32 ? 0 : FBS_FP32_SIZE) + biasPerElem);
    if (hasB1 != 0 && !b1Full) {
        maxByteLine += (2 * padLineNum * dtypeSize) + (isFp32 ? 0 : padLineNum * FBS_FP32_SIZE);
    }
    return maxByteLine;
}

/**
 * 行切分：totalLine 均分到 aivNum 核，每核按 rowsPerIter 分拍，尾拍取余。
 * 大核 = totalLine % aivNum 个，各多处理 1 行。
 */
inline void ComputeRowSplit(uint64_t totalLine, uint64_t aivNum, uint64_t rowsPerIter, uint64_t& headCoreNum,
                            uint64_t& lineHeadCore, uint64_t& iterHeadCore, uint64_t& lineHeadIter,
                            uint64_t& lineLastHeadIter, uint64_t& lineTailCore, uint64_t& iterTailCore,
                            uint64_t& lineTailIter, uint64_t& lineLastTailIter)
{
    if (aivNum == 0 || rowsPerIter == 0) { // G.EXP.22 防御：调用方已校验，此处兜底
        headCoreNum = 0;
        lineHeadCore = 0;
        iterHeadCore = 0;
        lineHeadIter = 0;
        lineLastHeadIter = 0;
        lineTailCore = 0;
        iterTailCore = 0;
        lineTailIter = 0;
        lineLastTailIter = 0;
        return;
    }
    lineTailCore = totalLine / aivNum;
    headCoreNum = totalLine % aivNum;
    lineHeadCore = lineTailCore + 1;

    iterHeadCore = FbsCeilDiv(lineHeadCore, rowsPerIter);
    lineHeadIter = rowsPerIter;
    lineLastHeadIter = lineHeadCore - (iterHeadCore - 1) * lineHeadIter;

    iterTailCore = FbsCeilDiv(lineTailCore, rowsPerIter);
    lineTailIter = rowsPerIter;
    lineLastTailIter = lineTailCore - (iterTailCore - 1) * lineTailIter;
}

/**
 * FusedBiasSoftmax 切分主入口（平台无关）。
 *
 * @param batch/channel/height/width x 的 4D 形状
 * @param dtypeSize      元素字节数（4/2）
 * @param dtypeKey       0=fp32, 1=fp16, 2=bf16
 * @param hasB1/hasB2    bias 是否存在
 * @param b1Full         bias1 为全尺寸（非广播）
 * @param ubSize         AIV UB 字节数
 * @param aivNum         AIV 核数
 * @param params         切分结果
 * @param plan           blockDim / tilingKey
 * @return false = 参数或 UB 预算不满足，host 侧返回 tiling 失败
 */
inline bool ComputeFusedBiasSoftmaxTilingCore(uint64_t batch, uint64_t channel, uint64_t height, uint64_t width,
                                              uint64_t dtypeSize, uint64_t dtypeKey, uint64_t hasB1, uint64_t hasB2,
                                              uint64_t b1Full, uint64_t ubSize, uint64_t aivNum,
                                              FusedBiasSoftmaxCoreParams& params, FusedBiasSoftmaxPlan& plan)
{
    const bool isFp32 = (dtypeKey == FBS_KEY_FP32);
    if (width == 0 || width > 4096 || height == 0 || dtypeSize == 0 || aivNum == 0) {
        return false;
    }
    if (hasB1 > 1 || hasB2 > 1 || b1Full > 1) {
        return false;
    }

    ComputePadding(width, dtypeSize, params.paddingNum, params.padLineNum);

    const uint64_t maxByteLine = ComputeMaxByteLine(params.padLineNum, dtypeSize, isFp32, hasB1, hasB2, b1Full == 1);
    if (maxByteLine == 0 || ubSize <= FBS_SOFTMAX_BUF_SIZE) {
        return false;
    }
    const uint64_t availableLinePerIter = (ubSize - FBS_SOFTMAX_BUF_SIZE) / maxByteLine;
    if (availableLinePerIter == 0) {
        return false;
    }

    const uint64_t totalLine = batch * channel * height;
    if (totalLine == 0) {
        return false;
    }
    ComputeRowSplit(totalLine, aivNum, availableLinePerIter, params.headCoreNum, params.lineHeadCore,
                    params.iterHeadCore, params.lineHeadIter, params.lineLastHeadIter, params.lineTailCore,
                    params.iterTailCore, params.lineTailIter, params.lineLastTailIter);

    params.coreNum = aivNum;
    plan.blockDim = aivNum;
    plan.tilingKey = dtypeKey;
    return true;
}

} // namespace FusedBiasSoftmaxHost

#endif // FUSED_BIAS_SOFTMAX_TILING_CORE_H
