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
 * FusedSoftmaxGradOg 切分算法核心（不依赖 platform_ascendc / gert / CANN 头）。
 * ophost UT（ascendc/tests/ophost/）直接驱动本文件的纯函数。
 *
 * 路径选择（tiling key，与 kernel 侧 TILING_KEY_IS 一致）：
 *   0: fp32 → 纯向量（2201 fixpipe C 16bit 结构上限：g 以 half 落盘的量化误差
 *      ~4.9e-4 必超 fp32 契约 rtol=1e-4，hi/lo split 多矩乘同样过不了 16bit 落盘）
 *   1: fp16 → 纯向量（fp16 契约 rtol=5e-4 与 half-g 量化误差 4.9e-4 过近）
 *   2: bf16 且 (S%16!=0 或 D%16!=0) → 纯向量（cube ND 对齐要求）
 *   3: bf16 且对齐 → MIX_AIC_1_2（AIC 矩乘 + AIV JVP）
 */

#ifndef FUSED_SOFTMAX_GRAD_TILING_CORE_H
#define FUSED_SOFTMAX_GRAD_TILING_CORE_H

#include <cstdint>

namespace FusedSoftmaxGradHost {

constexpr uint64_t FSG_KEY_FP32_VEC = 0;
constexpr uint64_t FSG_KEY_FP16_VEC = 1;
constexpr uint64_t FSG_KEY_BF16_VEC = 2;
constexpr uint64_t FSG_KEY_BF16_MIX = 3;

constexpr uint64_t FSG_FP32_ALIGN = 8;          // fp32 32B 对齐元素数
constexpr uint64_t FSG_HALF_ALIGN = 16;         // 16bit 32B 对齐元素数
constexpr uint64_t FSG_UB_RESERVE_VEC = 49152;  // 向量路径余量：行缓冲 + Sum tmp + slack
constexpr uint64_t FSG_UB_RESERVE_MIX = 65536;  // MIX 路径 AIV 余量：Sum tmp + slack
constexpr uint64_t FSG_MIX_ROW_BYTES_ELEM = 16; // MIX 每行每元素 UB 字节：p16(2)+g16(2)+p32(4)+g32(4)+t32(4)
constexpr uint64_t FSG_HALF_SIZE = 2;

/** 切分结果（plain POD，UT 可直接断言）。 */
struct FusedSoftmaxGradCoreParams {
    uint64_t useMix = 0;
    uint64_t padS = 0;
    uint64_t padD = 0;
    uint64_t rowsPerIter = 0;
};

/** blockDim / tilingKey / MIX 路径 g 缓冲字节数。 */
struct FusedSoftmaxGradPlan {
    uint64_t blockDim = 0;
    uint64_t tilingKey = 0;
    uint64_t gBytes = 0;
};

inline uint64_t FsgAlignUp(uint64_t a, uint64_t base)
{
    return (base == 0) ? 0 : (a + base - 1) / base * base;
}

/**
 * FusedSoftmaxGradOg 切分主入口（平台无关）。
 *
 * @param q/s 方阵假设解除后的两维（Q≠S 合法）
 * @param dLen gradOutput/values 末维
 * @param dtypeKey 0=fp32, 1=fp16, 2=bf16
 * @param ubSize AIV UB 字节数
 * @return false = 参数或 UB 预算不满足
 */
inline bool ComputeFusedSoftmaxGradTilingCore(uint64_t batch, uint64_t channel, uint64_t q, uint64_t s, uint64_t dLen,
    uint64_t dtypeKey, uint64_t ubSize, FusedSoftmaxGradCoreParams &params, FusedSoftmaxGradPlan &plan)
{
    if (q == 0 || s == 0 || dLen == 0 || batch == 0 || channel == 0 || ubSize == 0) {
        return false;
    }

    const bool isFp32 = (dtypeKey == FSG_KEY_FP32_VEC);
    const uint64_t align = isFp32 ? FSG_FP32_ALIGN : FSG_HALF_ALIGN;
    params.padS = FsgAlignUp(s, align);
    params.padD = FsgAlignUp(dLen, align);

    // MIX 资格：bf16 且 S/D 均 16 对齐（openfold 生产形态）
    params.useMix = (dtypeKey == FSG_KEY_BF16_VEC && dLen % FSG_HALF_ALIGN == 0 && s % FSG_HALF_ALIGN == 0) ? 1 : 0;

    if (params.useMix == 0) {
        // 向量路径：V plane fp32 驻留 UB（S*padD*4）+ 行缓冲/Sum tmp 余量
        const uint64_t vBytes = s * params.padD * 4;
        if (vBytes + FSG_UB_RESERVE_VEC > ubSize) {
            return false;
        }
        plan.tilingKey = dtypeKey; // 0/1/2 均为向量 key
        plan.gBytes = 0;
    } else {
        // MIX 路径：rowsPerIter 按 UB 预算（p16+g16+p32+g32+t32 = 16B/元素）
        const uint64_t avail = ubSize > FSG_UB_RESERVE_MIX ? ubSize - FSG_UB_RESERVE_MIX : 0;
        const uint64_t rowBytes = FSG_MIX_ROW_BYTES_ELEM * params.padS;
        if (rowBytes == 0 || avail < rowBytes) {
            return false;
        }
        params.rowsPerIter = avail / rowBytes;
        plan.tilingKey = FSG_KEY_BF16_MIX;
        plan.gBytes = batch * channel * q * s * FSG_HALF_SIZE;
    }
    return true;
}

} // namespace FusedSoftmaxGradHost

#endif // FUSED_SOFTMAX_GRAD_TILING_CORE_H
