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
 * FusedSoftmaxGrad 切分算法核心（不依赖 platform_ascendc / gert / CANN 头）。
 * ophost UT（tests/ascendc/ophost/）直接驱动本文件的纯函数。
 *
 * 路径选择（tiling key，与 kernel 侧 TILING_KEY_IS 一致）：
 *   0: fp32 → 纯向量（2201 fixpipe C 16bit 结构上限：g 以 half 落盘的量化误差
 *      ~4.9e-4 必超 fp32 契约 rtol=1e-4，hi/lo split 多矩乘同样过不了 16bit 落盘）
 *   1: fp16 → 纯向量（fp16 契约 rtol=5e-4 与 half-g 量化误差 4.9e-4 过近）
 *   2: 保留（旧 VEC key，当前 bf16 不再下发）
 *   3: bf16 → MIX_AIC_1_2（AIC 矩乘 + AIV JVP）；S/D 未 16 对齐时
 *      AIV 先把 go/V pad 到 padS/padD 再 Cube。M=N=K 时再把 K 垫 16，
 *      避免 CFG_NORM IterateAll 写 NaN（垫 N 仍会错，垫 K 已用 D80 对过）。
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
constexpr uint64_t FSG_AIV_UB_910B = 196608;    // 910B AIV UB；platform 回报偏小时仍按此决策 VEC/MIX
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

inline uint64_t FsgAlignUp(uint64_t a, uint64_t base) { return (base == 0) ? 0 : (a + base - 1) / base * base; }

inline bool HasEmptyGradDim(uint64_t batch, uint64_t channel, uint64_t q, uint64_t s, uint64_t dLen, uint64_t ubSize)
{
    return q == 0 || s == 0 || dLen == 0 || batch == 0 || channel == 0 || ubSize == 0;
}

inline bool FillVecPlan(uint64_t s, uint64_t padD, uint64_t aivUb, uint64_t dtypeKey, FusedSoftmaxGradPlan& plan)
{
    const uint64_t vBytes = s * padD * 4;
    if (vBytes + FSG_UB_RESERVE_VEC > aivUb) {
        return false;
    }
    plan.tilingKey = dtypeKey; // 0/1/2 均为向量 key
    plan.gBytes = 0;
    return true;
}

inline bool FillMixPlan(uint64_t batch, uint64_t channel, uint64_t q, uint64_t s, uint64_t dLen, uint64_t padS,
                        uint64_t padD, uint64_t aivUb, FusedSoftmaxGradCoreParams& params, FusedSoftmaxGradPlan& plan)
{
    // MIX 路径：rowsPerIter 按 UB 预算（p16+g16+p32+g32+t32 = 16B/元素）
    const uint64_t avail = aivUb > FSG_UB_RESERVE_MIX ? aivUb - FSG_UB_RESERVE_MIX : 0;
    const uint64_t rowBytes = FSG_MIX_ROW_BYTES_ELEM * padS;
    if (rowBytes == 0 || avail < rowBytes) {
        return false;
    }
    params.rowsPerIter = avail / rowBytes;
    plan.tilingKey = FSG_KEY_BF16_MIX;
    const uint64_t planes = batch * channel;
    plan.gBytes = planes * q * padS * FSG_HALF_SIZE;
    // 未对齐：A [Q,padD] + B [padS,padD]（B 按 padS 行，避免 Cube 读越界）
    if (s != padS || dLen != padD) {
        plan.gBytes += planes * (q * padD + padS * padD) * FSG_HALF_SIZE;
    }
    return true;
}

/**
 * FusedSoftmaxGrad 切分主入口（平台无关）。
 *
 * @param q/s 方阵假设解除后的两维（Q≠S 合法）
 * @param dLen gradOutput/values 末维
 * @param dtypeKey 0=fp32, 1=fp16, 2=bf16
 * @param ubSize AIV UB 字节数
 * @return false = 参数或 UB 预算不满足
 */
inline bool ComputeFusedSoftmaxGradTilingCore(uint64_t batch, uint64_t channel, uint64_t q, uint64_t s, uint64_t dLen,
                                              uint64_t dtypeKey, uint64_t ubSize, FusedSoftmaxGradCoreParams& params,
                                              FusedSoftmaxGradPlan& plan)
{
    if (HasEmptyGradDim(batch, channel, q, s, dLen, ubSize)) {
        return false;
    }

    const bool isFp32 = (dtypeKey == FSG_KEY_FP32_VEC);
    const uint64_t align = isFp32 ? FSG_FP32_ALIGN : FSG_HALF_ALIGN;
    params.padS = FsgAlignUp(s, align);
    params.padD = FsgAlignUp(dLen, align);

    // MIX 资格：全部 bf16。S/D 未对齐时 kernel 先 pad 再 Cube，不再降 VEC 点积。
    params.useMix = (dtypeKey == FSG_KEY_BF16_VEC) ? 1 : 0;
    // 910B CFG_NORM 在 M=N=K 时 IterateAll 写 NaN；把 K 再垫 16（走已验证的 D-pad）。
    if (params.useMix == 1 && q == params.padS && params.padS == params.padD) {
        params.padD += align;
    }

    // platform.GetCoreMemSize 在部分 CANN 上会回报偏小（64KB）；910B 实际 AIV
    // UB 为 192KB，取二者较大值做 VEC/MIX 预算，避免 MIX 因 avail=0 被拒。
    const uint64_t aivUb = ubSize > FSG_AIV_UB_910B ? ubSize : FSG_AIV_UB_910B;
    if (params.useMix == 0) {
        return FillVecPlan(s, params.padD, aivUb, dtypeKey, plan);
    }
    return FillMixPlan(batch, channel, q, s, dLen, params.padS, params.padD, aivUb, params, plan);
}

} // namespace FusedSoftmaxGradHost

#endif // FUSED_SOFTMAX_GRAD_TILING_CORE_H
