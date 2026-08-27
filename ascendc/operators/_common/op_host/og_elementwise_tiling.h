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
 * 逐元素算子共用的切分算法（平台无关，header-only，只依赖 <cstdint>）。
 *
 * complex_mul / complex_mul_grad 这类"全部输入输出同 shape、单路径 AIV"的算子，
 * 切分只有两件事：单拍元素数（吃满 UB）与核数。两个算子唯一的差别是 UB 里同时
 * 存活的 tile 份数（UB_SLOT_FACTOR），因此把算法抽到这里，各算子只声明自己的
 * factor。这样也避免了 host 侧切分代码在每个算子里复制一遍。
 *
 * 本文件由 scripts/merge_operators.py 从 operators/_common/op_host/ 平铺拷贝到
 * msopgen 工程的 op_host/ 下，与各算子的 *_tiling_core.h 同目录，故用无路径 include。
 * ophost UT 通过 tests/ophost/CMakeLists.txt 的 include 目录直接引用源码位置。
 */

#ifndef OG_ELEMENTWISE_TILING_H
#define OG_ELEMENTWISE_TILING_H

#include <cstdint>

namespace OgOpsCommon {

/** 32B / sizeof(float)：DataCopy 的对齐粒度。 */
constexpr int64_t ALIGN_FLOATS = 8;
/** 单核可用 UB 预算（910B 的 UB 为 192KB，留出 tiling/系统开销）。 */
constexpr int64_t UB_BUDGET_BYTES = 176 * 1024;

/**
 * 向下对齐到 kAlign 的倍数。
 *
 * 对齐量走模板参数而不是函数入参：除数在编译期就是正的字面常量，既省掉运行期
 * 判零分支，也让静态检查无需做路径分析即可确定不会除零。
 */
template <int64_t kAlign>
inline int64_t AlignDown(int64_t v)
{
    static_assert(kAlign > 0, "kAlign must be positive");
    return (v / kAlign) * kAlign;
}

/**
 * 单拍上限由各算子在自己的 core 头里算：
 *   TILE_FLOATS_CAP = UB_BUDGET_BYTES / (UB_SLOT_FACTOR * sizeof(float)) 向下对齐到 ALIGN_FLOATS
 * 不抽成 constexpr 函数：msopgen 的 opbuild 用 C++11 编译 host 源，
 * constexpr 函数体只能是单条 return，写成多分支会编译失败。
 */

/** 单拍 float 数：默认吃满 UB 上限；小 tensor 收缩到刚好覆盖。 */
inline int64_t PickTileFloats(int64_t totalElems, int64_t tileFloatsCap)
{
    const int64_t cap = tileFloatsCap < ALIGN_FLOATS ? ALIGN_FLOATS : tileFloatsCap;
    if (totalElems <= 0 || totalElems >= cap) {
        return cap;
    }
    const int64_t wanted = totalElems > ALIGN_FLOATS ? totalElems : ALIGN_FLOATS;
    const int64_t shrunk = AlignDown<ALIGN_FLOATS>(wanted);
    return shrunk < ALIGN_FLOATS ? ALIGN_FLOATS : shrunk;
}

/**
 * 平台无关的切分核心。aivNum / libApiWorkspace 由调用方从 platform 读出后传入，
 * 便于 ophost UT 直接构造。
 *
 * TilingT 需有 totalElems / tileFloats / coreNum 字段；
 * PlanT 需有 tilingKey / blockDim / totalWorkspace 字段。
 */
template <typename TilingT, typename PlanT>
inline bool ComputeElementwiseTilingCore(int64_t totalElems, uint32_t aivNum, uint64_t libApiWorkspace,
                                         int64_t tileFloatsCap, uint64_t tilingKey, TilingT &tiling, PlanT &plan)
{
    if (totalElems < 0) {
        return false;
    }
    const uint32_t coreNum = (aivNum == 0) ? 1U : aivNum;

    tiling.totalElems = totalElems;
    tiling.tileFloats = PickTileFloats(totalElems, tileFloatsCap);
    tiling.coreNum = static_cast<int64_t>(coreNum);

    plan.tilingKey = tilingKey;
    plan.blockDim = coreNum;
    plan.totalWorkspace = libApiWorkspace;
    return true;
}

} // namespace OgOpsCommon

#endif // OG_ELEMENTWISE_TILING_H
