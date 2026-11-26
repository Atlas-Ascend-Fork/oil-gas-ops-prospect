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
 * 索引/散射类算子的公共切分算法（平台无关，header-only，只依赖 <cstdint> / <limits>）。
 *
 * ROIAlignRotated(Grad) / SwapAlign2Nat(Grad) 的反向都属于“写索引型（Scatter 系）”。
 * 910B3 实测（见 SKILLS 对应的 ascendc-tiling-design/references/index/scatter.md）：
 *   - 按 index 元素空间分核会丢数据；
 *   - 标量写 GM 慢，且两个核并发写同一 64B cache line 的不同 32B sector 会丢 sector；
 *   - 正确做法是按“输出元素空间”按 64B 对齐整块分核，每个核独占连续整块，
 *     核内先在 UB 累加，再整片回写。这样同一输出元素单属主，累加无需原子且确定。
 *
 * 本文件把上述分核算法抽成与平台无关的 FlatSlabPlan，供各算子 tiling core 与
 * ophost UT 共用。slab 以 float 元素为单位，64B 对齐即 16 个 float。
 *
 * 注意：本文件由 scripts/merge_operators.py 从 operators/_common/op_host/ 平铺
 * 拷贝到 msopgen 工程的 op_host/ 下，故 include 不写路径。
 */

#ifndef OG_INDEX_TILING_H
#define OG_INDEX_TILING_H

#include <cstdint>
#include <limits>

namespace OgOpsCommon {

/** 索引类算子标量散写/矢量回写的 cache line 对齐字节数。 */
constexpr int64_t INDEX_ALIGN_BYTES = 64;
/** 64B / sizeof(float)：一个对齐块包含的 float 元素数。 */
constexpr int64_t INDEX_ALIGN_FLOATS = INDEX_ALIGN_BYTES / static_cast<int64_t>(sizeof(float));
/** 单核 UB 预算（910B 全系 UB 192KB，留出系统/tiling 开销）。 */
constexpr int64_t INDEX_UB_BUDGET_BYTES = 176 * 1024;

/** 向下/向上对齐到 INDEX_ALIGN_FLOATS 的倍数。 */
inline int64_t IndexAlignDown(int64_t v) { return (v / INDEX_ALIGN_FLOATS) * INDEX_ALIGN_FLOATS; }

inline int64_t IndexAlignUp(int64_t v)
{
    return ((v + INDEX_ALIGN_FLOATS - 1) / INDEX_ALIGN_FLOATS) * INDEX_ALIGN_FLOATS;
}

/** 安全乘法：任一参数为负或结果溢出时返回 false。 */
inline bool IndexMulChecked(int64_t a, int64_t b, int64_t& out)
{
    if (a < 0 || b < 0) {
        return false;
    }
    const int64_t kMax = std::numeric_limits<int64_t>::max();
    if (a != 0 && b > kMax / a) {
        return false;
    }
    out = a * b;
    return true;
}

/** 依次安全乘入 out；失败返回 false。 */
inline bool IndexMulInto(int64_t factor, int64_t& out)
{
    int64_t product = 0;
    if (!IndexMulChecked(out, factor, product)) {
        return false;
    }
    out = product;
    return true;
}

/**
 * 输出/输入元素空间的 64B 对齐 slab 切分结果。
 *
 * 语义：
 *   - total == 0 时 blockDim = 1、numSlabs = 0，kernel 直接空跑；
 *   - total > 0 时 slabFloats = min(total, slabFloatsCap 向下对齐)，
 *     每个 slab 起点都是 64B 对齐的；最后一个 slab 可以短于 slabFloats；
 *   - numSlabs 个 slab 连续铺满 [0, total)，按 blockDim 个核分配：
 *     前 remSlabs 个核各多拿 1 个 slab。
 */
struct FlatSlabPlan {
    int64_t total = 0;
    int64_t slabFloats = 0;
    int64_t numSlabs = 0;
    uint32_t blockDim = 0;
    int64_t baseSlabs = 0;
    int64_t remSlabs = 0;
};

/**
 * 计算 slab 切分。
 *
 * @param total           输出（反向时为 grad_input）元素总数，必须 >= 0
 * @param aivNum          PlatformAscendC::GetCoreNumAiv()，0 时兜底为 1
 * @param slabFloatsCap   单 slab 元素上限（由各算子按 UB 预算给出）
 * @param plan            输出切分结果
 * @return false = 参数非法（负数 / 非正 cap）
 */
inline bool ComputeFlatSlabPlan(int64_t total, uint32_t aivNum, int64_t slabFloatsCap, FlatSlabPlan& plan)
{
    if (total < 0 || slabFloatsCap <= 0) {
        return false;
    }

    plan.total = total;
    if (total == 0) {
        plan.slabFloats = 0;
        plan.numSlabs = 0;
        plan.blockDim = 1U;
        plan.baseSlabs = 0;
        plan.remSlabs = 0;
        return true;
    }

    const int64_t alignedCap = IndexAlignDown(slabFloatsCap);
    const int64_t cap = alignedCap < INDEX_ALIGN_FLOATS ? INDEX_ALIGN_FLOATS : alignedCap;
    plan.slabFloats = total < cap ? total : cap;
    plan.numSlabs = (total + plan.slabFloats - 1) / plan.slabFloats;

    uint32_t cores = (aivNum == 0U) ? 1U : aivNum;
    if (static_cast<int64_t>(cores) > plan.numSlabs) {
        cores = static_cast<uint32_t>(plan.numSlabs);
    }
    plan.blockDim = cores;
    plan.baseSlabs = plan.numSlabs / static_cast<int64_t>(cores);
    plan.remSlabs = plan.numSlabs % static_cast<int64_t>(cores);
    return true;
}

/** 第 blockIdx 个核负责的 slab 区间 [startSlab, endSlab)；越界核返回空区间。 */
inline void FlatSlabRangeForCore(const FlatSlabPlan& plan, uint32_t blockIdx, int64_t& startSlab, int64_t& endSlab)
{
    startSlab = 0;
    endSlab = 0;
    if (static_cast<int64_t>(blockIdx) >= static_cast<int64_t>(plan.blockDim)) {
        return;
    }
    const int64_t idx = static_cast<int64_t>(blockIdx);
    const int64_t extra = idx < plan.remSlabs ? idx : plan.remSlabs;
    startSlab = idx * plan.baseSlabs + extra;
    endSlab = startSlab + plan.baseSlabs + (idx < plan.remSlabs ? 1 : 0);
}

/** 第 slabIdx 个 slab 对应的元素区间 [startElem, endElem)。 */
inline void FlatSlabElemRange(const FlatSlabPlan& plan, int64_t slabIdx, int64_t& startElem, int64_t& endElem)
{
    startElem = slabIdx * plan.slabFloats;
    endElem = startElem + plan.slabFloats;
    if (endElem > plan.total) {
        endElem = plan.total;
    }
    if (startElem > plan.total) {
        startElem = plan.total;
        endElem = plan.total;
    }
}

} // namespace OgOpsCommon

#endif // OG_INDEX_TILING_H
