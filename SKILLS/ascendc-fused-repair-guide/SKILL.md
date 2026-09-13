---
name: ascendc-fused-repair-guide
description: >-
  Minimal repair for FusedBiasSoftmax / FusedSoftmaxGrad compile, tiling,
  NaN, or inplace no-op failures on 910B. Use when official fused tests
  fail after an optimize round.
---

# Repair（融合算子）

改编自 cannbot `ascendc-precision-debug` / `ascendc-runtime-debug`。只修当前回归，不借机改算法。

## 先分类

| 症状 | 先看 |
|---|---|
| ophost UT 期望对不上 | 手算 `ComputeMaxByteLine` / `usedAiv`，改测试或改公式，二选一要自洽 |
| 编译 / 出包失败 | 全量 `build.sh --pkg`，看 tiling GET_TILING_DATA 字段顺序 |
| 二次同 shape 输出不变 | MatchArgs 是否又被打开 |
| bf16 方形 Grad NaN | `padD+=16` 是否丢了；有没有改成垫 N |
| 未对齐 Grad 错 / 读越界 | workspace 是否含 A/B pad；Matmul 用 `padS/padD` |
| 精度偶发、确定性差 | `SoftMax` 共享 32K、Cast 舍入；先固定 seed 的官方用例 |

## 最小修复

1. 固定 1 个官方 shape + dtype。
2. 只改一处（tiling、kernel 同步或 pad）。
3. 重跑该用例 + 全量 fused NPU。
4. 不要用 Torch 拆解换 kernel 来「修绿」。
