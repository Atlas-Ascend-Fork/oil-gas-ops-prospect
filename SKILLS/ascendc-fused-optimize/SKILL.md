---
name: ascendc-fused-optimize
description: >-
  Iteratively optimize the registered AscendC fused attention operators
  (FusedBiasSoftmax / FusedSoftmaxGrad) against same-device torch softmax
  and matmul+JVP. Use when the user mentions cannbot, AscendC fused, MIX,
  tiling, double buffer, or fused softmax performance.
---

# Optimize（AscendC 融合算子 / 本仓库）

改编自 cannbot-skills：`ascendc-perf-optimize`（4 步分层流水）、`ascendc-performance-best-practices`、`ascendc-tiling-design`、`ops-profiling`、`npu-arch`。

去掉：OpenCode / 插件宿主、从零新建算子、直调模板、Catlass / PyPTO / TileLang / Triton、A5 / DAV_3510 / dav310 MicroAPI 模板、通信类 Step 2。

对象：`ascendc/operators/fused_bias_softmax/**`、`ascendc/operators/fused_softmax_grad/**` 及其 ophost UT、NPU 对比测试。这是 **ACLNN 注册调用**，不是 `<<<>>>` 直调。

开始前先读：

- 硬规则：[references/hard-rules.md](references/hard-rules.md)
- 评测：[references/eval.md](references/eval.md)
- 产物：[references/artifacts.md](references/artifacts.md)
- 方向：[../ascendc-fused-optimize-knowledge/SKILL.md](../ascendc-fused-optimize-knowledge/SKILL.md)
- 修复：[../ascendc-fused-repair-guide/SKILL.md](../ascendc-fused-repair-guide/SKILL.md)

## cannbot 4 步在本仓怎么走

| 步骤 | cannbot 名 | 本仓 |
|---|---|---|
| Step 1 | Tiling 理论建模 | 必做。Softmax 前向 = Reduction/AR-FullLoad；Grad MIX = Cube + Vec JVP |
| Step 2 | 卡间流水 | **跳过**（单卡、无集合通信） |
| Step 3 | 核间流水 | MIX 有 `SyncAll`（pad → Cube → JVP）。只在改 MIX 时做 |
| Step 4 | 单核流水 / bound | 必做。小 shape → scalar；多 iter 前向 → memory/double-buffer |

标杆：同设备 Torch 小算子。加速比 = 标杆 ms / 本算子 ms。

- 前向：`softmax(x + bias1 + bias2, -1)`（fp32 累加）
- 反向：`g = go @ V^T` + softmax JVP

## 核心循环

1. 建立或复用 baseline（精度 + 相对 Torch 倍数表）。
2. 开一轮：只选一个主方向（knowledge `pattern_index`）。
3. 做一次连贯改码（tiling core + kernel + ophost UT 一起）。
4. 先无卡 ophost UT，再官方 NPU 精度，再同一套 `bench_ops.py`。
5. 相对同设备 Torch 小算子 **不得 <1.0x**；bf16 大形态与 MIX 反向不得无说明回退。
6. 记一轮结论，再决定继续或停。

不要一上来把融合核拆回 `torch.softmax` / `bmm`，也不要换成 Triton。

## Stage 0：Baseline

2026-09-13 同设备复测（预热 5 / 计时 20）：

| 算子 | 代表倍数 |
|---|---|
| FusedBiasSoftmax bf16 典型 `[2,4,64,128]` / `[2,4,256,256]` | 2.89x / 2.67x |
| FusedBiasSoftmax bf16 `[2,4,64,64]` | 2.83x |
| FusedBiasSoftmax bf16 大形态 `[8,8,256,256]` | 2.68x |
| FusedBiasSoftmax fp32 小形态 `[2,4,64,64]` | 1.66x |
| FusedSoftmaxGrad bf16 MIX | 1.9x～2.1x |
| FusedSoftmaxGrad fp32 VEC | 1.14x |

写入 `extra-info/fused-optimize/baseline.md`（不提交）。

## Stage 1：选方向

读 [pattern_index.md](../ascendc-fused-optimize-knowledge/references/pattern_index.md)，只深读 1～2 张卡。

| 形态 | 相对 Torch | 优先方向 |
|---|---|---|
| 前向多 iter（`[8,8,256,256]` 约 5+ 拍） | ~2.7x | 已双缓冲；不要再切碎 tile |
| 前向小 fp32 / 小 bf16 | ≥1.6x | `scalar-small-case`：少核 + pybind 复用 executor |
| 前向已 1 iter 的典型 bf16 | ~2.7x | 不要为双缓冲把 1 拍形态切碎到变慢 |
| Grad bf16 MIX | ~2x | `keep-mix-bf16`；核间只动有依据的 `SyncAll` |
| Grad fp32 | ~1.12x | 保持 VEC；不要上 Cube（fixpipe 16bit） |

## Stage 2：改码

一次一个主题。改 tiling 公式必须同步改 `tests/ascendc/ophost/` 手算期望。

改完立刻走 [eval.md](references/eval.md)。精度/运行时失败走 repair-guide。

## Stage 3：停手

满足任一即停并写 `extra-info/fused-optimize/opt-note.md`：

- `[8,8,256,256]` bf16 前向从 ~1.29x 升到 ≥1.5x，且典型 bf16 不回退
- 连续两轮 geomean 增益 < 1.05x
- 官方精度失败且无法在不破坏原地语义的前提下修复

## 不要做

见 [hard-rules.md](references/hard-rules.md)。
