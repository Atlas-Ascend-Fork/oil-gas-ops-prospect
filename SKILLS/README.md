# SKILLS

本仓库可复用的算子开发技能，按后端拆开，每个后端配套优化（`*-optimize`）、症状定位（`*-optimize-knowledge`）与修复（`*-repair-guide`）三类。都不收录 OpenCode / MCP 宿主、从零新建算子、A5 / DAV_3510、TileLang / Catlass / PyPTO。

## Triton-Ascend（Helix）

只保留 Triton + Atlas A2 / 910B。对象：`triton-ascend/real_fft/`。

| 技能 | 何时用 |
|---|---|
| [triton-npu-optimize](triton-npu-optimize/SKILL.md) | 优化 `real_rfft` / `real_irfft` |
| [triton-npu-optimize-knowledge](triton-npu-optimize-knowledge/SKILL.md) | FFT 症状 → pattern |
| [triton-npu-repair-guide](triton-npu-repair-guide/SKILL.md) | Triton 编译 / JIT / 精度失败 |

## AscendC 融合算子（cannbot）

改编自 [cannbot-skills](https://gitcode.com/cann/cannbot-skills) 的 `ascendc-perf-optimize` / `ascendc-performance-best-practices` / `ascendc-tiling-design` / `ops-profiling` / `npu-arch`。对象：注册调用的 `FusedBiasSoftmax` / `FusedSoftmaxGrad`。

| 技能 | 何时用 |
|---|---|
| [ascendc-fused-optimize](ascendc-fused-optimize/SKILL.md) | 优化融合注意力，相对同设备 Torch 小算子闭环 |
| [ascendc-fused-optimize-knowledge](ascendc-fused-optimize-knowledge/SKILL.md) | 融合算子症状 → pattern |
| [ascendc-fused-repair-guide](ascendc-fused-repair-guide/SKILL.md) | 切分 / MIX NaN / 原地空跑的最小修复 |

未收录：`ops-direct-invoke` 新建流、卡间 Step 2、dav310 Softmax 模板、通信 / MC2。

优化轮次笔记写 `extra-info/`（不提交）。Triton 改动落 `triton-ascend/`；融合核改动落 `ascendc/operators/fused_*`。
