---
name: triton-npu-optimize-knowledge
description: >-
  Pattern and symptom cards for Triton-Ascend optimize on Atlas A2 / 910B.
  Use during triton-npu-optimize pattern triage for real_fft tiling, grid,
  host staging, L0C/L1 overflow, or vs torch.fft speedups.
---

# Optimize Knowledge（910B / 本仓库）

改编自 Helix `triton-npu-optimize-knowledge` v3。只留对当前 FFT-GEMM 有用的卡；A5 / CANN ext / TileLang 不收录。

本技能不拥有评测流程。改码与验收归 `triton-npu-optimize`。

## 阅读顺序

1. [references/pattern_index.md](references/pattern_index.md)
2. 有 profile 再看 [references/symptom_index.md](references/symptom_index.md)
3. 只深读 1～2 张 pattern 卡

## 推理

卡是路由，不是规则引擎。选完回到 optimize skill 做一轮改码。
