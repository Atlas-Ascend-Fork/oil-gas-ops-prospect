---
name: triton-npu-repair-guide
description: >-
  Minimal repairs for Triton-Ascend compile, JIT, UB/L0C overflow, and
  numerical mismatch versus torch.fft. Use when real_fft or other Triton
  kernels fail to compile or fail official tests during optimize.
---

# Repair（Triton-Ascend / 910B）

改编自 Helix `triton-npu-repair-guide`。只保留本仓会碰到的条目。

1. 打开 [references/repair-experience.md](references/repair-experience.md) 对症状。
2. 做最小改动。
3. 按 [../triton-npu-optimize/references/eval.md](../triton-npu-optimize/references/eval.md) 重跑。
4. 不要用纯 PyTorch 替换 NPU kernel 来「修复」。
