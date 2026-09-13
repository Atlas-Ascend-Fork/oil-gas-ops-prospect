---
name: triton-npu-optimize
description: >-
  Iteratively optimize a Triton-Ascend operator in this repo with correctness
  and performance gates against same-device torch.fft or the official harness.
  Use when optimizing real_rfft / real_irfft, Triton NPU kernels, tiling,
  grid, UB/L0C, or when the user mentions Helix-style operator optimize.
---

# Optimize（Triton-Ascend / 本仓库）

改编自 Helix `triton-npu-optimize`。去掉 OpenCode、MCP、`helix` CLI、TileLang、A5。
优化对象必须是带 `@triton.jit` 的仓内算子；默认是 `triton-ascend/real_fft/triton_fft_real.py`。

开始前先读：

- 硬规则：[references/hard-rules.md](references/hard-rules.md)
- 评测命令：[references/eval.md](references/eval.md)
- 产物约定：[references/artifacts.md](references/artifacts.md)
- 方向路由：[../triton-npu-optimize-knowledge/SKILL.md](../triton-npu-optimize-knowledge/SKILL.md)
- 编译失败：[../triton-npu-repair-guide/SKILL.md](../triton-npu-repair-guide/SKILL.md)

## 目标模式

- `kernel`：改 Triton 核 / launch / 切分。以 kernel 时延为主，但结论必须报相对同设备标杆的加速比。
- `operator`：整算子时延（包装、搬数、基缓存、grid 都算）。`real_fft` 默认用这个。

标杆：同设备 `torch.fft.rfft` / `irfft`，`norm="ortho"`。加速比 = 标杆 ms / 本算子 ms。

## 核心循环

1. 建立或复用 baseline（精度 + 性能表）。
2. 开一轮：只选一个主方向（见 knowledge 的 pattern_index）。
3. 做一次连贯改码。
4. 先官方正确性，再同一套 bench。
5. 相对 baseline：**常规末轴不得无说明地变慢**；业务 5D 体（`[1,35,128,128,128]`）不得掉到接近 1x。
6. 记一轮结论，再决定继续或停。

升级顺序（Helix 原序，去掉 compiler-source）：

`pattern triage → 官方/自建 bench → 必要时 msprof`

不要一上来改算法（DFT-GEMM → 拆 Cooley–Tukey 或回退 `torch.fft`）。

## Stage 0：Baseline

对 `real_fft`：

1. CPU：`python3 -m pytest tests/triton/real_fft/test_triton_fft_real.py -q`
2. NPU 精度：同一文件；大 shape 用官方 `verify_triton_fft_real_128_3d.py` 的 `--bench-only` / 默认三路对拍。
3. 性能：`python3 /root/work/tmp/oil-gas-qa/bench_fft.py`（或 [eval.md](references/eval.md) 里的等价脚本）。

写入 `extra-info/fft-optimize/baseline.md`（不提交）。至少包含：

- 常规末轴 / 非末轴 / 业务 5D 三组加速比
- 精度是否 ALL PASS
- 当前 `OIL_GAS_OPS_FFT_BLOCK_M` 与默认 512

## Stage 1：选方向

读 [pattern_index.md](../triton-npu-optimize-knowledge/references/pattern_index.md)，只深读 1～2 张卡。

当前 FFT 已测事实（2026-09-13 NPU）：

| 形态 | 相对 torch.fft | 优先方向 |
|---|---|---|
| 常规末轴 rfft/irfft（0.3～0.7 ms） | 约 0.75x～1.1x | `grid-match-work`、`host-staging-cache` |
| 常规非末轴 irfft | 约 1.3x | 不要为末轴优化而改坏 axis kernel |
| `128³` 末轴 | rfft ~0.80x，irfft ~0.75x | tiling / 少 transpose；注意 GEMM 是 O(n²) vs FFT O(n log n) |
| `[1,35,128,128,128]` 单轴 | 3.7x～10x，反向更高 | **回归门禁**，不是本轮主攻 |

## Stage 2：改码

一次只做一个主题，例如：

- 启动 grid 与物理核数、逻辑 block 数对齐
- 缓存 Fourier 基的转置，避免每 call `transpose().contiguous()`
- 末轴已是 last dim 时去掉多余 `movedim` / `contiguous`
- 在 L0C/L1 cap 内调 `BLOCK_M/K/N`（先手工、后才考虑 `triton.autotune`）

改完立刻走 [eval.md](references/eval.md)。编译/JIT 失败走 repair-guide，不要改成纯 PyTorch。

## Stage 3：停手条件

满足任一即停并写 `extra-info/fft-optimize/opt-note.md`：

- 常规末轴 irfft 从 ~0.75x 升到 ≥1.0x，且 5D 业务形态不回退
- 连续两轮相对上一轮 geomean 增益 < 1.05x
- 官方精度失败且无法在不破坏接口的前提下修复

## 不要做

见 [hard-rules.md](references/hard-rules.md)。
