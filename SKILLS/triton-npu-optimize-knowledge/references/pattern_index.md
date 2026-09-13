# Pattern Index

先看高优先级，再读对应卡。

## High Priority

### `grid-match-work`

- 逻辑 block 数远小于 AICore 数时，不要按满核 launch。
- 卡：[patterns/grid-match-work.md](patterns/grid-match-work.md)
- 来源：Helix `grid-flatten-and-ub-buffering`（取其「核数对齐工作量」；满核空转在小 FFT 上更常见）

### `host-staging-cache`

- 每 call 重复 `transpose().contiguous()`、重复造基、多余 `movedim` 会吃掉 0.3 ms 级形态。
- 卡：[patterns/host-staging-cache.md](patterns/host-staging-cache.md)
- 来源：Helix `algebraic-optimization`（少遍、可缓存的等价变换）

### `l0c-l1-tile-cap`

- 910B L0C 128KB、L1 ≈512KB。先 cap K/N 再 cap M；overflow 减半 BLOCK_M 并缓存成功值。
- 卡：[patterns/l0c-l1-tile-cap.md](patterns/l0c-l1-tile-cap.md)
- 来源：Helix `autotune` + 仓内已有 `_cap_fft_tiles` / `_launch_with_block_m_backoff`

## 其他

### `keep-axis-kernel`

- 非末轴必须走 transpose-free axis kernel，禁止为了「统一代码」改回 `movedim` 末轴路径。
- 卡：[patterns/keep-axis-kernel.md](patterns/keep-axis-kernel.md)

### `autotune-last`

- 结构已对、只剩 `BLOCK_*` 时再考虑 `triton.autotune`；910B 上优先公式 + 已有 backoff。
- 卡：[patterns/autotune-last.md](patterns/autotune-last.md)
- 来源：Helix `autotune`

### `gemm-vs-fft-flops`

- 末轴小 n 的稠密 DFT 是 O(n·freq)，原生 FFT 是 O(n log n)。切分再抠也补不回数量级；优先减 host/grid，不要承诺小 1D 必赢 `torch.fft`。
- 卡：[patterns/gemm-vs-fft-flops.md](patterns/gemm-vs-fft-flops.md)
