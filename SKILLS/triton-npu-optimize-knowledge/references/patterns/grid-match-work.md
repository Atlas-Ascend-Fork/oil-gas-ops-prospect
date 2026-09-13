# Grid Match Work

## Summary

Launch `min(num_aicore, num_blocks)` programs. The kernel already loops `range(pid, num_blocks, num_programs)`.

## Use When

- `num_blocks` is 1 或个位数，但仍 `kernel[(20,)]`。
- 小末轴 FFT（`[8,256]`、`[4,64,64]`）相对 `torch.fft` 慢在 launch。

## Avoid When

- `num_blocks` 已大于核数（大 5D、`128³` 末轴 rows=16k）。此时 `min` 结果仍是满核，改不动。

## What To Verify

- 大形态加速比不下降。
- `tl.num_programs(0)` 与 host grid 一致；不要另传错误的 `num_core` constexpr。
