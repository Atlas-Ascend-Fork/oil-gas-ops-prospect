# Host Staging Cache

## Summary

Cache Fourier basis **and** the `(freq_len, n)` transpose used by last-axis irfft / rfft backward. Skip `movedim` + extra `contiguous` when `dim` is already last.

## Use When

- 每 call 有 `basis.transpose(0, 1).contiguous()`。
- 末轴输入已是 last dim 仍 `movedim(dim, -1).contiguous()`。
- 小 shape irfft 峰值显存比 `torch.fft` 高数 MB。

## Avoid When

- 改 `make_basis` 公开返回值（必须仍是 4 个 `(n, freq_len)` 矩阵）。

## What To Verify

- `tests/triton/real_fft` 里 `make_basis` 解包仍为 4 元组。
- 往返与 `torch.fft` 相对误差不升。
- 5D 非末轴仍走 axis kernel，不经过这套 transpose 基。
