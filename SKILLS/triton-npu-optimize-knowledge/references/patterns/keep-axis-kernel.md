# Keep Axis Kernel

## Summary

When FFT axis is not last, keep the `(A, n, B)` / `(A, freq, B)` kernels. They avoid two full-tensor transposes and already beat `torch.fft` on 5D H/D.

## Use When

- 有人想把非末轴 unify 成 `movedim` + 末轴 GEMM。
- 优化末轴时顺手改了 axis launch / tile cap。

## Avoid When

- 正在修 axis 自己的 overflow（仍只改 cap，不改回 transpose）。

## What To Verify

- `[1,35,128,128,128]` dim=-2/-3 加速比不回退。
- `dim != ndim-1` 分支没有 `movedim` 到 last。
