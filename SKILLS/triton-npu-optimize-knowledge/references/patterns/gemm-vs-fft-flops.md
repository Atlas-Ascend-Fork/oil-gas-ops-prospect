# GEMM vs FFT Flops

## Summary

Last-axis kernel is dense DFT：每行 `n * (n//2+1)` 乘加。`torch.fft` 是 Cooley–Tukey。n=128 时 GEMM 大约多一个数量级 flop。大 batch / 5D 时 Cube + 少搬数仍能赢；小 1D 不要承诺必超原生 FFT。

## Use When

- `128³` 末轴仍 ~0.8x，有人想「再改 tiling 就能到 2x」。
- 评估是否该换算法（本 skill 包 **禁止** 换成 `torch.fft` 回退）。

## Avoid When

- 5D 已经 3x+：那是 host/搬数，不是 flop 叙事。
