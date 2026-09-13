# Autotune Last

## Summary

Helix 默认对自由 `BLOCK_*` 走 `triton.autotune`。本仓库 910B FFT 已有公式 cap + 失败减半缓存；先穷尽这两层。真要 autotune：`key` 必须含 `n` / `rows` / `freq_len`，搜索面是 `BLOCK_*` 不是 GPU 的 `num_warps`。

## Use When

- 结构稳定，且手工扫 128/256/512 仍有稳定差距。
- 编译次数可接受（autotune 会成倍拉长首 call）。

## Avoid When

- 小形态还在吃 launch / host。
- 没有 reset 语义的原地缓冲（本算子输出是新分配，这项较松）。
