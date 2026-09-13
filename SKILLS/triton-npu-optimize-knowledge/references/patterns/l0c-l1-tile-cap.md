# L0C / L1 Tile Cap

## Summary

Keep using `_cap_fft_tiles` then `_launch_with_block_m_backoff` with a shape key. Do not raise default `BLOCK_M=512` without re-checking L0C (`num_acc * M * N * 4 ≤ 128KiB`) and L1 (`K*N` 与 65% 安全系数)。

## Use When

- 新 n / freq_len 触发 overflow。
- 想加大 tile 抢大形态，但不确定 on-chip。

## Avoid When

- 用加大 tile「修复」小形态 0.75x。小形态先看 grid / host。

## What To Verify

- overflow 后同一 `cache_key` 不再重复失败编译。
- 数值与 BLOCK_M 无关（只改切分）。
