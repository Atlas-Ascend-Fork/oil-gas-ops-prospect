# Symptom Index

有 bench / msprof 再读。只选一张对得上的。

## `high-scalar-overhead`

- 单次 0.3～0.7 ms、加速比在 0.75x～1.1x 晃，核时延不是主因。
- 方向：`grid-match-work`、`host-staging-cache`

## `high-transfer-pressure`

- 末轴路径相对标杆多几十 MB，或每 call 出现 `transpose().contiguous()`。
- 方向：`host-staging-cache`、`keep-axis-kernel`

## `onchip-overflow`

- 日志含 `cbuf overflow` / `cc overflow` / `bisheng` / `mlircompilation`。
- 方向：`l0c-l1-tile-cap`；见 repair-guide

## `large-volume-already-fast`

- `[1,35,128,128,128]` 已是数倍到十余倍。当回归门禁，不当本轮主攻。
- 方向：不要为小形态引入会伤大 grid 的 `num_programs` 常数写死
