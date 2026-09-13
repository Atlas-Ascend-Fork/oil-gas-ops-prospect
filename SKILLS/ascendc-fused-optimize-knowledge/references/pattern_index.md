# Pattern 索引

| id | 何时 |
|---|---|
| [double-buffer-softmax](patterns/double-buffer-softmax.md) | 前向 `iterHead/Tail > 1`，MTE2 与 SoftMax 串行 |
| [grid-match-aiv](patterns/grid-match-aiv.md) | `totalLine < 48` 仍满核空转 |
| [scalar-small-case](patterns/scalar-small-case.md) | 单次 <0.2ms，加速比在 0.7x～1.1x 晃 |
| [keep-mix-bf16](patterns/keep-mix-bf16.md) | 想把 bf16 Grad 改回 VEC，或动 `SyncAll` / padK |
| [host-matchargs-off](patterns/host-matchargs-off.md) | 二次同 shape 空跑，或想「打开 MatchArgs 加速」 |

症状：

- `[8,8,256,256]` 前向只有 ~1.3x，且每核多拍 → double-buffer
- `totalLine=41` 却 `blockDim=48` → grid-match-aiv（已做）
- 小 fp32 0.79x → scalar-small-case，不要为此重写 SoftMax
- Grad 对齐 bf16 变慢且精度怪 → keep-mix-bf16
