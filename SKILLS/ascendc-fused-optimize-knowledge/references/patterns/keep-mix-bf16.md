# keep-mix-bf16

来源：本仓实证 + cannbot Step 3（有核间同步才分析）。

## 事实

- 全部 bf16 Grad 走 MIX。未对齐先 pad 再 Cube。
- `Q == padS == padD` 时 `padD += 16`，避开 CFG_NORM `IterateAll` 写 NaN。垫 N 仍会错。
- AIC 做完所有 plane 的 `IterateAll` 后 **一次** `SyncAll`，AIV 再 JVP。不要改成每 plane 双 Sync。
- fp32/fp16 保持 VEC（fixpipe C 16bit）。

## 不要

把 bf16 降 VEC、打开 MatchArgs、或抄 dav310 Softmax 模板替换 JVP。
