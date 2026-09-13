---
name: ascendc-fused-optimize-knowledge
description: >-
  Routes a FusedBiasSoftmax / FusedSoftmaxGrad performance symptom to one
  910B pattern card distilled from cannbot tiling, double-buffer, scalar,
  and MIX sync guidance. Use when choosing the next fused optimize round.
---

# Knowledge（融合算子 / 910B）

先读 [pattern_index.md](references/pattern_index.md)，再只打开 1～2 张卡。

来源（cannbot-skills，不整包拷贝）：

- `ascendc-perf-optimize` Step 1/3/4（跳过卡间）
- `ascendc-performance-best-practices` 的 double buffer / softmax AR-FullLoad
- `npu-arch` DAV_2201：UB 192KB，24 AIC + 48 AIV
- 本仓已证实：bf16 Grad 必须 MIX；`Q=S=D` 垫 K；MatchArgs 必须关

## 怎么拆融合算子

cannbot：融合算子拆成基础族再适配。

| 本仓算子 | 拆成 | 切分层 |
|---|---|---|
| FusedBiasSoftmax | Broadcast(bias) + Reduction(Softmax, AR-FullLoad) | 只 UB。R=width 一次载入，沿 `batch*channel*q` 切行 |
| FusedSoftmaxGrad MIX | MatMul(Cube `g=go@V^T`) + Elementwise(JVP) | L1/L0 归 MatmulImpl；JVP 在 UB。中间 `SyncAll` |
| FusedSoftmaxGrad VEC | 点积 + JVP | 只 UB；仅 fp32/fp16 |

不要去搬 dav310 `softmax_v2_*.template`（Reg/MicroAPI，3510）。
