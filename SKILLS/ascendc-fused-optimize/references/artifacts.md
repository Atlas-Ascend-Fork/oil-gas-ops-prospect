# 产物

- 算子改动：`ascendc/operators/fused_bias_softmax/`、`ascendc/operators/fused_softmax_grad/`、对应 `tests/ut/op_host/`。
- 轮次笔记：`extra-info/fused-optimize/`（**不提交**）。
- 每轮至少写：方向、改了什么、ophost / 官方精度、vs Torch 表、keep 或 revert。
- 对外测试报告若更新倍数，只写 Atlas A2 / CANN 9.0.0，不写主机。
