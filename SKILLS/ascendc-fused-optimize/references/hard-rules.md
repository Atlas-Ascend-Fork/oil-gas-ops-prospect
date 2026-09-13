# 硬规则

1. **只改融合注意力 AscendC。** 对象是 `fused_bias_softmax` / `fused_softmax_grad`。不要动 `complex_mul`、`triton-ascend/real_fft`。
2. **NPU 路径必须保持为真 kernel。** 禁止用 `torch.softmax` / `bmm` / 纯 PyTorch 拆解冒充优化。
3. **不要改官方精度门槛**（`tests/ascendc/fused_bias_softmax/`、ophost UT 的契约阈值）。
4. **接口与语义不变**：前向原地写回 `x`；反向原地覆写 `probs`；bias1 广播/全尺寸语义不变。
5. **芯片按 910B / Atlas A2（DAV_2201，UB 192KB，24 AIC + 48 AIV）。** 不要上 A5 / DAV_3510 / dav310 MicroAPI Softmax 模板。
6. **bf16 Grad 保持 MIX。** 禁止把对齐/未对齐 bf16 再降回纯 VEC。`Q=S=D` 时继续垫 K（`padD+=16`），禁止垫 N。
7. **禁止重新打开 MatchArgs。** CANN 9 按 shape 命中旧 executor 会让原地写回空跑。
8. **禁止 `DestroyExecutor` / `ResetExecutor` 热路径。**
9. **评测在 NPU 上、标杆同设备 Torch 小算子。** 报告不要写主机名、IP、容器名。
10. **一轮一次主题。** 失败就回退该轮 diff。
11. **全量 OPP**：kernel/tiling 改动用 `bash ascendc/build.sh --pkg -v Ascend910B1 -j 32`，不要 `--ops` 子集。
