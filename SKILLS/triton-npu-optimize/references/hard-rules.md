# 硬规则

1. **只改 Triton-Ascend 路径。** 不要动 `ascendc/operators/**`、融合 softmax、ComplexMul。
2. **NPU 路径必须保持为真 kernel。** 禁止用 `torch.fft` / 纯 PyTorch 拆解当「优化」或为了绿测回退。CPU 回退只允许在非 NPU / 无 Triton 时，语义已有。
3. **不要改官方精度门槛**（`tests/triton/real_fft/`、`verify_triton_fft_real_128_3d.py` 的 atol/rtol）。
4. **接口不变**：`real_rfft` / `real_irfft` / `make_basis` 的公开签名与 `norm="ortho"` 约束不变。`make_basis` 仍返回 4 个 `(n, freq_len)` 矩阵。
5. **芯片按 910B / Atlas A2。** 不要上 Helix A5 卡（`force_simt_only`、`al.scope` CANN ext）。
6. **不要引入 `triton.autotune` 大海捞针**，除非手工切分已证明结构合理且编译次数可接受。优先 shape 公式 + 已有 `_cap_fft_tiles` / `_launch_with_block_m_backoff`。
7. **5D 业务形态是回归门禁。** `[1,35,128,128,128]` 各轴相对 `torch.fft` 的加速比不得无说明地掉回 ~1x。
8. **评测在 NPU 容器内跑**，命令见 [eval.md](eval.md)。报告不要写主机名、IP、容器名。
9. **一轮一次主题。** 失败就回退该轮 diff，不要堆互不相干的改动。
10. **AOT 门禁**：改 kernel 签名或 constexpr 后跑 `python3 triton-ascend/tools/aot_compile_check.py`（有 CANN/triton 的环境）。
