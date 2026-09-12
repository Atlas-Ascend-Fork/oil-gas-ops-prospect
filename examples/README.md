# 示例

可直接运行的算子调用示例。

| 示例 | 说明 | 需要 NPU |
|------|------|----------|
| [`complex_mul_example.py`](complex_mul_example.py) | AscendC 复数逐点乘的前向、反向与精度对拍 | 否（无卡时走 PyTorch 参考实现） |
| [`real_fft_example.py`](real_fft_example.py) | Triton-Ascend 实数 FFT 的正反变换、往返重建与 autograd | 否（无卡时回退到 `torch.fft`） |

## 运行

两个示例都会自动探测设备：有 NPU 时走自定义算子，无 NPU 时走各自的回退实现，因此在开发容器里也能跑通。未安装 whl 时会自动回落到仓内源码（见 [`_bootstrap.py`](_bootstrap.py)），不需要额外设置 `PYTHONPATH`：

```bash
python3 examples/complex_mul_example.py
python3 examples/real_fft_example.py
```

强制要求走 AscendC 自定义算子（用于确认算子确实被调用，失败时直接抛异常而不是静默回退）：

```bash
OIL_GAS_OPS_COMPLEX_MUL_IMPL=ascendc \
OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL=1 \
python3 examples/complex_mul_example.py
```

## 相关文档

- [算子调用](../docs/zh/invocation/quick_op_invocation.md)
- [接口列表](../docs/zh/api_list.md)
- [快速入门](../docs/QUICKSTART.md)
