# Conv3d / conv3d

`torch.nn.Conv3d` / `F.conv3d` 的等价替换（纯 PyTorch 组合实现，无自定义 kernel），
命名与包结构与 torch 原生一一对应，仅引用的包不同：

```python
from torch.nn import Conv3d                                    # 原生
from torch.nn.functional import conv3d                         # 原生
from oil_gas_ops_prospect.conv3d_custom import Conv3d          # 本仓替换
from oil_gas_ops_prospect.conv3d_custom.functional import conv3d  # 本仓替换
```

## 何时用
本算子可直接替换 `nn.Conv3d`，对任意 stride / dilation / groups / padding_mode 的统一前/反向行为。
如下两种情况下优选使用：
- 消除大尺寸场景原生报错：NPU 上原生 `aclnnConvolutionBackward` 对大 kernel depthwise（kernel>=7 + 大空间）会因 L1 buffer 超限崩溃，本实现分步构造反向可正常完成；
- 提供大尺寸场景反向计算性能改善：大尺寸场景性能相对原生约2.4倍提升，小尺寸场景提升不明显，部分劣化。

## 行为

- `Conv3d`（模块）：`use_custom` 跟随 NPU 可用性（`torch_npu` 已导入时走 `conv3d` 自定义前/反向，否则回退 `F.conv3d`）；
- `conv3d`（函数）：始终走自定义前/反向（与原生数学等价），签名与 `F.conv3d` 一致；
- 前向与 `F.conv3d` 完全一致；反向三梯度与原生 autograd 数学等价（float64 实测 <= 1.78e-14），支持双阶反向；
- 反向构造：grad_input = `F.conv3d(grad_output, flip(W)^T, ...)`（stride>1 时零膨胀 + output_padding，
  负 padding 裁剪），grad_weight = `torch.nn.grad.conv3d_weight`，grad_bias = 求和。

## 验证

- 精度用例：[`tests/composite/conv3d_custom/`](../../tests/composite/conv3d_custom/)（场景组合精度 / 训练模式 / 性能基准）；
- 数学等价性与 NPU 实机精度已验证：float64 三梯度与原生 autograd 一致（阈值 1e-12 内）；NPU float32 实际训练尺寸全模式对比通过，20 epochs 训练 loss 正常收敛。

## 对外暴露

- whl：`oil_gas_ops_prospect.conv3d_custom`（`setup.py` 的 `packages` / `package_dir` 登记）；
- 算子登记：`scripts/local_pkg.py` 自动发现。

数学定义与约束详见 [`docs/zh/op_list.md`](../../docs/zh/op_list.md)，签名见 [`docs/zh/api_list.md`](../../docs/zh/api_list.md)。
