# Conv3d / conv3d

支持参数域内兼容 `torch.nn.Conv3d` / `F.conv3d` 的纯 PyTorch 组合实现，无自定义 kernel：

```python
from torch.nn import Conv3d                                    # 原生
from torch.nn.functional import conv3d                         # 原生
from oil_gas_ops_prospect.conv3d_custom import Conv3d          # 本仓替换
from oil_gas_ops_prospect.conv3d_custom.functional import conv3d  # 本仓替换
```

## 何时用
本算子面向已验证参数组合提供与 `nn.Conv3d` 一致的数学语义。`kernel_size`、`stride`、数值 `padding`、`dilation` 支持 `int` 或 3 元素序列（元组或列表），`groups` 为 `int`；模块的 `padding_mode` 支持 `zeros` / `reflect` / `replicate` / `circular`。字符串 `padding="same"` / `"valid"` 受支持，会在卷积前换算为显式补边（`"same"` 要求 stride 为 1，否则抛 `ValueError`）；`Conv3d` 构造器不提供原生模块的 `device` / `dtype` 关键字。dtype 范围以底层 PyTorch 后端支持且经本仓测试者为准。

在仓内覆盖的特定大 kernel depthwise NPU 形态中，分步反向可避开原生反向曾触发的 L1 buffer 限制；该观察不代表所有大 kernel 或任意形态。性能结论也仅适用于测试脚本实际执行并记录的设备、版本与 shape。

## 行为

- `Conv3d`（模块）：检测到 `torch.npu` 接口时 `use_custom` 默认开启，否则回退 `F.conv3d`；
- `conv3d`（函数）：始终走自定义前/反向；在上述支持参数域内与 `F.conv3d` 数学一致；
- weight/bias 的 state_dict key 与 `nn.Conv3d` 互通；
- 反向构造：grad_input = `F.conv3d(grad_output, flip(W)^T, ...)`（stride>1 时零膨胀 + output_padding，
  负 padding 裁剪），grad_weight = `torch.nn.grad.conv3d_weight`，grad_bias = 求和。
- 复数输入的反向对权重取共轭，权重梯度由实部/虚部的实数卷积梯度组合；
  复数精度回归在 NPU 上暂时关闭 HF32，完成后恢复原设置。默认 HF32 下的累加误差不按完整 FP32 阈值验收。

## 验证

- 精度用例：[`tests/composite/conv3d_custom/`](../../tests/composite/conv3d_custom/)（仓内列出的参数组合、训练调用形态与性能基准）；
- 已有用例覆盖 CPU float64/float32 组合及 NPU float32 的指定形态；结论仅适用于实际执行并通过的 case。

## 对外暴露

- whl：`oil_gas_ops_prospect.conv3d_custom`（`setup.py` 的 `packages` / `package_dir` 登记）；
- 算子登记：`scripts/local_pkg.py` 自动发现。

数学定义与约束详见 [`docs/zh/op_list.md`](../../docs/zh/op_list.md)，签名见 [`docs/zh/api_list.md`](../../docs/zh/api_list.md)。
