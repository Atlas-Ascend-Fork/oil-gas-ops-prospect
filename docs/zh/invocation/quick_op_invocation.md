# 算子调用

本文给出各算子的端到端调用示例。可运行脚本见 [`examples/`](../../../examples/)。

## 前置条件

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
```

并已通过 `bash build.sh` 完成 OPP 包与 whl 的安装。

## AscendC：复数逐点乘

`complex_mul` **默认走 PyTorch 参考实现**，需显式切到 AscendC 融合算子：

```python
import os
os.environ["OIL_GAS_OPS_COMPLEX_MUL_IMPL"] = "ascendc"

import torch
import torch_npu  # noqa: F401
from oil_gas_ops_prospect.complex_mul import complex_mul

shape = (4, 1024)
ur = torch.randn(shape, device="npu")
ui = torch.randn(shape, device="npu")
vr = torch.randn(shape, device="npu")
vi = torch.randn(shape, device="npu")

wr, wi = complex_mul(ur, ui, vr, vi)
# wr = ur*vr - ui*vi
# wi = ur*vi + ui*vr
```

`OIL_GAS_OPS_COMPLEX_MUL_IMPL` 的三种取值：

| 取值 | 行为 |
|------|------|
| `torch`（默认） | 始终走 PyTorch 参考实现 |
| `ascendc` | 走 AscendC 算子；张量不在 NPU 上且设置了 `OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL=1` 时抛错 |
| `auto` | 张量在 NPU 上时走 AscendC 算子，否则回退 |

**注意**：默认情况下 AscendC 调用失败会**静默回退**到 PyTorch 实现。验证算子确实被调用时，请设置：

```bash
export OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL=1
```

### 反向

`complex_mul` 是 `torch.autograd.Function` 封装，反向自动调用 `ComplexMulGrad`：

```python
ur.requires_grad_(True)
vr.requires_grad_(True)

wr, wi = complex_mul(ur, ui, vr, vi)
(wr.sum() + wi.sum()).backward()

print(ur.grad.shape, vr.grad.shape)
```

### 与参考实现对拍

```python
from oil_gas_ops_prospect.complex_mul import torch_complex_mul

ref_wr, ref_wi = torch_complex_mul(ur, ui, vr, vi)
rel = (wr - ref_wr).abs().max() / ref_wr.abs().max()
print(f"rel_err = {rel.item():.3e}")
```

## AscendC：融合注意力

`fused_bias_softmax` / `fused_softmax_grad` / `attention_core` 默认同样走 PyTorch 参考实现：

```python
import os
os.environ["OIL_GAS_OPS_FUSED_BIAS_SOFTMAX_IMPL"] = "ascendc"
os.environ["OIL_GAS_OPS_REQUIRE_ASCENDC_FUSED_BIAS_SOFTMAX"] = "1"

import torch
import torch_npu  # noqa: F401
from oil_gas_ops_prospect.fused_bias_softmax import (
    attention_core,
    fused_bias_softmax,
    fused_softmax_grad,
)

b, h, q, s, d = 2, 4, 64, 128, 32
logits = torch.randn(b, h, q, s, device="npu")
bias1 = torch.randn(b, h, 1, s, device="npu")   # 广播
bias2 = torch.randn(b, h, q, s, device="npu")   # 全尺寸
probs = fused_bias_softmax(logits, bias1, bias2)  # NPU 上原地写回 logits

go = torch.randn(b, h, q, d, device="npu")
values = torch.randn(b, h, s, d, device="npu")
g_logits = fused_softmax_grad(probs, go, values)  # NPU 上原地覆写 probs

q_t = torch.randn(b, h, q, d, device="npu", requires_grad=True)
k_t = torch.randn(b, h, s, d, device="npu", requires_grad=True)
v_t = torch.randn(b, h, s, d, device="npu", requires_grad=True)
out = attention_core(q_t, k_t, v_t, bias1, bias2)
out.sum().backward()
```

约束摘要：输入必须是 4D；`fused_bias_softmax` 的 `S ≤ 4096`；`fused_softmax_grad` 允许 `Q ≠ S`。完整约束见[算子列表](../op_list.md)。

## Triton-Ascend：实数 FFT

```python
import torch
import torch_npu  # noqa: F401
from oil_gas_ops_prospect.real_fft import real_irfft, real_rfft

n = 256
x = torch.randn(8, n, device="npu")

# 正向：实数 -> 半谱实部/虚部，频率轴长度 n // 2 + 1 = 129
xr, xi = real_rfft(x, dim=-1)
print(xr.shape)  # torch.Size([8, 129])

# 反向：半谱 -> 实数
y = real_irfft(xr, xi, n=n, dim=-1)
print(torch.allclose(x, y, atol=1e-4))
```

约束：

- **只支持 `norm="ortho"`**，其他值抛 `ValueError`。
- `real_irfft` 要求 `xr.shape == xi.shape`。
- 非 NPU 环境（或 Triton 不可用）自动回退到 `torch.fft`，因此同一段代码在 CPU 上也能跑，可用于精度自检。

### autograd

```python
x = torch.randn(8, 256, device="npu", requires_grad=True)
xr, xi = real_rfft(x, dim=-1)
(xr.pow(2).sum() + xi.pow(2).sum()).backward()
print(x.grad.shape)
```

### 性能调优

`OIL_GAS_OPS_FFT_BLOCK_M` 控制 DFT-matmul 的行分块大小，默认 512，可扫 128 / 256 / 512：

```bash
OIL_GAS_OPS_FFT_BLOCK_M=256 python3 your_script.py
```

## 与 aclnn 接口直连

安装 OPP 包后，C++ 侧可直接调用 `aclnnComplexMul` / `aclnnComplexMulGrad` /
`aclnnFusedBiasSoftmax` / `aclnnFusedSoftmaxGrad`。仓内的 pybind 封装可作参考：

- [`ascendc/pybind/csrc/complex_mul/bindings.cpp`](../../../ascendc/pybind/csrc/complex_mul/bindings.cpp)
- [`ascendc/pybind/csrc/fused_bias_softmax/bindings.cpp`](../../../ascendc/pybind/csrc/fused_bias_softmax/bindings.cpp)

编译该扩展需要的 include / library 路径见 [`ascendc/pybind_extensions.py`](../../../ascendc/pybind_extensions.py) 的 `build_extensions()`。

## 相关文档

- [接口列表](../api_list.md)：完整签名、参数表与环境变量
- [算子列表](../op_list.md)：按算子的数学定义、输入输出与约束
- [调试与性能分析](../debug/op_debug_prof.md)：调用不生效、精度异常时的定位方法
