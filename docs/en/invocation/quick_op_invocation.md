# Operator Invocation

English | [简体中文](../../zh/invocation/quick_op_invocation.md)

This page gives end-to-end invocation examples for each operator. Runnable scripts live in [`examples/`](../../../examples/).

## Prerequisites

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
```

The OPP package and the whl must already be installed through `bash build.sh --install`.

## AscendC: Complex Element-Wise Multiply

`complex_mul` **uses the PyTorch reference implementation by default**, so switching to the AscendC fused operator has to be explicit:

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

The three values of `OIL_GAS_OPS_COMPLEX_MUL_IMPL`:

| Value | Behavior |
|-------|----------|
| `torch` (default) | always uses the PyTorch reference implementation |
| `ascendc` | uses the AscendC operator; raises when the tensors are not on the NPU and `OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL=1` is set |
| `auto` | uses the AscendC operator when the tensors are on the NPU, otherwise falls back |

**Note**: by default a failed AscendC call **falls back silently** to the PyTorch implementation. To verify that the operator really runs, set:

```bash
export OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL=1
```

### Backward

`complex_mul` is wrapped in a `torch.autograd.Function`, so the backward calls `ComplexMulGrad` automatically:

```python
ur.requires_grad_(True)
vr.requires_grad_(True)

wr, wi = complex_mul(ur, ui, vr, vi)
(wr.sum() + wi.sum()).backward()

print(ur.grad.shape, vr.grad.shape)
```

### Comparing Against the Reference Implementation

```python
from oil_gas_ops_prospect.complex_mul import torch_complex_mul

ref_wr, ref_wi = torch_complex_mul(ur, ui, vr, vi)
rel = (wr - ref_wr).abs().max() / ref_wr.abs().max()
print(f"rel_err = {rel.item():.3e}")
```

## AscendC: Fused Attention

`fused_bias_softmax` / `fused_softmax_grad` / `attention_core` also use the PyTorch reference implementation by default:

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
bias1 = torch.randn(b, h, 1, s, device="npu")   # broadcast
bias2 = torch.randn(b, h, q, s, device="npu")   # full size
probs = fused_bias_softmax(logits, bias1, bias2)  # writes back when the input is contiguous and the fused NPU path succeeds

go = torch.randn(b, h, q, d, device="npu")
values = torch.randn(b, h, s, d, device="npu")
g_logits = fused_softmax_grad(probs, go, values)  # overwrites probs when the input is contiguous and the fused NPU path succeeds

q_t = torch.randn(b, h, q, d, device="npu", requires_grad=True)
k_t = torch.randn(b, h, s, d, device="npu", requires_grad=True)
v_t = torch.randn(b, h, s, d, device="npu", requires_grad=True)
out = attention_core(q_t, k_t, v_t, bias1, bias2)
out.sum().backward()
```

A non-contiguous input is copied through `contiguous()`, so the output aliases only that contiguous copy and does not overwrite the caller's original tensor. The PyTorch reference fallback is also not in place.

Constraint summary: the inputs must be 4D; `fused_bias_softmax` requires `S ≤ 4096`; `fused_softmax_grad` allows `Q ≠ S`. For the full set of constraints see the [Operator List](../op_list.md).

## Triton-Ascend: Real FFT

```python
import torch
import torch_npu  # noqa: F401
from oil_gas_ops_prospect.real_fft import real_irfft, real_rfft

n = 256
x = torch.randn(8, n, device="npu")

# forward: real -> half-spectrum real/imaginary parts, frequency axis length n // 2 + 1 = 129
xr, xi = real_rfft(x, dim=-1)
print(xr.shape)  # torch.Size([8, 129])

# inverse: half spectrum -> real
y = real_irfft(xr, xi, n=n, dim=-1)
print(torch.allclose(x, y, atol=1e-4))
```

Constraints:

- **Only `norm="ortho"` is supported**; other values raise `ValueError`.
- `real_irfft` requires `xr.shape == xi.shape`.
- Outside an NPU environment (or when Triton is unavailable) it falls back to `torch.fft` automatically, so the same code also runs on CPU and can be used for a precision self-check.

### autograd

```python
x = torch.randn(8, 256, device="npu", requires_grad=True)
xr, xi = real_rfft(x, dim=-1)
(xr.pow(2).sum() + xi.pow(2).sum()).backward()
print(x.grad.shape)
```

### Performance Tuning

`OIL_GAS_OPS_FFT_BLOCK_M` controls the row tile size of the DFT matmul. It defaults to 512 and can be swept over 128 / 256 / 512:

```bash
OIL_GAS_OPS_FFT_BLOCK_M=256 python3 your_script.py
```

## Triton-Ascend: Complex Reciprocal

```python
import torch
import torch_npu  # noqa: F401
from oil_gas_ops_prospect.complex_reciprocal import complex_reciprocal

z = torch.randn(4, 1024, dtype=torch.complex64, device="npu")
w = complex_reciprocal(z)                       # 1/z, one Vector Core kernel

# real/imag-separated entry: 1 / (a - i*c) without building the complex denominator
a = torch.randn(4, 1024, device="npu")
w_fused = complex_reciprocal(a, -0.5)           # xi may be a scalar or a broadcastable tensor
```

Constraints:

- The NPU kernel accepts only `complex64` / `float32`; on CPU, for other dtypes, or when Triton is unavailable it falls back to torch `1 / z`, so the same code runs on CPU for a precision self-check.
- `stable=True` (the default) avoids overflow and underflow of `a^2+b^2`. `stable=False` is about 10% faster in bandwidth but is wrong when `|z|` falls outside roughly `[1e-19, 1.8e19]`.
- A single tensor of more than `2^31 - 1` elements raises `ValueError`.

### autograd

```python
z = torch.randn(4, 1024, dtype=torch.complex64, device="npu", requires_grad=True)
complex_reciprocal(z).abs().pow(2).sum().backward()   # d(1/z)/dz = -w^2
print(z.grad.shape)
```

### Performance Tuning

`OIL_GAS_OPS_RECIP_BLOCK` sets the number of complex elements per tile (default 2048, clamped to `[16, 2048]`):

```bash
OIL_GAS_OPS_RECIP_BLOCK=1024 python3 your_script.py
```

## Composite-Torch: Conv3d

```python
import torch
import torch_npu  # noqa: F401
from oil_gas_ops_prospect.conv3d_custom import Conv3d
from oil_gas_ops_prospect.conv3d_custom.functional import conv3d

x = torch.randn(2, 4, 8, 8, 8, device="npu")

# nn.Module form: use_custom is on by default once the torch.npu interface is detected
module = Conv3d(4, 8, kernel_size=3, padding=1).to("npu")
y = module(x)

# functional form: always the custom forward/backward; string padding is accepted
weight = torch.randn(8, 4, 3, 3, 3, device="npu")
y_same = conv3d(x, weight, padding="same")
```

Behavior:

- The `Conv3d` module falls back to `F.conv3d` when the `torch.npu` interface is absent; tests may set `module.use_custom = True` to drive the custom path on CPU.
- The weight/bias state_dict keys match `nn.Conv3d`, so `load_state_dict` works in both directions:

  ```python
  native = torch.nn.Conv3d(4, 8, kernel_size=3, padding=1)
  module.load_state_dict(native.state_dict())
  ```

- `padding="same"` requires a stride of 1; the constructor does not accept the `device` / `dtype` keyword arguments of `nn.Conv3d` (use `.to(...)` as above).

## Calling the aclnn APIs Directly

With the OPP package installed, C++ code can call `aclnnComplexMul` / `aclnnComplexMulGrad` /
`aclnnFusedBiasSoftmax` / `aclnnFusedSoftmaxGrad` directly. The in-repo pybind wrappers serve as a reference:

- [`ascendc/pybind/csrc/complex_mul/bindings.cpp`](../../../ascendc/pybind/csrc/complex_mul/bindings.cpp)
- [`ascendc/pybind/csrc/fused_bias_softmax/bindings.cpp`](../../../ascendc/pybind/csrc/fused_bias_softmax/bindings.cpp)

The include / library paths needed to build that extension are in `build_extensions()` of [`ascendc/pybind_extensions.py`](../../../ascendc/pybind_extensions.py).

## Related Documents

- [API List](../api_list.md): full signatures, parameter tables and environment variables
- [Operator List](../op_list.md): per-operator mathematical definitions, inputs/outputs and constraints
- [Debugging and Profiling](../debug/op_debug_prof.md): how to diagnose a call that has no effect or a precision anomaly
