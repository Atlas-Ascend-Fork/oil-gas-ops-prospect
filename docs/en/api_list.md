# API List

English | [简体中文](../zh/api_list.md)

All public APIs are exposed by the Python package `oil_gas_ops_prospect`. That directory does not exist in the source tree: at packaging time `setup.py` explicitly maps `ascendc/pybind/`, `triton-ascend/real_fft/`, `triton-ascend/complex_reciprocal/`, `triton-ascend/interpolate/`, and `composite-torch/conv3d_custom/` under the same package name, while when running straight from the in-repo sources with no whl installed, `scripts/local_pkg.py` discovers them automatically and provides the same import path.

```text
oil_gas_ops_prospect
├── complex_mul           # complex_mul, torch_complex_mul, torch_complex_mul_grad
├── fused_bias_softmax    # fused_bias_softmax, fused_softmax_grad, attention_core, torch_*
├── real_fft              # real_rfft, real_irfft, make_basis
├── complex_reciprocal    # complex_reciprocal (either z or xr, xi)
├── interpolate           # interpolate, bilinear, trilinear
└── conv3d_custom         # Conv3d; functional submodule: conv3d
```

For mathematical definitions, shape / dtype constraints and precision thresholds see the [Operator List](op_list.md).

---

## `oil_gas_ops_prospect.complex_mul`

### `complex_mul`

```python
from oil_gas_ops_prospect.complex_mul import complex_mul

def complex_mul(
    ur: torch.Tensor,
    ui: torch.Tensor,
    vr: torch.Tensor,
    vi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]
```

Complex element-wise multiply \((u_r, u_i) \cdot (v_r, v_i) \rightarrow (w_r, w_i)\). It is wrapped in a `torch.autograd.Function` whose backward is carried by the OPP operator `ComplexMulGrad`; there is **no** separate public `complex_mul_grad` function in Python.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ur`, `ui` | `torch.Tensor` | real and imaginary part of the first complex number. The AscendC path requires `float32`, identical shapes and contiguous memory |
| `vr`, `vi` | `torch.Tensor` | real and imaginary part of the second complex number, same shape as `ur` |

Returns `(wr, wi)` with the same shape as the inputs.

**Implementation selection**: the PyTorch reference implementation is used by default. The AscendC fused operator is called only when the tensors are on the NPU and `OIL_GAS_OPS_COMPLEX_MUL_IMPL` is set to `ascendc` or `auto`. On failure it silently falls back to PyTorch unless `OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL=1` is set. With `IMPL=ascendc`, tensors not on the NPU and the custom operator being required raises `RuntimeError`.

### Reference Implementations

The same module also exports two pure PyTorch functions for comparison and CPU testing:

```python
def torch_complex_mul(
    ur: torch.Tensor, ui: torch.Tensor, vr: torch.Tensor, vi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]

def torch_complex_mul_grad(
    gwr: torch.Tensor, gwi: torch.Tensor,
    ur: torch.Tensor, ui: torch.Tensor, vr: torch.Tensor, vi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

The formulas and constraints are those of [Operator List · ComplexMul / ComplexMulGrad](op_list.md#complexmul). Neither uses the NPU custom operator.

---

## `oil_gas_ops_prospect.fused_bias_softmax`

### `fused_bias_softmax`

```python
from oil_gas_ops_prospect.fused_bias_softmax import fused_bias_softmax

def fused_bias_softmax(
    x: torch.Tensor,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
) -> torch.Tensor
```

\(y = \mathrm{softmax}(x + \mathrm{bias1} + \mathrm{bias2},\, -1)\).

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | logits, `[B, H, Q, S]`, `fp32` / `fp16` / `bf16` |
| `bias1` | `torch.Tensor` or `None` | `[B, H, 1, S]` broadcast or `[B, H, Q, S]` full size; `None` means no bias |
| `bias2` | `torch.Tensor` or `None` | `[B, H, Q, S]`; `None` means no bias |

Returns softmax probabilities with the same shape / dtype as `x`. When the fused NPU path is entered successfully, the result shares storage with and writes back to the caller's original `x` only if the supplied tensor is contiguous. For a non-contiguous input, the wrapper creates a `contiguous()` copy; the result aliases that copy and does not overwrite the caller's original non-contiguous tensor. The reference implementation is not in place.

**Implementation selection**: PyTorch by default. AscendC is used when the tensors are on the NPU and `OIL_GAS_OPS_FUSED_BIAS_SOFTMAX_IMPL` is `ascendc` or `auto`. Failures fall back silently unless `OIL_GAS_OPS_REQUIRE_ASCENDC_FUSED_BIAS_SOFTMAX=1`.

### `fused_softmax_grad`

```python
from oil_gas_ops_prospect.fused_bias_softmax import fused_softmax_grad

def fused_softmax_grad(
    probs: torch.Tensor,
    grad_output: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor
```

Computes \(g = \mathrm{grad\_output} \cdot V^{\top}\) first, then the softmax JVP, yielding \(\partial L / \partial \mathrm{logits}\).

| Parameter | Type | Description |
|-----------|------|-------------|
| `probs` | `torch.Tensor` | forward softmax probabilities, `[B, H, Q, S]` (`Q ≠ S` is legal) |
| `grad_output` | `torch.Tensor` | \(\partial L / \partial o\), `[B, H, Q, D]` |
| `values` | `torch.Tensor` | \(V\), `[B, H, S, D]` |

The three inputs share a dtype (`fp32` / `fp16` / `bf16`). Returns a tensor with the same shape as `probs`. When the fused NPU path is entered successfully, the result shares storage with and overwrites the caller's original `probs` only if the supplied tensor is contiguous. For a non-contiguous input, the wrapper creates a `contiguous()` copy; the result aliases that copy and does not overwrite the caller's original non-contiguous tensor. The reference implementation is not in place. The implementation switch is shared with `fused_bias_softmax`.

### `attention_core`

```python
from oil_gas_ops_prospect.fused_bias_softmax import attention_core

def attention_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
) -> torch.Tensor
```

An equivalent wrapper for openfold `AttentionCoreFunction.apply`. It is **not** a standalone OPP operator but a forward/backward composition:

```text
forward:   logits = q @ k^T
           probs  = softmax(logits + bias1 + bias2, -1)   # fused_bias_softmax
           o      = probs @ v
backward:  grad_v = probs^T @ go                          # before probs is overwritten, matching the original CUDA ordering
           g_logits = softmax_jvp(probs, go, v)           # the AscendC fused branch overwrites its contiguous internal probs passed to the extension
           grad_bias* = sum g_logits over the broadcast dimensions of each bias
           grad_q = g_logits @ k
           grad_k = (q^T @ g_logits)^T
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `torch.Tensor` | query, `[B, H, Q, D]` |
| `k` | `torch.Tensor` | key, `[B, H, S, D]` |
| `v` | `torch.Tensor` | value, `[B, H, S, D]` |
| `bias1` | `torch.Tensor` or `None` | typically `[B, H, 1, S]` (mask bias) |
| `bias2` | `torch.Tensor` or `None` | typically `[B, H, Q, S]` (triangle bias) |

Returns `o` with shape `[B, H, Q, D]`. autograd is supported; on an NPU with `ascendc`/`auto` the forward and backward each use one of the two fused operators. Usage matches openfold: `o = attention_core(q, k, v, mask_bias, triangle_bias)`.

### Reference Implementations

```python
def torch_fused_bias_softmax(
    x: torch.Tensor, bias1: torch.Tensor | None = None, bias2: torch.Tensor | None = None,
) -> torch.Tensor

def torch_fused_softmax_grad(
    probs: torch.Tensor, grad_output: torch.Tensor, values: torch.Tensor,
) -> torch.Tensor
```

Both compute in the fp32 domain, are not in place, and write back in the original dtype. They exist for comparison and CPU testing.

---

## `oil_gas_ops_prospect.real_fft`

### `real_rfft`

```python
from oil_gas_ops_prospect.real_fft import real_rfft

def real_rfft(
    x: torch.Tensor,
    dim: int,
    norm: str = "ortho",
    block_n: int | None = None,
    block_m: int | None = None,
    block_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

Performs a real forward FFT along `dim` and returns two real tensors `(real, imag)` whose `dim` axis has length `n // 2 + 1`. autograd is supported.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | input real tensor with length \(n\) along `dim` |
| `dim` | `int` | dimension to transform along |
| `norm` | `str` | **only `"ortho"` is supported**; other values raise `ValueError` |
| `block_n`, `block_m`, `block_k` | `int` or `None` | Triton kernel tiling; `None` lets the implementation choose. `block_m` can also be overridden with `OIL_GAS_OPS_FFT_BLOCK_M` (default 512) |

On an NPU with Triton available the fused kernel is used (computing and returning `float32`); otherwise it falls back to `torch.fft.rfft`.

### `real_irfft`

```python
from oil_gas_ops_prospect.real_fft import real_irfft

def real_irfft(
    xr: torch.Tensor,
    xi: torch.Tensor,
    n: int,
    dim: int,
    norm: str = "ortho",
    block_k: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
) -> torch.Tensor
```

Reconstructs a real signal of length `n` from the half-spectrum real part `xr` and imaginary part `xi`. autograd is supported.

| Parameter | Type | Description |
|-----------|------|-------------|
| `xr`, `xi` | `torch.Tensor` | real and imaginary part of the half spectrum; **their shapes must be identical**, otherwise `ValueError` is raised |
| `n` | `int` | length of the `dim` axis after reconstruction |
| `dim` | `int` | dimension to transform along; the NPU path requires `xr.shape[dim] == n // 2 + 1` |
| `norm` | `str` | **only `"ortho"` is supported** |
| `block_k`, `block_m`, `block_n` | `int` or `None` | Triton kernel tiling |

When not on an NPU, or when Triton is unavailable, it falls back to `torch.fft.irfft`.

### `make_basis`

```python
from oil_gas_ops_prospect.real_fft import make_basis

def make_basis(
    n: int,
    device: torch.device,
    norm: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

Builds, and caches by `(n, device, norm)`, the dense Fourier basis of length `n`. `norm` must be `"ortho"`.

Returns `(rfft_cos, rfft_sin, irfft_cos, irfft_sin)`, all `float32` with shape `(n, n // 2 + 1)`. It normally needs no direct call and is used internally by `real_rfft` / `real_irfft`.

---

## `oil_gas_ops_prospect.complex_reciprocal`

### `complex_reciprocal`

```python
from oil_gas_ops_prospect.complex_reciprocal import complex_reciprocal

def complex_reciprocal(
    z: torch.Tensor,
    xi: torch.Tensor | float | None = None,
    *,
    stable: bool = True,
    block_size: int | None = None,
) -> torch.Tensor
```

Complex64 reciprocal. The call form selects the entry:

- `complex_reciprocal(z)`: `1/z`, where `z` is a complex64 tensor. On an NPU a single Triton kernel (Vector Core) computes it; on CPU, or when Triton is unavailable, it falls back to torch `1 / z` (including autograd).
- `complex_reciprocal(xr, xi)`: `1 / (xr + i*xi)`, with the real and imaginary parts given separately, avoiding an Ai_Cpu operator that would first build a complex64 tensor. A common denominator is "a real tensor plus a constant imaginary part" (for example `1/(a - i*c)`). At least one of `xr` / `xi` is a float32 tensor (they may broadcast); the other may be a tensor or a scalar. The output is complex64.

| Parameter | Type | Description |
|-----------|------|-------------|
| `z` | `torch.Tensor` | one-argument form: a complex64 tensor; two-argument form: a float32 real-part tensor or scalar |
| `xi` | `torch.Tensor` / `float` / `None` | imaginary part of the two-argument form (broadcastable); omitted in the one-argument form |
| `stable` | `bool` | `True` (default) scales by `t=\|a\|+\|b\|` to avoid overflow and underflow of `a^2+b^2`; `False` uses the naive formula (about 10% higher bandwidth, wrong when `\|z\|` falls outside ~[1e-19, 1.8e19]) |
| `block_size` | `int` or `None` | complex elements per tile, rounded down to a power of two and clamped to `[16, 2048]`; `None` means `OIL_GAS_OPS_RECIP_BLOCK` if set, otherwise the default `2048` |

---

## `oil_gas_ops_prospect.interpolate`

### `interpolate`

```python
from oil_gas_ops_prospect.interpolate import interpolate

def interpolate(
    x: torch.Tensor,
    size: int | Sequence[int] | None = None,
    scale_factor: float | Sequence[float] | None = None,
    mode: str = "trilinear",
    align_corners: bool = False,
    antialias: bool = False,
    recompute_scale_factor: bool | None = None,
) -> torch.Tensor
```

2D/3D linear interpolation. On the NPU, 5D `float32` `trilinear` uses the Cube GEMM path (one batched matrix multiply per spatial axis, autograd-ready); 4D and every other signature fall back to `F.interpolate`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | 4D `[B,C,H,W]` or 5D `[B,C,D,H,W]`; the NPU fast path requires `float32` |
| `size` | `int` / sequence / `None` | target spatial size; a 2-tuple for 4D, a 3-tuple for 5D |
| `scale_factor` | `float` / sequence / `None` | passing it falls back to `F.interpolate` |
| `mode` | `str` | `"bilinear"` for 4D, `"trilinear"` for 5D; other modes fall back |
| `align_corners` | `bool` | both values are supported |
| `antialias` | `bool` | `True` falls back to the native kernel |
| `recompute_scale_factor` | `bool` / `None` | used only on the fallback path |

Convenience entries:

```python
from oil_gas_ops_prospect.interpolate import bilinear, trilinear

y2 = bilinear(x2, size=(256, 256))
y3 = trilinear(x3, size=(160, 160, 160))
```

#### Behavior and Constraints

- The NPU fast path supports only 5D `float32` `trilinear`; 4D, fp16/bf16 and fp64 fall back to `F.interpolate`.
- Outputs smaller than 128 MiB use the native kernel. This threshold and the 1 GiB per-launch chunk budget are internal implementation details, not public environment variables.
- First-order autograd is supported; the backward is three axis GEMMs with transposed weights. 4D uses native autograd.
- Recorded verification results are under [`tests/triton/interpolate/report/`](../../tests/triton/interpolate/report/).

See [Environment Variables · Operator Behavior](#operator-behavior) for the public knob.

---

## `oil_gas_ops_prospect.conv3d_custom`

### `Conv3d`

```python
from oil_gas_ops_prospect.conv3d_custom import Conv3d

class Conv3d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

A pure PyTorch composition that is mathematically consistent with `torch.nn.Conv3d` within the supported parameter domain below; `forward(x)` returns \((N, C_{out}, D_{out}, H_{out}, W_{out})\). The current constructor does not provide the `device` / `dtype` keyword arguments of `nn.Conv3d`.

- **Parameter range**: `kernel_size`, `stride`, numeric `padding`, and `dilation` accept an `int` or a 3-element sequence (tuple or list); `groups` is an `int`. The string padding values `"same"` / `"valid"` are supported and are converted to explicit padding before the convolution; `"same"` requires a stride of 1, otherwise `ValueError` is raised.
- **state_dict interoperability**: the keys match those of `nn.Conv3d`, so `load_state_dict` works in both directions (strict).
- **Path selection**: `use_custom` is enabled by default when the `torch.npu` interface is detected; otherwise it falls back to `F.conv3d`. Tests may force `module.use_custom = True` to drive the custom path on CPU.
- **Backward and dtype**: the internal `_Conv3dFunction` organizes forward/backward. Mathematical equivalence is limited to the supported parameter domain above and combinations covered by this repository. Supported dtypes are those supported by the underlying PyTorch backend and tested in this repository, including complex inputs (CPU complex128/complex64 and NPU complex64), for which the backward conjugates the weight and recombines the complex weight gradient from real-valued convolution gradients.
- **`padding_mode`**: the module supports `zeros` / `reflect` / `replicate` / `circular`; non-`zeros` modes apply `F.pad` before convolution.

### `conv3d`

```python
from oil_gas_ops_prospect.conv3d_custom.functional import conv3d

def conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    groups: int = 1,
) -> torch.Tensor
```

A pure PyTorch functional compatibility implementation whose forward/backward is always organized by the internal `_Conv3dFunction` (a `torch.autograd.Function`). `stride`, numeric `padding`, and `dilation` accept an `int` or a 3-element sequence (tuple or list), and `groups` is an `int`; the string padding values `"same"` / `"valid"` are converted to explicit padding before the convolution (`"same"` requires a stride of 1). It is mathematically consistent with `F.conv3d` for these supported parameters and dtypes supported by the underlying PyTorch backend and tested in this repository. For the nn.Module form and NPU fallback logic, see `Conv3d`.

---

## aclnn APIs

Once the OPP `.run` package is installed, the CANN side uses the standard two-phase entry (`GetWorkspaceSize` + execute). The in-repo pybind bindings call it exactly this way:

| API | OpType | Tensors (in order) | Notes |
|-----|--------|--------------------|-------|
| `aclnnComplexMul` / `aclnnComplexMulGetWorkspaceSize` | `ComplexMul` | `ur, ui, vr, vi → wr, wi` | all `float32` / `ND`, identical shapes |
| `aclnnComplexMulGrad` / `aclnnComplexMulGradGetWorkspaceSize` | `ComplexMulGrad` | `gwr, gwi, ur, ui, vr, vi → gur, gui, gvr, gvi` | all `float32` / `ND`, identical shapes |
| `aclnnFusedBiasSoftmax` / `aclnnFusedBiasSoftmaxGetWorkspaceSize` | `FusedBiasSoftmax` | `x, bias1, bias2 → y` | `bias1`/`bias2` may be `nullptr`; `y` may alias `x` |
| `aclnnFusedSoftmaxGrad` / `aclnnFusedSoftmaxGradGetWorkspaceSize` | `FusedSoftmaxGrad` | `probs, gradOutput, values → out` | `out` may alias `probs` |

The opapi shared library of the custom operators lives in `${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib` and must be added to `LD_LIBRARY_PATH` before calling. For the binding implementation see [`ascendc/pybind/csrc/`](../../ascendc/pybind/csrc/).

The Triton operators have no aclnn entry.

---

## Environment Variables

Custom environment variables all use the `OIL_GAS_OPS_` prefix.

### Operator Behavior

| Environment Variable | Values | Default | Description |
|----------------------|--------|---------|-------------|
| `OIL_GAS_OPS_COMPLEX_MUL_IMPL` | `torch` / `ascendc` / `auto` | `torch` | implementation selection for `complex_mul` |
| `OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL` | `0` / `1` | `0` | set to `1` to forbid the PyTorch fallback and raise on an AscendC failure |
| `OIL_GAS_OPS_FUSED_BIAS_SOFTMAX_IMPL` | `torch` / `ascendc` / `auto` | `torch` | implementation selection for `fused_bias_softmax` / `fused_softmax_grad` / `attention_core` |
| `OIL_GAS_OPS_REQUIRE_ASCENDC_FUSED_BIAS_SOFTMAX` | `0` / `1` | `0` | set to `1` to forbid the PyTorch fallback and raise on an AscendC failure |
| `OIL_GAS_OPS_RECIP_BLOCK` | positive integer (power of two) | `2048` | tile size of `complex_reciprocal` (rounded down to a power of two, clamped to `[16, 2048]`); an explicit `block_size` argument takes precedence |
| `OIL_GAS_OPS_INTERPOLATE_IMPL` | `auto` / `triton` / `torch` | `auto` | implementation choice for `interpolate`; `auto` uses Cube for large 5D fp32 outputs and falls back natively for 4D, small outputs and non-fp32 inputs |
| `OIL_GAS_OPS_FFT_BLOCK_M` | positive integer | `512` | row tile size of the DFT matmul in `real_rfft` / `real_irfft`, for performance tuning (sweep 128 / 256 / 512). Read once per process on first use, so set it before starting Python |

### Build and Artifacts

| Environment Variable | Values | Default | Description |
|----------------------|--------|---------|-------------|
| `ASCEND_HOME_PATH` | path | `/usr/local/Ascend/cann-9.0.0` | CANN installation root |
| `OIL_GAS_OPS_OUTPUT_DIR` | path | `<repo>/output` | overrides the build artifact directory |
| `OIL_GAS_OPS_SKIP_EXT` | `0` / `1` | `0` | set to `1` to make `setup.py` skip building the pybind extensions |

### Testing and Validation

| Environment Variable | Values | Default | Description |
|----------------------|--------|---------|-------------|
| `OIL_GAS_OPS_RUN_PKG` | path | auto-detected | tells `tests/ascendc/*/run_test.sh` which OPP `.run` package to install |
| `OIL_GAS_OPS_WHL_REQUIRE_NPU_EXT` | `0` / `1` | `0` | set to `1` to make the whl smoke test require the compiled NPU extension; otherwise that case is skipped |
| `OIL_GAS_OPS_WHL_TEST` | `0` / `1` | `0` | set to `1` by `tests/run_whl_tests.sh` / `bash build.sh --install` to declare that an installed whl is really being validated; in-repo sources are then not mapped onto the `oil_gas_ops_prospect` import name, and resolving to an in-repo path is a hard failure |
| `OIL_GAS_OPS_RUN_LATENT3D_FFT_TEST` | `0` / `1` | `0` | set to `1` to run the heavy NPU FFT case with the large `(2,41,41,41,2048)` shape |
| `OIL_GAS_OPS_RUN_128_FFT_TEST` | `0` / `1` | `0` | set to `1` to run the 128³ NPU backward and memory cases in `tests/triton/real_fft` |

### Cross-Validation Against an External Model Repository (optional)

Some tests of `triton-ascend/real_fft` can cross-validate against an implementation in an external model repository. They are all skipped by default and must be enabled explicitly:

| Environment Variable | Values | Description |
|----------------------|--------|-------------|
| `OIL_GAS_OPS_MODEL_REPO` | path | root of the external model repository; setting it enables cross-validation |
| `OIL_GAS_OPS_MODEL_CONFIG` | path | configuration file used for shape derivation (absolute path) |
| `OIL_GAS_OPS_FFT_BACKBONE` | `<module>:<class>` | overrides the default backbone import path |
| `OIL_GAS_OPS_SPECTRAL_CONV` | `<module>:<class>` | selects the spectral conv implementation to compare against |

For details see [`triton-ascend/real_fft/README.md`](../../triton-ascend/real_fft/README.md) (Chinese).
