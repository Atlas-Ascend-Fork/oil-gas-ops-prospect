# Operator List

English | [简体中文](../zh/op_list.md)

This repository currently delivers 6 AscendC/CANN OPP operators, 4 Triton-Ascend Python operators and 1 Composite-Torch (pure PyTorch) operator. Each operator is documented with its mathematical definition, inputs/outputs and constraints; for Python / aclnn signatures see the [API List](api_list.md).

## Overview

| Name / OpType | Backend | aclnn / Python | Source | Function | dtype / format | Supported SoC |
|---------------|---------|----------------|--------|----------|----------------|---------------|
| `ComplexMul` | AscendC | `aclnnComplexMul` / `complex_mul` | [`ascendc/operators/complex_mul/`](../../ascendc/operators/complex_mul/) | complex element-wise multiply | `float32` / `ND` | Ascend 910B |
| `ComplexMulGrad` | AscendC | `aclnnComplexMulGrad` (the backward is driven by the autograd of `complex_mul`) | [`ascendc/operators/complex_mul_grad/`](../../ascendc/operators/complex_mul_grad/) | backward of `ComplexMul` | `float32` / `ND` | Ascend 910B |
| `FusedBiasSoftmax` | AscendC | `aclnnFusedBiasSoftmax` / `fused_bias_softmax` | [`ascendc/operators/fused_bias_softmax/`](../../ascendc/operators/fused_bias_softmax/) | `softmax(x + bias1 + bias2, -1)`, in-place for contiguous inputs that successfully enter the fused path | `fp32` / `fp16` / `bf16` / `ND` | Ascend 910B |
| `FusedSoftmaxGrad` | AscendC | `aclnnFusedSoftmaxGrad` / `fused_softmax_grad` | [`ascendc/operators/fused_softmax_grad/`](../../ascendc/operators/fused_softmax_grad/) | attention softmax JVP, overwrites probs for contiguous inputs that successfully enter the fused path | `fp32` / `fp16` / `bf16` / `ND` | Ascend 910B |
| `Conv3DGelu` | AscendC | `aclnnConv3dGeluCustom` / `conv3d_gelu` | [`ascendc/operators/conv3d_gelu/`](../../ascendc/operators/conv3d_gelu/) | `gelu_tanh(conv3d(x, weight, bias))` | `float32` / `NCDHW` | Ascend 910B |
| `Conv3DGeluGrad` | AscendC | `aclnnConv3dGeluGrad` (called by `conv3d_gelu` autograd) | [`ascendc/operators/conv3d_gelu_grad/`](../../ascendc/operators/conv3d_gelu_grad/) | recompute conv3d, then GELU backward, then conv3d backward | `float32` / `NCDHW` | Ascend 910B |
| `real_rfft` | Triton-Ascend | `oil_gas_ops_prospect.real_fft.real_rfft` | [`triton-ascend/real_fft/`](../../triton-ascend/real_fft/) | real forward FFT: `x → (xr, xi)` | the NPU kernel is `float32` | Ascend 910B |
| `real_irfft` | Triton-Ascend | `oil_gas_ops_prospect.real_fft.real_irfft` | [`triton-ascend/real_fft/`](../../triton-ascend/real_fft/) | real reconstruction from a half spectrum: `(xr, xi), n → y` | the NPU kernel is `float32` | Ascend 910B |
| `complex_reciprocal` | Triton-Ascend | `oil_gas_ops_prospect.complex_reciprocal.complex_reciprocal` | [`triton-ascend/complex_reciprocal/`](../../triton-ascend/complex_reciprocal/) | complex reciprocal `1/z` (`z` or `xr, xi`) | the NPU kernel is `complex64` | Ascend 910B |
| `interpolate` | Triton-Ascend | `oil_gas_ops_prospect.interpolate.interpolate` | [`triton-ascend/interpolate/`](../../triton-ascend/interpolate/) | 2D/3D linear interpolation; 5D fp32 uses Cube GEMM, 4D uses the native kernel | `float32` for 5D; 4D is supported by the native path | Ascend 910B |
| `Conv3d` / `conv3d` | Composite-Torch | `oil_gas_ops_prospect.conv3d_custom.Conv3d` / `.conv3d` | [`composite-torch/conv3d_custom/`](../../composite-torch/conv3d_custom/) | compatible `nn.Conv3d` / `F.conv3d` implementation within the supported parameter domain (custom forward/backward) | data types supported by the underlying PyTorch backend and tested here | supported by the PyTorch backend; tested on CPU / Ascend 910B |

Naming conventions:

- The C++ OpType is PascalCase (`ComplexMul`), the aclnn API is `aclnnComplexMul`.
- Public Python functions use snake_case (`complex_mul`) with no `npu_` prefix.
- The AscendC kernel entry is pinned to a snake_case file name through `opFile.value` / `opInterface.value` (for example `complex_mul`), so renaming the OpType does not touch the kernel sources, but the aclnn API name follows the OpType.

---

## AscendC Operators

The deliverable is a CANN OPP `.run` operator package. Once installed it exposes `aclnn` APIs, and it is called from Python through `ascendc/pybind/`.

### ComplexMul

Complex element-wise multiply. An AIV-only element-wise operator supporting dynamic rank and dynamic shape.

#### Mathematical Definition

The inputs are treated as two complex numbers \(u = u_r + i u_i\) and \(v = v_r + i v_i\), and the output is \(w = u \cdot v\):

\[
\begin{aligned}
w_r &= u_r v_r - u_i v_i \\
w_i &= u_r v_i + u_i v_r
\end{aligned}
\]

#### Inputs and Outputs

| Direction | Name | Meaning | shape | dtype |
|-----------|------|---------|-------|-------|
| input | `ur`, `ui` | real and imaginary part of the first complex number | any rank, identical to each other | `float32` / `ND` |
| input | `vr`, `vi` | real and imaginary part of the second complex number | same as `ur` | `float32` / `ND` |
| output | `wr`, `wi` | real and imaginary part of the product | same as `ur` | `float32` / `ND` |

#### Constraints

- The four inputs and two outputs must share the **same shape, same dtype, `float32`, `ND`**.
- Inputs must be contiguous (the Python binding calls `contiguous()`).
- **AIV-only**: the total element count is split evenly across the AIV cores (32B aligned), and within a core it is further split into tiles by the UB budget. The forward UB model uses 13 tiles (double buffering for 4 input queues and 2 output queues, plus 1 tmp).
- The tiling algorithm lives in `op_host/complex_mul_tiling_core.h` and is covered by [`tests/ut/op_host/complex_mul/test_complex_mul_tiling.cpp`](../../tests/ut/op_host/complex_mul/test_complex_mul_tiling.cpp).
- Python uses the PyTorch reference implementation by default; this operator is called only when the tensors are on the NPU and `OIL_GAS_OPS_COMPLEX_MUL_IMPL` is `ascendc` or `auto`.
- Precision acceptance: relative error with a default threshold of `1e-5` (see [`tests/ascendc/complex_mul/`](../../tests/ascendc/complex_mul/)).

There is no separate `complex_mul_grad` function on the Python side; the backward is invoked through the `torch.autograd.Function` of `complex_mul`, which calls `ComplexMulGrad`.

---

### ComplexMulGrad

Backward of `ComplexMul`. An AIV-only element-wise operator.

#### Mathematical Definition

For \(w = u \cdot v\) with a known \(\partial L / \partial w = g_w = g_{w_r} + i g_{w_i}\):

\[
\begin{aligned}
g_{u_r} &= g_{w_r} v_r + g_{w_i} v_i \\
g_{u_i} &= -g_{w_r} v_i + g_{w_i} v_r \\
g_{v_r} &= g_{w_r} u_r + g_{w_i} u_i \\
g_{v_i} &= -g_{w_r} u_i + g_{w_i} u_r
\end{aligned}
\]

#### Inputs and Outputs

| Direction | Name | Meaning | shape | dtype |
|-----------|------|---------|-------|-------|
| input | `gwr`, `gwi` | real and imaginary part of \(\partial L / \partial w\) | same as the forward inputs | `float32` / `ND` |
| input | `ur`, `ui`, `vr`, `vi` | the inputs saved by the forward | same as `gwr` | `float32` / `ND` |
| output | `gur`, `gui`, `gvr`, `gvi` | \(\partial L / \partial u\) and \(\partial L / \partial v\) | same as `gwr` | `float32` / `ND` |

#### Constraints

- The six inputs and four outputs must share the **same shape, same dtype, `float32`, `ND`**.
- **AIV-only**. The backward UB model uses 21 tiles (double buffering for 6 inputs and 4 outputs, plus tmp).
- The tiling algorithm lives in `op_host/complex_mul_grad_tiling_core.h`, shares `og_elementwise_tiling.h` with the forward, and is covered by the same ophost UT.
- C++ / aclnn can call it directly; from Python it is only triggered through `complex_mul(...).backward()`.

---

### FusedBiasSoftmax

openfold attention forward fusion: the bias accumulation and the softmax along the last dimension are fused into a single kernel; the low-level aclnn output may alias `x`. It replaces the three launches of `attn_core_inplace_cuda.forward_` (bias1 / bias2 / in-place softmax).

#### Mathematical Definition

\[
y = \mathrm{softmax}(x + b_1 + b_2,\; \mathrm{dim}=-1)
\]

Both \(b_1\) and \(b_2\) may be absent (treated as 0). Non-`fp32` inputs are accumulated and softmaxed in the fp32 domain, then written back in the original dtype.

#### Inputs and Outputs

| Direction | Name | Meaning | shape | dtype |
|-----------|------|---------|-------|-------|
| input | `x` | logits | `[B, H, Q, S]` | `fp32` / `fp16` / `bf16`, `ND` |
| input (optional) | `bias1` | broadcastable bias (for example a mask bias) | `[B, H, 1, S]` or `[B, H, Q, S]` | same as `x` |
| input (optional) | `bias2` | full-size bias (for example a triangle bias) | `[B, H, Q, S]` | same as `x` |
| output | `y` | softmax probabilities | same as `x` | same as `x` |

The low-level aclnn API allows **`y` to alias `x`**. The Python wrapper shares storage with and writes back to the caller's original `x` only when `x` is contiguous and the fused NPU path is entered successfully. A non-contiguous input is copied through `contiguous()` first, so `y` aliases the copy and does not overwrite the original non-contiguous tensor. The PyTorch reference implementation `torch_fused_bias_softmax` is not in place.

#### Constraints

- **Must be 4D.** `S = width` satisfies \(1 \le S \le 4096\); `Q = height`, `B` and `H` must all be greater than 0.
- The `B/H/S` of `bias1` must match `x`, and its `Q` dimension may only be `1` (broadcast) or equal to the `Q` of `x` (full size).
- If `bias2` is present it must be **exactly the same full size** as `x`.
- A bias that is present must have the **same dtype** as `x`.
- **AIV-only**. Rows (`B*H*Q`) are split evenly across the AIV cores, each row's `S` is aligned to 32B, and 32KB of UB is reserved for the high-level SoftMax API.
- tiling keys: `0=fp32`, `1=fp16`, `2=bf16`.
- The tiling algorithm lives in `op_host/fused_bias_softmax_tiling_core.h` and is covered by [`tests/ut/op_host/fused_bias_softmax/test_fused_bias_softmax_tiling.cpp`](../../tests/ut/op_host/fused_bias_softmax/test_fused_bias_softmax_tiling.cpp).
- Precision acceptance ([`tests/ascendc/fused_bias_softmax/`](../../tests/ascendc/fused_bias_softmax/)): fp32 `max_diff ≤ 1e-5`; bf16 contract `atol=2e-2, rtol=4e-3`.

---

### FusedSoftmaxGrad

openfold attention backward fusion: it first computes \(g = go \cdot V^\top\), then performs the softmax Jacobian-vector product along the last dimension; the low-level aclnn output may alias `probs`. It replaces `attn_core_inplace_cuda.backward_`, that is `bmm + _softmax_backward_data + copy_`.

#### Mathematical Definition

\[
\begin{aligned}
g &= \mathrm{grad\_output} \cdot V^{\top} \\
\mathrm{out} &= P \odot \bigl(g - \mathrm{rowsum}(P \odot g)\bigr)
\end{aligned}
\]

\(P\) is the forward softmax probability (its last dimension sums to 1). The computation is done in the **fp32 domain** and then written back in the original dtype. `out` is \(\partial L / \partial \mathrm{logits}\).

#### Inputs and Outputs

| Direction | Name | Meaning | shape | dtype |
|-----------|------|---------|-------|-------|
| input | `probs` | forward softmax probability \(P\) | `[B, H, Q, S]` | `fp32` / `fp16` / `bf16`, `ND` |
| input | `gradOutput` | \(\partial L / \partial o\) | `[B, H, Q, D]` | same as `probs` |
| input | `values` | attention \(V\) | `[B, H, S, D]` | same as `probs` |
| output | `out` | \(\partial L / \partial \mathrm{logits}\) | same as `probs` | same as `probs` |

The low-level aclnn API allows **`out` to alias `probs`**. The Python wrapper shares storage with and overwrites the caller's original `probs` only when `probs` is contiguous and the fused NPU path is entered successfully. A non-contiguous input is copied through `contiguous()` first, so `out` aliases the copy and does not overwrite the original non-contiguous tensor. The PyTorch reference implementation `torch_fused_softmax_grad` is not in place.

`Q ≠ S` is legal (the openfold MSA / triangle attention shapes).

#### Constraints

- **Must be 4D.** `B/H` must match across the three inputs; the trailing dimension `D` of `gradOutput` and `values` must be identical and greater than 0; `Q` and `S` must be greater than 0.
- The three inputs must share the **same dtype**, one of `fp32` / `fp16` / `bf16`.
- Path dispatch (the tiling key, consistent with `TILING_KEY_IS` in the kernel):

  | key | Path | Condition |
  |-----|------|-----------|
  | 0 | pure vector (AIV) | `fp32` |
  | 1 | pure vector (AIV) | `fp16` |
  | 2 | pure vector (AIV) | `bf16` with `S%16 ≠ 0` or `D%16 ≠ 0` |
  | 3 | MIX_AIC_1_2 (AIC matmul → workspace → AIV JVP) | `bf16` with `S%16==0` and `D%16==0` |

- `fp32` / `fp16` do not use the cube: the Ascend 910B fixpipe C is structurally limited to 16 bits, and the quantization error of storing `g` as half would exceed the fp32 contract (`rtol=1e-4`).
- The vector path requires that the \(V\) plane (`S * pad(D) * 4` bytes) plus roughly 48KB of headroom fits into the AIV UB; otherwise tiling fails.
- The MIX path workspace is the 16MB system reservation plus a `g` buffer of `B*H*Q*S*sizeof(half)`.
- The tiling algorithm lives in `op_host/fused_softmax_grad_tiling_core.h` and is covered by [`tests/ut/op_host/fused_softmax_grad/test_fused_softmax_grad_tiling.cpp`](../../tests/ut/op_host/fused_softmax_grad/test_fused_softmax_grad_tiling.cpp).
- Precision acceptance shares the directory with the forward: fp32 `max_diff ≤ 1e-5`; bf16 `atol=2e-2, rtol=4e-3`. On the MIX path the half-`g` quantization error is amplified by `rowsum(P⊙g)` proportionally to \(\sum P^2\), so production inputs are expected to be softmax distributions.

`attention_core(q, k, v, bias1, bias2)` is a composed wrapper over the two operators above following the semantics of openfold `AttentionCoreFunction.apply`; it is not a standalone OPP operator. For its mathematical definition see the [API List](api_list.md#attention_core).

### Conv3DGelu

A cube+vector fusion of Conv3d and the tanh-approximation GELU. It matches the tanh approximation that `F.gelu` / `nn.GELU` actually run on NPU, which is also CPU `F.gelu(approximate="tanh")`. CPU `nn.GELU()` defaults to erf and is not the reference for this operator.

#### Mathematical Definition

\[
y = \text{gelu}_{\tanh}(\text{conv3d}(x, W, b))
\]

\[
\text{gelu}_{\tanh}(x) = \frac{x}{1 + \exp(-1.5957691216057308\,(x + 0.044715\, x^3))}
\]

For 1×1 kernels with stride 1, padding 0, at least 8 input channels and at most 64 output channels, GELU is written directly into the user output. Other shapes run one output plane at a time, so the private-format buffer follows that plane. The backward does not save the pre-activation. `Conv3DGeluGrad` recomputes conv3d, applies the GELU backward, then the conv3d backward, in one call.

#### Inputs and Outputs

| Direction | Name | Meaning | shape | dtype |
|-----------|------|---------|-------|-------|
| input | `x` | feature map | `NCDHW` | `float32` |
| input | `filter` | kernel | `(C_out, C_in/groups, Kd, Kh, Kw)` | `float32` |
| input | `bias` | optional bias | `(C_out,)` | `float32` |
| output | `y` | result after GELU | `NCDHW` | `float32` |

#### Constraints

- `float32` and `groups=1` only. `groups` other than 1, or a `stride` / `padding` / `dilation` whose length is not 1 or 3, raises `ValueError`. When `padding_mode` is not `zeros`, Python pads first.
- Precision is checked against CPU `F.gelu(F.conv3d(...), approximate="tanh")` with `torch_npu.npu.conv.allow_hf32` off, and the relative error must be `< 1e-5`. See [`tests/ascendc/conv3d_gelu/`](../../tests/ascendc/conv3d_gelu/).
- Required shapes match the original `oil_gas_ops` conv3d_gelu cases, including `(2, 32, 160, 160, 160)` with a `3×3×3` kernel.

---

## Triton-Ascend Operators

The deliverable is a set of Triton kernels inside the Python whl, with autograd support. On the NPU the dense Fourier basis matrix is run on the cube as a `tl.dot` GEMM; when not on an NPU, or when Triton is unavailable, it falls back to `torch.fft` with identical interface semantics.

### real_rfft

Performs a real forward FFT along the given dimension and returns the real and imaginary parts of the half spectrum.

#### Mathematical Definition

For a real sequence of length \(n\) (`norm="ortho"`):

\[
X[k] = n^{-1/2} \sum_{t=0}^{n-1} x[t]\, e^{-2\pi i k t / n},\quad k = 0,1,\ldots,\lfloor n/2 \rfloor
\]

The implementation builds and caches basis matrices (`make_basis`):

\[
\begin{aligned}
\mathrm{rfft\_cos}[t,k] &= n^{-1/2}\cos(2\pi k t / n) \\
\mathrm{rfft\_sin}[t,k] &= -n^{-1/2}\sin(2\pi k t / n)
\end{aligned}
\]

then computes \(X_r = x \cdot \mathrm{rfft\_cos}\) and \(X_i = x \cdot \mathrm{rfft\_sin}\). The frequency axis has length \(n//2+1\).

The backward (autograd) maps \((\partial L/\partial X_r,\, \partial L/\partial X_i)\) back to \(\partial L/\partial x\) with the same basis, matching `torch.fft.rfft(..., norm="ortho")`.

#### Inputs and Outputs

| Direction | Name | Meaning | shape | dtype |
|-----------|------|---------|-------|-------|
| input | `x` | real signal | any rank, with length \(n\) along `dim` | the NPU kernel computes and returns `float32` |
| input | `dim` | dimension to transform along | scalar | `int` |
| input | `norm` | normalization | only `"ortho"` is accepted | `str` |
| input | `block_n` / `block_m` / `block_k` | Triton GEMM tiling; `None` lets the implementation choose | optional | `int` |
| output | `xr`, `xi` | real and imaginary part of the half spectrum | `dim` becomes \(n//2+1\), other dimensions unchanged | `float32` |

#### Constraints

- **Only `norm="ortho"` is supported**; other values raise `ValueError`.
- On an NPU with Triton available the fused kernel is used; otherwise it falls back to `torch.fft.rfft`, which allows a precision self-check on CPU.
- Basis matrices are cached by `(n, device, norm)`.
- `block_m` defaults to 512 and can be overridden with `OIL_GAS_OPS_FFT_BLOCK_M`; the implementation shrinks it to fit the 910B L0C capacity and automatically retries with half the value on overflow.
- autograd is supported.

---

### real_irfft

Reconstructs a real signal of length \(n\) from the real and imaginary parts of a half spectrum; the conjugate-symmetric negative frequencies are filled in by the implementation following `torch.fft.irfft`.

#### Mathematical Definition

With `norm="ortho"` it is equivalent to `torch.fft.irfft(torch.complex(xr, xi), n=n, dim=dim, norm="ortho")`. On the basis-matrix side:

\[
\begin{aligned}
w[0] &= 1,\quad w[k] = 2\ (1 \le k < (n+1)//2),\quad \text{Nyquist (}n\text{ even)}=1 \\
\mathrm{irfft\_cos}[t,k] &= n^{-1/2}\, w[k]\, \cos(2\pi k t / n) \\
\mathrm{irfft\_sin}[t,k] &= n^{-1/2}\, w[k]\, \sin(2\pi k t / n)
\end{aligned}
\]

\[
y[t] = \sum_{k=0}^{n//2} \bigl( X_r[k]\,\mathrm{irfft\_cos}[t,k] - X_i[k]\,\mathrm{irfft\_sin}[t,k] \bigr)
\]

#### Inputs and Outputs

| Direction | Name | Meaning | shape | dtype |
|-----------|------|---------|-------|-------|
| input | `xr`, `xi` | real and imaginary part of the half spectrum | identical to each other; `dim` must be \(n//2+1\) | the NPU kernel is `float32` |
| input | `n` | length of the reconstructed real signal | scalar | `int` |
| input | `dim` | dimension to transform along | scalar | `int` |
| input | `norm` | normalization | only `"ortho"` is accepted | `str` |
| input | `block_k` / `block_m` / `block_n` | Triton GEMM tiling; `None` lets the implementation choose | optional | `int` |
| output | `y` | reconstructed real signal | length \(n\) along `dim`, other dimensions unchanged | same as `xr` |

#### Constraints

- **Only `norm="ortho"` is supported**; other values raise `ValueError`.
- `xr.shape` must equal `xi.shape`, otherwise `ValueError` is raised.
- On the NPU path, `xr.shape[dim] != n//2+1` raises `ValueError`.
- When not on an NPU, or when Triton is unavailable, it falls back to `torch.fft.irfft`.
- It shares the basis cache and `OIL_GAS_OPS_FFT_BLOCK_M` with `real_rfft`.
- autograd is supported.

`make_basis(n, device, norm)` is an internal helper that normally needs no direct call; for its signature see the [API List](api_list.md#make_basis).

For implementation and debugging details see [`triton-ascend/real_fft/README.md`](../../triton-ascend/real_fft/README.md) (Chinese).

---

### complex_reciprocal

Reciprocal of a complex64 tensor. Seismic inversion workloads use it in place of `aclnnReciprocal`, which runs on Ai_Cpu on the NPU.

#### Mathematical Definition

For \(z = a + ib\):

\[
w = \frac{1}{z} = \frac{\bar{z}}{|z|^2} = \frac{a - ib}{a^2 + b^2}
\]

- **stable path** (default): scale by \(t = |a| + |b|\) and compute \(w = \frac{\hat{a} - i\,\hat{b}}{t\,(\hat{a}^2 + \hat{b}^2)}\) with \(\hat{a} = a/t,\ \hat{b} = b/t\), which is algebraically equal to \(\frac{a - ib}{a^2+b^2}\) but avoids overflow and underflow of \(a^2+b^2\);
- **naive path**: compute \(a^2+b^2\) directly. Bandwidth is higher (about 10%), but the result is wrong when \(|z|\) falls outside roughly \([10^{-19}, 1.8\times10^{19}]\).

Backward: \(\partial(1/z)/\partial z = -1/z^2 = -w^2\). The kernel supports both interleaved and real/imag-separated gradient outputs.

The two-argument form `complex_reciprocal(xr, xi)` computes \(1/(xr + i\,xi)\). `xr` / `xi` are broadcastable float32 tensors or scalars, avoiding an Ai_Cpu operator that would first build a complex64 tensor.

#### Inputs and Outputs

| Direction | Name | Meaning | shape | dtype |
|-----------|------|---------|-------|-------|
| input | `z` | complex tensor | arbitrary | `complex64` |
| output | `w` | reciprocal | same as `z` | `complex64` |
| input | `xr`, `xi` | real / imaginary part (broadcastable; at least one is a tensor) | arbitrary | `float32` or a scalar |
| output | `w` | `1/(xr + i*xi)` | broadcast result | `complex64` |

#### Constraints

- The NPU kernel accepts only `complex64` / `float32`. On CPU, or for other data types, it falls back to the torch reference (including autograd).
- From `2^30` elements on, the kernel takes the int64 offset branch (selected automatically; covered by AOT). A single tensor of more than `2^31 - 1` elements is rejected with `ValueError`, because the in-block int32 arithmetic would wrap around.
- `block_size` is the number of complex elements per tile, `2048` by default. It is rounded down to a power of two and clamped to `[16, 2048]`, and can be overridden with `OIL_GAS_OPS_RECIP_BLOCK`.
- autograd is supported (a custom backward kernel), as are both entries (interleaved and real/imag-separated).

For implementation and usage notes see [`triton-ascend/complex_reciprocal/README.md`](../../triton-ascend/complex_reciprocal/README.md).

---

### interpolate

2D/3D linear interpolation. For 5D `float32` `trilinear` input, each spatial axis is evaluated as one batched GEMM on the Cube unit; 4D `bilinear` and other signatures fall back to `F.interpolate`.

#### Mathematical definition

For one axis, linear interpolation is a matrix multiply:

\[
Y = X W,\qquad W\in\mathbb{R}^{n_{\text{in}}\times n_{\text{out}}}
\]

Each column of \(W\) has two non-zero entries when `align_corners=False`:

\[
W_{i_0,j}=1-w_j,\qquad W_{i_1,j}=w_j
\]

3D trilinear interpolation is the separable composition of the three axes:

\[
Y = \bigl((X W_w) W_h\bigr) W_d
\]

The implementation runs three batched GEMMs in `W -> H -> D` order:
`M = batch × remaining spatial extent`, `K = input axis length`, and
`N = output axis length`. Each pass uses a persistent grid of `num_aicore`
programs; tile counts and block sizes are `tl.constexpr`, while shapes and
strides are runtime arguments. The weight matrices are generated by calling
PyTorch's own `F.interpolate` on an identity matrix, so they match the native
fp32 weights exactly.

#### Inputs and outputs

| Direction | Name | Meaning | Shape | Dtype |
|-----------|------|---------|-------|-------|
| Input | `x` | input features | 4D `[B,C,H,W]` or 5D `[B,C,D,H,W]` | `float32` for the NPU kernel |
| Input | `size` | target spatial size | 2- or 3-tuple | `int` |
| Input | `mode` | interpolation mode | `"bilinear"` (4D) or `"trilinear"` (5D) | `str` |
| Input | `align_corners` | alignment mode | `False` / `True` | `bool` |
| Output | `y` | interpolation result | spatial dims replaced by `size` | same as `x` |

#### Constraints

- The NPU fast path supports only 5D `float32` `trilinear`, with either `align_corners` value. 4D, `nearest`, `bicubic`, `scale_factor`, `antialias`, fp16/bf16 and fp64 fall back to `F.interpolate`.
- The fast path requires an explicit `size`; `scale_factor` uses the native kernel.
- Outputs smaller than 128 MiB use the native kernel. This threshold and the 1 GiB per-launch chunk budget are internal implementation details, not public environment variables.
- First-order autograd is supported; the backward is three axis GEMMs with transposed weights. 4D uses native `F.interpolate` autograd.
- Recorded verification results are under [`tests/triton/interpolate/report/`](../../tests/triton/interpolate/report/).

For implementation, environment variables and debugging notes see [`triton-ascend/interpolate/README.md`](../../triton-ascend/interpolate/README.md).

---

## Composite-Torch Operators

The deliverable is a pure PyTorch composed module inside the Python whl (`torch.autograd.Function` organizes the forward/backward, with no custom kernel), behaving identically on CPU and NPU.

### Conv3d

A compatible implementation of `torch.nn.Conv3d` / `F.conv3d` within the supported parameter domain below. When the `torch.npu` interface is detected, the module uses the custom `conv3d` forward/backward by default; otherwise it falls back to `F.conv3d`. weight/bias remain top-level parameters, so the state_dict keys match those of `nn.Conv3d`.

#### Mathematical Definition

Within the supported parameter domain, the forward uses the same 3D cross-correlation definition as `F.conv3d`. Given \(\partial L/\partial \text{out}\), the backward produces three gradients:

\[
\begin{aligned}
\text{grad\_bias}[o] &= \sum_{n,d,h,w} \text{grad\_out}[n,o,d,h,w] \\
\text{grad\_weight} &= \texttt{torch.nn.grad.conv3d\_weight}(\text{input}, \text{grad\_out}, \ldots) \\
\text{grad\_in} &= \texttt{F.conv3d}\bigl(\widetilde{\text{grad\_out}},\ \text{flip}(W)^{\top},\ \text{pad}=d(K{-}1)-p,\ \ldots\bigr)
\end{aligned}
\]

where \(\widetilde{\text{grad\_out}}\) is the zero-dilated grad_output (with \(s-1\) zeros inserted between elements when stride>1, and `output_padding` padding it back to the input size), and \(\text{flip}(W)^{\top}\) is the spatial flip plus in/out channel transpose (transposed within each group when groups>1). Negative padding crops both sides of the dilated tensor.

#### Inputs and Outputs

| Direction | Name | Meaning | shape | dtype |
|-----------|------|---------|-------|-------|
| input | `x` | input features | \((N, C_{in}, D, H, W)\) | data types supported by the underlying PyTorch backend and tested in this repository |
| output | `output` | output features | \((N, C_{out}, D_{out}, H_{out}, W_{out})\) | same as the input |

`kernel_size`, `stride`, numeric `padding`, and `dilation` accept an `int` or a 3-element sequence (tuple or list), and `groups` is an `int`. The module's `padding_mode` supports `zeros` / `reflect` / `replicate` / `circular`. The string padding values `"same"` / `"valid"` are supported and are converted to explicit padding before the convolution; `"same"` requires a stride of 1, otherwise `ValueError` is raised. The current `Conv3d` constructor does not have the `device` / `dtype` keyword arguments of `nn.Conv3d`.

#### Constraints and Behavior

- `use_custom` is enabled by default when the `torch.npu` interface is detected; tests may force `module.use_custom = True` to drive the custom path on CPU.
- Arguments accept only the supported types and values above; mathematical consistency applies only to this parameter domain and combinations covered by this repository.
- With `padding_mode != 'zeros'`, the input goes through `F.pad` first and then enters the custom Function with `conv_padding=0`; gradients propagate through the autograd of `F.pad`.
- The weight/bias state_dict keys interoperate with `nn.Conv3d`.
- dtype support depends on the underlying PyTorch backend and must be covered by the corresponding CPU / NPU tests in this repository before being listed as supported.
- Complex inputs are supported: the backward conjugates the flipped weight, and the complex weight gradient is recombined from real-valued `conv3d_weight` calls on the real and imaginary parts (`torch.nn.grad.conv3d_weight` itself takes real tensors only).
- For the specific large-kernel depthwise NPU shapes covered here, the step-by-step backward avoids an L1 buffer limit observed in the native backward. This observation is not a guarantee for every large kernel or arbitrary shape.

#### Verification

- Precision / performance cases: [`tests/composite/conv3d_custom/`](../../tests/composite/conv3d_custom/) (the listed parameter combinations, training call patterns, and performance baselines).
- Existing cases cover selected CPU float64/float32 combinations, specified NPU float32 shapes, and CPU complex128/complex64 plus NPU complex64 ([`test_conv3d_custom_complex.py`](../../tests/composite/conv3d_custom/test_conv3d_custom_complex.py), including double backward via `gradcheck` / `gradgradcheck` on CPU complex128). Conclusions apply only to cases that were actually executed and passed, not to unlisted parameters or data types.
- The complex NPU cases run with `torch.npu.conv.allow_hf32 = False` (restored afterwards). Under the default HF32 setting some gradients show a relative error of roughly `2e-4`, which is not accepted against the full fp32 threshold.

For implementation and usage notes see [`composite-torch/conv3d_custom/README.md`](../../composite-torch/conv3d_custom/README.md) (Chinese).
