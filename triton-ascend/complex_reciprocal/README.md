# Complex Reciprocal Triton Ascend Kernel

Triton-Ascend complex64 reciprocal operator: `1/z` runs as a single Vector Core kernel
over the interleaved float32 view of the complex tensor, replacing the slow and unstable
`aclnnReciprocal` (Ai_Cpu) path. Commonly used by seismic-inversion workloads where the
denominator has the form `real tensor + constant imaginary part`.

- `triton_complex_reciprocal.py`: `complex_reciprocal(z)` and the two-argument form
  `complex_reciprocal(xr, xi)` (real/imag-separated), with autograd backward,
  stable/naive numeric paths, CPU reference fallback, and NPU Triton kernels.
- `tests/triton/complex_reciprocal/verify_complex_reciprocal.py`: five-way accuracy comparison
  against CPU float64 and wall-clock benchmarks (torch `1/z` baseline, plan-A float32
  decomposition, Triton kernels), plus the fused real/imag-separated end-to-end comparison
  and a `block_size` sweep.
- `tests/triton/complex_reciprocal/test_complex_reciprocal.py`: pytest coverage for CPU
  fallback, NPU forward/backward precision, extreme magnitudes, int64-offset branch, and
  the fused tensor+scalar expression.

## Commands

在仓库根目录 `oil-gas-ops-prospect/` 下执行（容器内一般为 `/root/projects/oil-gas-ops-prospect`）。

Run the default pytest coverage. This covers CPU fallback (bit-exact against torch) and
skips NPU-only cases when NPU is unavailable:

```bash
cd /root/projects/oil-gas-ops-prospect
python3 -m pytest tests/triton/complex_reciprocal/test_complex_reciprocal.py -q
```

Run full NPU verification: forward accuracy against CPU float64 across shapes and value
distributions (randn / shifted / tiny / huge), backward accuracy, wall-clock benchmarks
with speedups, and the fused end-to-end comparison:

```bash
python3 tests/triton/complex_reciprocal/verify_complex_reciprocal.py
```

Each section prints a comparison table directly:

```text
shape                 分布     方法                     max_abs     max_rel
-----------------------------------------------------------------------
304x304x176           randn  triton stable        2.987e-04   2.619e-07
...
304x304x176           torch 1/z             264.9385      1.0      1.0x
304x304x176           方案A 拆解                 45.2061      5.8      5.9x
304x304x176           triton stable           0.6749    385.6    392.6x
```

Add the large shapes and the `block_size` sweep, and persist results as JSON:

```bash
python3 tests/triton/complex_reciprocal/verify_complex_reciprocal.py \
  --full \
  --block-sweep \
  --json tmp/recip_report.json
```

Override the kernel tile size without touching code (power of two, clamped to
`[16, 2048]`; larger tiles overflow the Vector Core unified buffer at runtime):

```bash
export OIL_GAS_OPS_RECIP_BLOCK=1024
python3 tests/triton/complex_reciprocal/verify_complex_reciprocal.py
```

Run the no-NPU AOT compile gate (checks every kernel constexpr branch compiles through
TTIR → bishengir → npubin):

```bash
python triton-ascend/tools/aot_compile_check.py --filter complex_reciprocal
```

## Usage in downstream code

```python
from oil_gas_ops_prospect.complex_reciprocal import complex_reciprocal

# generic elementwise 1/z on complex64, autograd-ready
g = complex_reciprocal(z)

# fused: 1 / (a - i*c) without building the complex denominator
g = complex_reciprocal(a, -c)
```

Both fall back to torch on CPU (or when triton is absent); on NPU the Triton kernel path
requires float32/complex64 inputs. The `stable=True` default keeps ~2ulp accuracy over the
full magnitude range where the naive `a^2+b^2` form overflows (|z| beyond ~[1e-19, 1.8e19]).

## Measured results (910B2)

Forward, large shape `304x304x176` (16.3M complex64, wall-clock incl. host dispatch):

| Method | Time | Effective BW | Speedup |
|--------|------|--------------|---------|
| torch `1/z` (aclnnReciprocal, Ai_Cpu) | 264.9 ms | 2.0 GB/s | 1.0x |
| Plan-A float32 decomposition | 45.2 ms | 5.8 GB/s | 5.9x |
| `complex_reciprocal` (stable) | 0.675 ms | 386 GB/s | 392x |

Fused end-to-end: 471.7 ms → 0.730 ms (**646x**). Forward+backward runs at
478 GB/s combined. For tensors smaller than ~1M elements the Triton python launch overhead
(~0.3ms) makes the torch single-op path faster — the win is on large tensors.
