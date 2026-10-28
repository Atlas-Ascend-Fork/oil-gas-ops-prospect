# Complex Exp Triton Ascend Kernel

Triton-Ascend complex64 exponential operator: `e^z` runs as a single Vector Core kernel
over the interleaved float32 view of the complex tensor, replacing the slow
`aclnnExp` (Ai_Cpu) path (measured 66ms vs 0.026ms for f32 exp at 4M elements —
three orders of magnitude degradation). Commonly used by seismic-inversion
workloads for phase factors `e^(a+i·b) = e^a·(cos b + i·sin b)`.

- `triton_complex_exp.py`: `complex_exp(z)` with autograd backward
  (`grad_z = g·conj(e^z)`, computed from the saved forward output without
  recomputing exp), CPU reference fallback, and the NPU Triton kernel.
- `tests/triton/complex_exp/test_complex_exp.py`: pytest coverage for CPU fallback
  (bit-exact against torch), NPU forward/backward precision, `cexpf` special values
  (pure-real overflow, the `exp(a/4)` overflow band, `±inf`/NaN), double backward,
  and the int64-offset branch.
- `tests/triton/complex_exp/verify_complex_exp.py`: NPU verification — forward
  accuracy against CPU float64, first- and second-order backward accuracy,
  wall-clock benchmarks with speedups, and a `--mode precision/perf` protocol for
  the `npu_python` suite.
- `tests/triton/complex_exp/report/`: recorded test environment, precision and
  performance results, and the raw `--json` output.

For the full mathematical definition, constraints, signatures and environment
variables see [`docs/en/op_list.md`](../../docs/en/op_list.md) /
[`docs/en/api_list.md`](../../docs/en/api_list.md) (English) and
[`docs/zh/op_list.md`](../../docs/zh/op_list.md) /
[`docs/zh/api_list.md`](../../docs/zh/api_list.md) (Chinese).

## Commands

```bash
# pytest coverage (bit-exact CPU fallback; NPU cases are skipped when none is present)
python3 -m pytest tests/triton/complex_exp/test_complex_exp.py -q

# full NPU verification (precision + performance)
python3 tests/triton/complex_exp/verify_complex_exp.py

# npu_python suite protocol (triggered by build.sh -u --precision/--perf via tests/suites.json)
python3 tests/triton/complex_exp/verify_complex_exp.py --mode precision --rtol 1e-5

# AOT compile gate (forward/backward, each with its int32 and int64 branches)
python3 triton-ascend/tools/aot_compile_check.py --filter complex_exp
```

Override the kernel tile size without touching code (power of two, clamped to
`[16, 2048]`):

```bash
export OIL_GAS_OPS_EXP_BLOCK=1024
```

## Usage in downstream code

```python
from oil_gas_ops_prospect.complex_exp import complex_exp

# elementwise e^z on complex64, autograd-ready
w = complex_exp(z)
```

Falls back to `torch.exp` on CPU (or when triton is absent); on NPU the Triton
kernel path requires complex64 inputs. Numerics follow `cexpf` (what
`torch.exp` on complex64 uses): a float32 `exp/cos/sin` composition, but with the
`cexpf` special cases reproduced — a pure-real overflow returns `inf+0j` rather
than `inf+nanj`, `b == 0` short-circuits to `exp(a) + i*b`, `a > ln(FLT_MAX)` uses
`exp(a/4)` multiplied four times so the real part can stay finite inside the
overflow band (`89+1j` → `2.43e38+infj`), and `b = ±inf`/NaN follows `cexpf`'s
`(a, b)` table.

Second order and higher derivatives are supported: the kernel produces only the
first-order value, so when `backward` itself runs in grad mode (`create_graph=True`
/ `gradcheck`) it evaluates the equivalent `g * conj(w)` with torch ops and lets
autograd differentiate again. Plain training still uses the backward kernel.

Known limitation: the triton-ascend backend's `tl.sin` / `tl.cos` do no argument
reduction once `|b| >= 2^25` (33554432), so results diverge from `torch.exp` from
that magnitude onwards. Phase angles in the targeted workloads are small; use
`torch.exp` when imaginary parts get that large.

## Measured results (910B2)

Forward, large shape `304x304x176` (16.3M complex64, wall-clock incl. host
dispatch): see `verify_complex_exp.py` output; the Ai_Cpu `aclnnExp` baseline
degrades ~2500x vs f32 while the Triton kernel restores full Vector Core
bandwidth (hundreds of GB/s class, matching `complex_reciprocal`).

Small-input overhead: the kernel pays a fixed host-side dispatch/launch cost of
~0.26 ms per call, so for very small tensors (roughly below ~3e4 complex64
elements) the plain `torch.exp` fallback can be faster (e.g. 13 elements:
0.26 ms vs 0.045 ms). The operator targets large complex64 tensors, where it
wins by 12x–160x; call `torch.exp` directly for tiny inputs if the extra
latency matters.
