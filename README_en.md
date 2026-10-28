# oil-gas-ops-prospect

English | [简体中文](./README.md)

## 🚀 Overview

oil-gas-ops-prospect is an Ascend custom operator library for the **oil & gas exploration** domain. The repository name combines **oil-gas-ops** (oil & gas operators) with **prospect** (exploration target), targeting exploration workloads such as seismic imaging and full waveform inversion.

The repository is split into three operator backends by **deliverable form**; the public Python APIs are exposed by operator name, independent of the implementation backend:

| Backend | Directory | Deliverable | Use case |
|---------|-----------|-------------|----------|
| AscendC | [`ascendc/`](ascendc/) | CANN OPP `.run` operator package exposing `aclnn` APIs | operators that need fine-grained control of tiling and pipelines and require peak performance |
| Triton-Ascend | [`triton-ascend/`](triton-ascend/) | Triton kernels inside the Python whl | operators expressed quickly in Python, relying on autograd, with dynamic or frequently changing shapes |
| Composite-Torch | [`composite-torch/`](composite-torch/) | pure PyTorch composed modules inside the Python whl | operators expressible by composing existing torch operators, with no custom kernel needed |

## 📌 Version Compatibility

The following CANN version has been verified for this repository:

| CANN Version | Release Date |
|--------------|--------------|
| [9.0.0 official release](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0) | 2026/04/30 |

This repository requires the **Toolkit package**, the **Ops package** for your SoC, and the **Triton-Ascend** build matching your CANN version (`triton-ascend==3.2.1` is verified). It currently targets **Ascend 910B** (`Ascend910B1`). See [Environment Setup](docs/en/context/quick_install.md) for the complete installation steps, SoC support matrix, and environment variable configuration.

## ⬇️ Getting the Source

```bash
git clone https://gitcode.com/cann/oil-gas-ops-prospect.git
cd oil-gas-ops-prospect
```

## ⚡️ Quick Start

### Build and Install

The repository root offers three build entries.

**Build only the OPP `.run` package**

```bash
bash build.sh --pkg
```

Builds only the AscendC OPP `.run` into `output/<SOC>/`, without building the Python whl and without installing.

**Build without installing**

```bash
bash build.sh
```

Builds both the OPP `.run` and the Python whl into `output/`, without installing into CANN / Python and without running the smoke test. The wheel preferentially links the `libcust_opapi.so` generated in the current OPP build workspace, so the default top-level flow does not require the OPP to be installed first.

**Build and install**

```bash
bash build.sh --install
```

Builds the OPP and the whl, installs them into CANN / Python and runs the whl smoke test.

All artifacts land under `output/` in the repository root:

```text
output/
├── <SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run   # OPP operator package
└── wheels/oil_gas_ops_prospect-*.whl                # Python whl
```

After installation, later shells still need the custom opapi shared library path:

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
```

Commonly used options:

```bash
bash build.sh --help                                # list every option
bash build.sh --pkg -h                              # Package Options
bash build.sh -u -h                                 # Test Options
bash build.sh --clean                               # clean first, then build (combinable with the above)
bash build.sh --ops=complex_mul,complex_mul_grad    # build only the given operators
bash build.sh --install --skip-tests                # install and skip the whl smoke test
bash build.sh --soc=Ascend910B1 -j16                # pick SOC and parallelism
```

See [Build Options](docs/en/context/build.md) for the complete build entry and option reference.

### Using the APIs

```python
import torch
import torch_npu  # noqa: F401
from oil_gas_ops_prospect.real_fft import real_irfft, real_rfft

x = torch.randn(8, 256, device="npu")
xr, xi = real_rfft(x, dim=-1)          # real forward FFT, frequency axis length 256 // 2 + 1 = 129
y = real_irfft(xr, xi, n=256, dim=-1)  # reconstruct the real signal from the half spectrum
```

Runnable examples live in [`examples/`](examples/); for more invocation patterns see [Operator Invocation](docs/en/invocation/quick_op_invocation.md).

### Testing

For the categories see [`tests/README.md`](tests/README.md) (Chinese).

```bash
bash build.sh -u                                 # all suites (needs a 910B)
bash build.sh -u --ut                            # UT: tiling + small on-device launch
bash build.sh -u --st                            # ST: multi-shape aclnn
bash build.sh -u --precision                     # precision
bash build.sh -u --perf                          # performance

bash tests/run_whl_tests.sh                      # validate only the installed whl
```

Operator precision, performance and device memory on the NPU must be verified on a real 910B. For problem diagnosis see [Debugging and Profiling](docs/en/debug/op_debug_prof.md).

## 📖 Supported API List

The current delivery contains 6 AscendC operators, 5 Triton-Ascend operators and 1 Composite-Torch operator:

| Name / OpType | Backend / Deliverable | Function | dtype / format |
|---------------|-----------------------|----------|----------------|
| [`ComplexMul`](ascendc/operators/complex_mul/) | AscendC / OPP `.run` (AIV-only) | complex element-wise multiply: `(ur, ui), (vr, vi) -> (wr, wi)` | `float32` / `ND`, all tensors share a shape |
| [`ComplexMulGrad`](ascendc/operators/complex_mul_grad/) | AscendC / OPP `.run` (AIV-only) | backward of `ComplexMul` | `float32` / `ND`, all tensors share a shape |
| [`FusedBiasSoftmax`](ascendc/operators/fused_bias_softmax/) | AscendC / OPP `.run` | `softmax(x + bias1 + bias2, -1)`, in-place for contiguous inputs that successfully enter the fused path | `fp32` / `fp16` / `bf16`, `ND` |
| [`FusedSoftmaxGrad`](ascendc/operators/fused_softmax_grad/) | AscendC / OPP `.run` (may use MIX) | attention softmax JVP, overwriting contiguous probs that successfully enter the fused path | `fp32` / `fp16` / `bf16`, `ND` |
| [`Conv3DGelu`](ascendc/operators/conv3d_gelu/) | AscendC / OPP `.run` (MIX cube+vector) | `gelu_tanh(conv3d(x, weight, bias))`, without retaining the pre-activation between the two ops | `float32` / `NCDHW` |
| [`Conv3DGeluGrad`](ascendc/operators/conv3d_gelu_grad/) | AscendC / OPP `.run` (AIV + built-in conv3d backward) | recompute conv3d, then GELU backward, then conv3d backward | `float32` / `NCDHW` |
| [`real_rfft`](triton-ascend/real_fft/) | Triton-Ascend / Python whl | real forward FFT: `x -> (xr, xi)`, frequency axis length `n // 2 + 1` | the NPU kernel is `float32`; falls back to PyTorch outside an NPU environment |
| [`real_irfft`](triton-ascend/real_fft/) | Triton-Ascend / Python whl | reconstructs a real signal from the half-spectrum real and imaginary parts: `(xr, xi), n -> y` | the NPU kernel is `float32`; falls back to PyTorch outside an NPU environment |
| [`complex_reciprocal`](triton-ascend/complex_reciprocal/) | Triton-Ascend / Python whl | complex reciprocal `1/z` (either `z` or `xr, xi`, with a custom backward) | the NPU kernel is `complex64`; falls back to PyTorch outside an NPU environment |
| [`complex_exp`](triton-ascend/complex_exp/) | Triton-Ascend / Python whl | complex exponential `e^z` (custom backward) | the NPU kernel is `complex64`; falls back to PyTorch outside an NPU environment |
| [`interpolate`](triton-ascend/interpolate/) | Triton-Ascend / Python whl | 2D/3D linear interpolation; 5D fp32 uses Cube GEMM, 4D uses the native kernel | `float32` for 5D; 4D is supported by the native path |
| [`Conv3d`](composite-torch/conv3d_custom/) | Composite-Torch / Python whl | compatible with `nn.Conv3d` within the supported parameter domain (custom forward/backward, interoperable state_dict keys) | dtypes supported by the underlying PyTorch backend and tested here |

For `Conv3d` / `conv3d`, `kernel_size`, `stride`, numeric `padding`, and `dilation` accept an `int` or a 3-element sequence, and `groups` is an `int`; the module supports `zeros` / `reflect` / `replicate` / `circular`. The string padding values `"same"` / `"valid"` are supported (`"same"` requires a stride of 1), while the constructor does not provide the `device` / `dtype` keyword arguments of native `nn.Conv3d`.

For mathematical definitions, constraints and supported ranges see the [Operator List](docs/en/op_list.md); for full signatures, parameters and environment variables see the [API List](docs/en/api_list.md).

## 🔍 Directory Structure

```text
oil-gas-ops-prospect/
├── ascendc/                # AscendC operators (deliverable: CANN OPP .run package)
│   ├── operators/          #   operator sources + the manifest.tsv registry
│   ├── pybind/             #   Python API (packaged as oil_gas_ops_prospect) + C++ bindings
│   ├── scripts/            #   internal modules of the msopgen build flow
│   └── build.sh            #   AscendC-side entry (--pkg to package / -u --ophost to run UT)
├── triton-ascend/          # Triton-Ascend operators (packaged per operator as oil_gas_ops_prospect.<op>)
│   ├── real_fft/           #   real_rfft / real_irfft
│   ├── complex_reciprocal/ #   complex_reciprocal (polymorphic: z or xr, xi)
│   ├── complex_exp/        #   complex_exp
│   ├── interpolate/        #   interpolate (5D Cube GEMM, 4D native)
│   └── tools/              #   AOT compilation check
├── composite-torch/        # Composite-Torch operators (pure PyTorch composition, packaged as oil_gas_ops_prospect.<op>)
│   └── conv3d_custom/      #   Conv3d (compatible with nn.Conv3d within the supported parameter domain)
├── tests/                   # UT / ST / Python precision and performance; operators/ no longer holds tests
│   ├── ut/op_host/         #   tiling googletest
│   ├── st/aclnn<OpType>/   #   two-phase aclnn system tests
│   ├── ascendc/ophost/     #   aggregated ophost_ut build (googletest)
│   ├── ascendc/complex_mul/#   NPU precision/performance comparison
│   ├── ascendc/fused_bias_softmax/
│   ├── ascendc/conv3d_gelu/ #   fused conv3d and tanh-approximate GELU
│   ├── triton/real_fft/    #   rFFT / iRFFT precision (includes the CPU fallback path)
│   ├── triton/complex_reciprocal/ # complex reciprocal precision (CPU fallback + NPU verify)
│   ├── triton/complex_exp/ # complex exponential precision (CPU fallback + NPU verify)
│   ├── triton/interpolate/ #   2D/3D interpolation verification (CPU fallback + NPU verify + report)
│   └── composite/conv3d_custom/   # Conv3d / conv3d precision and performance baseline
├── examples/               # Python examples + C++ samples under examples/aclnn/<op>/
├── scripts/                # helper scripts
├── SKILLS/                 # reusable operator development skills (optimization / diagnosis / repair), split by backend
├── docs/                   # documentation
├── cmake/                  # CMake modules (third-party resolution, secure compilation options)
├── .gitcode/workflows/     # GitCode Action gating pipelines
├── build.sh                # top-level build entry (builds packages by default; --install installs; -u runs tests)
├── CMakeLists.txt          # top-level CMake (ophost tiling UT)
└── version.cmake           # version and dependency definitions
```

See [Directory Structure](docs/en/context/dir_structure.md) for a per-directory description of responsibilities.

## 🛠️ Development Guide

See the [Operator Development Guide](docs/en/develop/operator_development_guide.md) for the complete flow, directory conventions, naming conventions, and self-check steps for adding an AscendC / Triton-Ascend / Composite-Torch operator.

Pre-submission self-check:

```bash
bash build.sh -u                        # all test suites (needs a 910B)
bash scripts/check_license_header.sh    # license header check
```

## 💬 Related Information

- [Quick Start](docs/QUICKSTART_en.md)
- [Documentation Index](docs/README_en.md)
- [Directory Structure](docs/en/context/dir_structure.md)
- [Operator List](docs/en/op_list.md)
- [API List](docs/en/api_list.md)
- [Build Options](docs/en/context/build.md)
- [Operator Development Guide](docs/en/develop/operator_development_guide.md)
- [Contribution Guide](CONTRIBUTING.md) (Chinese)
- [Security Statement](SECURITY.md) (Chinese)
- [Changelog](CHANGELOG.md) (Chinese)
- [License](LICENSE) (Apache License 2.0)
- [SIG: oil-gas-engineering](https://gitcode.com/cann/community/tree/master/CANN/sigs/oil-gas-engineering)

---

PS: the features and documentation of this project are still evolving; follow the repository for the latest updates.

- **Feedback**: report issues via GitCode [Issues](https://gitcode.com/cann/oil-gas-ops-prospect/issues).
- **Community**: join the conversation through GitCode [Discussions](https://gitcode.com/cann/oil-gas-ops-prospect/discussions).
