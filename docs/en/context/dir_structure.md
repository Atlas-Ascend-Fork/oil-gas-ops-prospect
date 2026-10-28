# Directory Structure

English | [简体中文](../../zh/context/dir_structure.md)

```text
oil-gas-ops-prospect/
├── LICENSE                                     # license (Apache License 2.0)
├── README.md                                   # project entry document
├── CONTRIBUTING.md                             # contribution guide
├── SECURITY.md                                 # security statement (includes the public-address declaration table)
├── CHANGELOG.md                                # change log
├── OAT.xml                                     # open-source compliance scan configuration
├── Third_Party_Open_Source_Software_List.yaml  # third-party component list
├── Third_Party_Open_Source_Software_Notice     # third-party copyright and license notices
├── .gitignore                                  # Git ignore rules
├── .clang-format                               # C++ code format
├── .pre-commit-config.yaml                     # pre-commit hooks
├── build.sh                                    # top-level build entry (builds packages by default; --install installs; -u runs tests)
├── CMakeLists.txt                              # top-level CMake (ophost tiling UT)
├── version.cmake                               # version and dependency definitions
├── requirements.txt                            # Python dependency declaration
├── pyproject.toml                              # Python packaging metadata
├── setup.py                                    # Python packaging and pybind extension build
│
├── .gitcode/
│   └── workflows/                              # GitCode Action gating pipelines
│       ├── PR-pipeline_oil-gas-ops-prospect.yml #   gate orchestration
│       ├── compile_action.yml                  #   compilation (x86 / aarch64)
│       ├── llt_action.yml                      #   LLT (ophost UT + Triton precision + AOT compile)
│       └── staticcheck_action.yml              #   static checks
│
├── cmake/                                      # CMake modules
│   ├── security_flags.cmake                    #   secure compilation options
│   └── third_party/
│       └── googletest.cmake                    #   googletest dependency resolution (gitcode mirror)
│
├── ascendc/                                    # ★ AscendC operators (deliverable: CANN OPP .run package)
│   ├── README.md                               #   detailed AscendC-side notes
│   ├── build.sh                                #   AscendC-side entry (--pkg to package / -u --ophost to run UT)
│   ├── build.py                                #   packaging implementation (msopgen + merge + CustomOp)
│   ├── operators/
│   │   ├── manifest.tsv                        #   operator registry (tab separated)
│   │   ├── _common/                            #   host/kernel headers shared by several operators
│   │   ├── complex_mul/                        #   ComplexMul (op_host / op_kernel only)
│   │   │   ├── op_host/                        #     OpDef / InferShape / Tiling
│   │   │   └── op_kernel/                      #     kernel implementation
│   │   ├── complex_mul_grad/                   #   ComplexMulGrad
│   │   ├── fused_bias_softmax/                 #   FusedBiasSoftmax
│   │   └── fused_softmax_grad/                 #   FusedSoftmaxGrad
│   ├── pybind/                                 #   packaged as oil_gas_ops_prospect (modules exposed by operator name)
│   │   ├── __init__.py
│   │   ├── complex_mul.py                      #     ComplexMul Python API
│   │   ├── fused_bias_softmax.py               #     FusedBiasSoftmax / FusedSoftmaxGrad
│   │   └── csrc/                               #     <op_family>/bindings.cpp → <op_family>_npu
│   ├── pybind_extensions.py                    #   unified C++ extension definitions
│   ├── pybind_setup.py                         #   in-place build entry
│   ├── scripts/                                #   internal build-flow modules (imported by build.py)
│   └── _work/                                  #   temporary msopgen project (build output, not version controlled)
│
├── triton-ascend/                              # ★ Triton-Ascend operators (packaged per operator as oil_gas_ops_prospect.<op>)
│   ├── real_fft/                               #   real_rfft / real_irfft
│   │   ├── triton_fft_real.py                  #     Triton kernels and autograd wrappers
│   │   └── README.md
│   ├── complex_reciprocal/                     #   complex_reciprocal (either z or xr, xi)
│   │   ├── triton_complex_reciprocal.py        #     Triton kernel and autograd wrapper
│   │   └── README.md                           #     operator notes
│   ├── complex_exp/                            #   complex_exp
│   │   ├── triton_complex_exp.py               #     Triton kernel and autograd wrapper
│   │   └── README.md                           #     operator notes
│   └── tools/
│       └── aot_compile_check.py                #   AOT compilation check
│
├── composite-torch/                            # ★ Composite-Torch operators (pure PyTorch composition, packaged as oil_gas_ops_prospect.<op>)
│   ├── README.md                               #   track overview and integration conventions
│   └── conv3d_custom/                          #   Conv3d / conv3d (compatible with nn.Conv3d / F.conv3d within the supported parameter domain)
│       ├── __init__.py                         #     exports Conv3d
│       ├── conv3d.py                           #     Conv3d module (nn.Module form, NPU fallback)
│       ├── functional.py                       #     conv3d functional entry + autograd implementation
│       └── README.md                           #     operator notes
│
├── tests/                                       # ★ all tests (operators/ no longer holds tests)
│   ├── README.md                               #   test categories, suite directories and what a new operator must carry
│   ├── suites.json                             #   suite list for build.sh -u
│   ├── opensource_gates.py                     #   lightweight gates of the open-source report (documentation / packaging entries / installation traces / API smoke)
│   ├── _common/aclnn/                          #   shared ST scaffolding
│   ├── ut/op_host/<op>/                        #   tiling googletest
│   ├── st/aclnn<OpType>/                       #   multi-case two-phase aclnn
│   ├── conftest.py                             #   pytest collection configuration
│   ├── test_oil_gas_ops_prospect_whl.py        #   smoke case based on the installed whl
│   ├── run_whl_tests.sh                        #   build whl → install → run smoke
│   ├── ascendc/
│   │   ├── ophost/                             #   aggregated ophost_ut build
│   │   ├── complex_mul/                        #   NPU precision/performance comparison tests
│   │   └── fused_bias_softmax/                 #   fused attention NPU precision/performance comparison
│   ├── composite/conv3d_custom/                #   Conv3d / conv3d CPU precision and performance baseline
│   └── triton/
│       ├── real_fft/                           #   rFFT / iRFFT precision (includes the CPU fallback path)
│       ├── complex_reciprocal/                 #   complex reciprocal precision (CPU fallback + NPU verify)
│       └── complex_exp/                        #   complex exponential precision (CPU fallback + NPU verify + report)
│
├── examples/                                   # Python examples + C++ two-phase samples
│   ├── _common/                                #   aclnn sample scaffolding
│   ├── aclnn/<op>/                             #   test_aclnn_*.cpp
│   └── *.py
├── SKILLS/                                     # reusable operator development skills (optimization / diagnosis / repair), split by backend
├── scripts/                                    # helper scripts
│   ├── build_options.sh                        #   option parsing and sectioned help for the root build.sh
│   ├── cann_env.sh                             #   CANN root resolution and aclnn build environment (sourced by the aclnn runners)
│   ├── run_tests.py                            #   orchestrates UT/ST/precision/performance and writes the Markdown report
│   ├── opensource_report.py                    #   generates the open-source test report from the -u JSON
│   ├── check_license_header.sh                 #   license header check
│   └── local_pkg.py                            #   resolves the oil_gas_ops_prospect import name when running the in-repo sources with no whl installed
├── docs/                                       # documentation
│   ├── README.md                               #   documentation index (Chinese)
│   ├── README_en.md                            #   documentation index (English)
│   ├── QUICKSTART.md                           #   quick start (Chinese)
│   ├── QUICKSTART_en.md                        #   quick start (English)
│   ├── zh/                                     #   Chinese documents
│   │   ├── op_list.md                          #     operator list
│   │   ├── api_list.md                         #     API list
│   │   ├── context/                            #     background and environment
│   │   ├── develop/                            #     development guides
│   │   │   ├── operator_development_guide.md   #       steps, conventions and acceptance for new operators
│   │   │   └── precision_acceptance_template.md
│   │   ├── debug/                              #     debugging and profiling
│   │   └── invocation/                         #     invocation examples
│   └── en/                                     #   English documents (mirrors zh/)
└── output/                                     # unified artifact directory (build output, not version controlled)
    ├── <SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run
    └── wheels/oil_gas_ops_prospect-*.whl
```

## How the Three Operator Backends Divide the Work

The repository is split into three business directories by **deliverable form**, marked with ★:

| Directory | Backend | Deliverable | When to use |
|-----------|---------|-------------|-------------|
| `ascendc/` | AscendC | CANN OPP `.run` operator package exposing `aclnn` APIs | operators that need fine-grained control of tiling and pipelines and chase peak performance |
| `triton-ascend/` | Triton-Ascend | Triton kernels inside the Python whl | operators expressed quickly in Python, relying on autograd, with shifting shapes |
| `composite-torch/` | Composite-Torch (pure PyTorch) | composed modules inside the Python whl (`torch.autograd.Function`) | operators expressible by composing existing torch operators, with no custom kernel needed |

All three are exposed through the unified Python package `oil_gas_ops_prospect` by **operator name** (that directory does **not** exist in the source tree; `package_dir` in `setup.py` assembles it at packaging time, regardless of backend):

- `oil_gas_ops_prospect.complex_mul` → `ascendc/pybind/complex_mul.py`
- `oil_gas_ops_prospect.fused_bias_softmax` → `ascendc/pybind/fused_bias_softmax.py`
- `oil_gas_ops_prospect.real_fft` → `triton-ascend/real_fft/`
- `oil_gas_ops_prospect.complex_reciprocal` → `triton-ascend/complex_reciprocal/`
- `oil_gas_ops_prospect.complex_exp` → `triton-ascend/complex_exp/`
- `oil_gas_ops_prospect.conv3d_custom` → `composite-torch/conv3d_custom/`

When running straight from the in-repo sources with no whl installed, `scripts/local_pkg.py` (activated automatically by the examples and by the pytest `conftest.py`) maps those directories onto the same import name, so call sites are identical whether or not the package is installed. Operators under `composite-torch/` are discovered automatically by that script and need no manual registration. When validating an installed whl, set `OIL_GAS_OPS_WHL_TEST=1` (`tests/run_whl_tests.sh` already handles this) so the in-repo sources are not tested by accident.

## Artifacts and Temporary Directories

The following directories are produced by the build, are excluded in `.gitignore`, and are not version controlled:

| Directory | Contents |
|-----------|----------|
| `output/` | OPP `.run` operator packages and the Python whl |
| `ascendc/_work/` | temporary operator project generated by msopgen, ophost UT build directory |
| `build/`, `*.egg-info/` | Python extension build intermediates |
| `test_result/` | exported test reports |

`OIL_GAS_OPS_OUTPUT_DIR` overrides the location of `output/` and takes effect for `build.sh`, `ascendc/build.sh --pkg` and `tests/run_whl_tests.sh` alike.
