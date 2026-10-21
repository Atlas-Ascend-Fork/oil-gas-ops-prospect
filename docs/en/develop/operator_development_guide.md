# Operator Development Guide

English | [简体中文](../../zh/develop/operator_development_guide.md)

This page explains how to add an AscendC / Triton-Ascend / Composite operator, and the directory, naming, wrapping and acceptance conventions you must follow. Paths follow [`dir_structure.md`](../context/dir_structure.md). For the Issue / PR process see the [Contribution Guide](../../../CONTRIBUTING.md) (Chinese).

The repository has three operator backends with different deliverables and development flows. Decide first which one your new operator belongs to:

| Scenario | Choice |
|----------|--------|
| needs fine-grained control of inter-core tiling, UB allocation and pipelines, chasing peak performance | AscendC operator → `ascendc/` |
| expressed quickly in Python, relies on autograd, shapes keep changing | Triton-Ascend operator → `triton-ascend/` |
| expressible by composing existing torch operators (including a custom forward/backward via `autograd.Function`), no custom kernel needed | Composite-Torch operator → `composite-torch/` |

---

## Part I. AscendC Operators

### 1. Directory Structure

```text
ascendc/operators/${op_name}/
├── op_host/
│   ├── ${op_name}.cpp                  # operator prototype (OpDef), InferShape, TilingFunc
│   ├── ${op_name}_tiling_data_def.h    # TilingData struct definition
│   ├── ${op_name}_tiling_core.h        # platform-independent tiling algorithm (header-only)
│   └── ${op_name}_tiling_impl.h        # implementation bound to gert::TilingContext
└── op_kernel/
    ├── ${op_name}.cpp                  # kernel entry
    ├── ${op_name}_impl.h               # kernel implementation
    └── ${op_name}_tiling.h             # kernel-side TilingData view
```

`operators/` holds implementations only. C++ tests and samples go into `tests/` and `examples/` in the repository root. Host / kernel headers shared by several operators go into `ascendc/operators/_common/` and are flattened into `op_host/` and `op_kernel/` when merged into the msopgen project.

A forward/backward pair may share one Python module, but **each takes its own line in the manifest and its own directory under `operators/`**. Shared logic goes into `_common/`.

### 2. Register the Operator

Append a line to `ascendc/operators/manifest.tsv` (**tab separated**: directory name, OpType, kernel entry file name). Confirm the registration with:

```bash
cd ascendc && bash build.sh list-ops
```

### 3. The Tiling Algorithm Must Be Testable

The entry signatures of `<Op>TilingFunc` / `InferShape` depend on `gert::TilingContext`, and `Compute<Op>Tiling` depends on `platform_ascendc::PlatformAscendC`. The CANN binary package ships no gert faker, so neither can be unit tested directly.

Hence the convention: **extract the platform-independent tiling algorithm into `op_host/<op>_tiling_core.h`**, so the UT drives the core functions directly and covers the real algorithm.

On the host side use `BEGIN_TILING_DATA_DEF` (required by opbuild); on the kernel side mirror the same layout with a **plain struct** (`ccec` does not understand that macro). `TilingFunc` is responsible for: shape / dtype validation → calling `*_tiling_core.h` → `SetBlockDim` / `SetTilingKey` → `SaveToBuffer`. A `GetCoreNumAiv()` of 0 must either fail or provide an explicit fallback, and that branch must be covered by the ophost UT.

The tiling strategy of the existing operators is a useful reference: split the total element count evenly across the AIV cores (32B aligned), then split into tiles within a core according to the UB budget.

### 4. Naming Conventions

| Side | Rule | Example |
|------|------|---------|
| C++ OpType | PascalCase | `ComplexMul`, `FusedBiasSoftmax` |
| aclnn | `aclnn` + OpType | `aclnnComplexMul` |
| kernel entry / `opFile.value` | snake_case, matching the source file name | `complex_mul` |
| public Python function | snake_case, **no** `npu_` prefix | `complex_mul`, `fused_bias_softmax` |
| backward OPP operator | `Grad` suffix | `ComplexMulGrad`, `FusedSoftmaxGrad` |
| environment variable | `OIL_GAS_OPS_` prefix | `OIL_GAS_OPS_COMPLEX_MUL_IMPL` |

Renaming the OpType does not affect the kernel sources (the entry is pinned by `opFile.value` / `opInterface.value`), but the aclnn symbol follows the OpType.

### 5. OpDef and the Kernel Entry

Use `ops::OpDef` + `OP_ADD`, not the legacy `REG_OP`. `opFile.value` / `opInterface.value` must equal the entry file name under `op_kernel/` (without the extension). When aclnn is needed, set `aclnnSupport.value = support_aclnn` (or follow what you measure for that operator). Element-wise float32/ND operators should reuse `DeclareRequiredFloatNdIo` / `AddAscend910bElementwiseConfig` from `_common/op_host/og_tiling_context.h`.

The entry symbol must match `opFile.value`. Ascend 910B vector instructions work at a granularity of **64 elements**: when `count` is not a multiple of 64, the tail may be skipped or written as 0. Compute with the aligned length, write out with the original `count` via `DataCopyPad`, and never write out of bounds. Plan UB in 32B units.

### 6. Python Wrapping

| File | Purpose |
|------|---------|
| `ascendc/pybind/<op_family>.py` | public API, packaged as `oil_gas_ops_prospect.<op_family>` |
| `ascendc/pybind/csrc/<op_family>/bindings.cpp` | calls the two-phase aclnn entry directly, built as `<op_family>_npu` |

Once you create `csrc/<op_family>/*.cpp`, `pybind_extensions.py` picks the extension up automatically; there is no need to hand-write a `CppExtension` in `setup.py`. Follow the existing `complex_mul.py`:

| Rule | Description |
|------|-------------|
| the functional API goes through `autograd.Function.apply` | calling `ext.forward()` directly breaks the gradient graph |
| an independent PyTorch reference is mandatory | `IMPL=torch` is the default and comparisons depend on it |
| implementation switch | `OIL_GAS_OPS_<OP>_IMPL=torch\|ascendc\|auto`, default `torch` |
| forcing the custom operator | `OIL_GAS_OPS_REQUIRE_ASCENDC_<OP>=1` forbids the silent fallback |
| cache the extension with `@lru_cache` | avoids repeated imports |
| public names snake_case, no `npu_` | `complex_mul`, not `npu_complex_mul` / `ComplexMul` |

A composed wrapper such as `attention_core` may live in the same module, but [api_list.md](../api_list.md) must state that it is "not a standalone OPP operator".

### 7. Add Unit Tests

The AscendC C++ tests live in the repository root; the operator directory keeps only the implementation. This repository has no ATK / `OP_API_UT`.

```text
tests/ut/op_host/${op_name}/test_${op_name}_tiling.cpp   # tiling UT (googletest)
tests/st/aclnn<OpType>/                                  # multi-case two-phase aclnn + golden (needs an NPU)
examples/aclnn/${op_name}/test_aclnn_${op_name}.cpp       # inline small tensor, two-phase aclnn (needs an NPU)
```

`tests/ascendc/ophost/CMakeLists.txt` collects `tests/ut/op_host/*/test_*.cpp` automatically
and adds the include path of each `op_host`. Python precision comparisons go into `tests/ascendc/${op_name}/`.
Register the ST / precision / sample entries in `tests/suites.json`.

The ophost UT should cover: UB budget constraints, alignment and upper bounds, small-tensor shrinking, degenerate inputs (negative element counts, `aivNum=0`, empty tensors), and the coverage invariants of the inter-core split (every element handled exactly once by one AIV core, with no gaps or overlaps between ranges). Fused operators additionally cover bias broadcast / full size, `Q≠S`, and VEC / MIX path dispatch.

### 8. Build and Verify

```bash
# packaging (the AscendC side produces only the .run)
cd ascendc && bash build.sh --pkg --ops ${op_name}

# repository root
bash build.sh --pkg      # OPP .run only
bash build.sh            # build the .run and the whl, no installation
bash build.sh --install  # build and install

# on a 910B: UT + ST + precision + performance (writes test_result/test_report.md)
bash build.sh -u --ut --st --precision --perf --ops=${op_name}
```

For the full option set see [build.md](../context/build.md).

### 9. NPU Precision/Performance Tests

Organize them under `tests/ascendc/${op_name}/` following the existing structure, always using an equivalent PyTorch/torch_npu expression as the baseline:

```text
ascendc/pybind/${op_name}.py                 # Python API → oil_gas_ops_prospect.${op_name}
ascendc/pybind/csrc/${op_name}/bindings.cpp  # C++: calls the operator through the custom aclnn API

tests/ascendc/${op_name}/
├── test_${op_name}_npu.py  # precision/performance comparison script
├── run_test.sh             # install the OPP package → build the extension → run the tests
└── README.md
```

`ascendc/pybind/${op_family}.py` becomes `oil_gas_ops_prospect.${op_family}` directly, and
the `*.cpp` files under `ascendc/pybind/csrc/<op_family>/` are compiled automatically into the extension module `<op_family>_npu`
with no build script changes. For details see [`ascendc/pybind/README.md`](../../../ascendc/pybind/README.md) (Chinese).

The precision standard is the [Accuracy Evaluation Standard for Open-Source Ecosystem Operators](https://gitcode.com/cann/opbase/blob/master/docs/en/ops_precision_standard/experimental_standard.md). Acceptance thresholds for the operators already delivered here:

| Operator | dtype | Threshold | Source |
|----------|-------|-----------|--------|
| `ComplexMul` / `Grad` | fp32 | relative error, default `1e-5` | `tests/ascendc/complex_mul/` |
| `FusedBiasSoftmax` / `FusedSoftmaxGrad` | fp32 | `max_diff ≤ 1e-5` | `tests/ascendc/fused_bias_softmax/` |
| same as above | bf16 | `atol=2e-2, rtol=4e-3` | same as above |

A new operator must be documented in the PR using the [Precision Acceptance Report Template](precision_acceptance_template.md), with a 100% pass rate. Critical or major open issues must be closed before merging. Precision, performance and device memory on the NPU must be verified on a real 910B.

### 10. Notes on Porting CANN Framework Operators

Host tiling/infershape code ported from the original CANN repositories often depends on CANN internal framework headers (`op_host/tiling_base.h`, `op_host/tiling_templates_registry.h`, `err/ops_err.h`, `log/log.h` and so on). Those headers are missing, or pull in a slog dependency, in the opbuild environment of an msopgen custom operator project. How to handle it (keeping source changes minimal):

- provide a local compat shim that replaces `OP_CHECK_IF`, `OP_CHECK_NULL_WITH_CONTEXT`, `OPS_REPORT_*`, `OP_LOG*`, `ASCENDC_EXTERN_C` and similar macros with equivalent or degraded implementations;
- replace only the unavailable includes with that shim, leaving the tiling algorithm itself untouched;
- if an operator really needs extra CANN header directories, append them to `_EXTRA_INCLUDE_SUBDIRS` in `ascendc/scripts/patch_op_host.py`.

For more detail see [`ascendc/README.md`](../../../ascendc/README.md) (Chinese).

The current delivery and on-device verification target **Ascend 910B** (`Ascend910B1` by default). A new SOC must be added to `SUPPORTED_SOC_VERSIONS`, given a matching entry in `AICore().AddConfig(...)` of the OpDef, and pass precision acceptance on that SOC.

---

## Part II. Triton-Ascend Operators

### 1. Directory Structure

```text
triton-ascend/${op_family}/
├── __init__.py                         # public exports
├── triton_${op_family}.py              # Triton kernels and autograd wrappers
└── README.md                           # operator notes

tests/triton/${op_family}/
└── test_triton_${op_family}.py         # precision test for the CPU fallback path
```

### 2. A CPU Fallback Is Mandatory

The kernel must provide a **PyTorch fallback implementation** with the same interface semantics as the NPU path. The existing `real_rfft` is a useful reference:

```python
def real_rfft(x, dim, norm="ortho", ...):
    if norm != "ortho":
        raise ValueError(f"only norm='ortho' is supported, got {norm!r}")
    if not (_is_npu_tensor(x) and _has_triton()):
        return _torch_rfft_real(x, dim=dim, norm=norm)
    return _RealRFFTFunction.apply(x, dim, norm, ...)
```

### 3. Register It in the whl

Register the new subpackage in `packages` and `package_dir` of `setup.py` so that it lands in the whl:

```python
packages=[
    ...,
    "oil_gas_ops_prospect.${op_family}",
],
package_dir={
    ...,
    "oil_gas_ops_prospect.${op_family}": "triton-ascend/${op_family}",
},
```

Do not add a `triton` / `ascendc` intermediate package. When running straight from the in-repo sources with no whl installed, `scripts/local_pkg.py` resolves the import name; when validating a whl, set `OIL_GAS_OPS_WHL_TEST=1`.

### 4. Include It in the AOT Compile Check

New kernel branches must be reachable by `triton-ascend/tools/aot_compile_check.py` so that every branch is known to compile (TTIR → Linalg → bishengir → npubin).

Write the constraints and mathematical definition into [op_list.md](../op_list.md) and the signature into [api_list.md](../api_list.md).

### 5. Switching Between Local Sources and the whl

The verification scripts use the in-repo sources directly by default (`scripts/local_pkg.py` maps each operator directory onto
`oil_gas_ops_prospect.<operator name>`), so changing an operator does not require rebuilding the whl:

```bash
python3 tests/triton/real_fft/verify_triton_fft_real_128_3d.py --print-shape-only
```

To validate the **installed whl**, set `OIL_GAS_OPS_WHL_TEST=1` so the in-repo sources are not tested by accident:

```bash
bash tests/run_whl_tests.sh
```

---

## Part III. Composite-Torch Operators (Pure PyTorch Composition)

Operators implemented purely by composing existing torch operators (`torch.autograd.Function` organizes the forward/backward, with no custom kernel at all) go under `composite-torch/`, a sibling of `triton-ascend/`.

### 1. Directory Structure

```text
composite-torch/${op_family}/
├── __init__.py                         # public exports
├── ${op_family}.py                     # torch.autograd.Function wrapper + public API
└── README.md                           # operator notes

tests/composite/${op_family}/
└── test_${op_family}.py                # pytest cases
```

`composite-torch/` is itself not a package directory (same as `triton-ascend/`, and the test directory takes the short name `tests/composite/` just like `tests/triton/`). A simple implementation may collapse into a single top-level file `composite-torch/${op_family}.py`, or follow torch conventions and split the module class from the functional entry inside a package (as `conv3d_custom/` does with `conv3d.py` + `functional.py` mapping to `nn.Conv3d` / `F.conv3d`).

### 2. Register It in the whl

Same as the Triton flow: register by operator name in `packages` and `package_dir` of `setup.py`:

```python
packages=[
    ...,
    "oil_gas_ops_prospect.${op_family}",
],
package_dir={
    ...,
    "oil_gas_ops_prospect.${op_family}": "composite-torch/${op_family}",
},
```

Do not add an intermediate backend package. When running straight from the in-repo sources with no whl installed, `scripts/local_pkg.py` **discovers** the operators under `composite-torch/` automatically (a subdirectory containing `__init__.py`, or a single top-level `.py` file), so no manual registration is needed; when validating a whl, set `OIL_GAS_OPS_WHL_TEST=1`.

### 3. Wrapping Conventions

| Rule | Description |
|------|-------------|
| the functional API goes through `autograd.Function.apply` | with a custom backward, composing the calls directly breaks the gradient graph |
| public names snake_case, no `npu_` prefix | consistent with the other tracks |
| no dependency on a custom extension | pure torch composition; if a kernel is needed, use the AscendC / Triton track |

Write the constraints and mathematical definition into [op_list.md](../op_list.md) and the signature into [api_list.md](../api_list.md).

---

## Part IV. Pre-Submission Self-Check

```bash
bash build.sh --install
bash build.sh -u
bash scripts/check_license_header.sh
```

New source files need the Apache License 2.0 header; follow the format of the existing files. When operator semantics or constraints change, update the corresponding [op_list.md](../op_list.md) and [api_list.md](../api_list.md) pages under `docs/zh/` and `docs/en/` as a pair, together with the matching `README.md`.

### Deliverable Checklist

| Category | Path |
|----------|------|
| Host / Kernel | `ascendc/operators/<op>/op_host/`, `op_kernel/` (including `*_tiling_core.h`) |
| registry | a new line in `ascendc/operators/manifest.tsv` |
| Python API | `ascendc/pybind/<family>.py` (snake_case public name, autograd, torch reference, `OIL_GAS_OPS_*` switches) |
| aclnn bindings | `ascendc/pybind/csrc/<family>/bindings.cpp` |
| ophost UT | `tests/ut/op_host/<op>/test_<op>_tiling.cpp` |
| aclnn example / ST | `examples/aclnn/<op>/`, `tests/st/aclnn<OpType>/`; registered in `tests/suites.json` |
| NPU comparison | `tests/ascendc/<op>/` (or `tests/triton/<family>/` for Triton) |
| documentation | `docs/zh/op_list.md`, `api_list.md` and their English counterparts under `docs/en/`; AscendC also needs `ascendc/README.md` |
| precision report | attach the [template](precision_acceptance_template.md) to the PR description |
| Triton registration | `packages` / `package_dir` in `setup.py`; covered by the AOT check |
| Composite-Torch registration | `composite-torch/<family>/` (pure Python, `packages` / `package_dir` in `setup.py`; tests under `tests/composite/<family>/`) |

Do not create `tests/`, `examples/`, `testplan/`, `teste2e/` or `delivery/` under `operators/<op>/`. `output/`, `ascendc/_work/`, `build/`, `*.so` and similar are already in `.gitignore` and must not be committed.
