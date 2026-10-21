# Build Options

English | [简体中文](../../zh/context/build.md)

## Overview

`build.sh` sits in the repository root. It compiles the sources into an installable OPP `.run` plus a Python whl, or runs tests by category and writes a report. Option parsing lives in [`scripts/build_options.sh`](../../../scripts/build_options.sh).

The script has two mutually exclusive paths; a single invocation follows exactly one of them:

| Path | How to enter | What it does |
|------|--------------|--------------|
| [Build](#build) | omit `-u` | builds packages into `output/`; installs only when `--install` is given |
| [Test Report](#test-report) | `-u` | no packaging, no installation; runs tests and writes reports |

`--pkg` and `--install` cannot be combined with `-u`, nor with each other. This repository has no `--opapi` / `--opgraph` / `--opkernel` options like ops-math.

## Usage

1. Configure environment variables

   For environment setup see [Environment Setup](quick_install.md). With the default installation path, as root:

   ```bash
   source /usr/local/Ascend/cann-9.0.0/set_env.sh
   ```

   When CANN is not in the default path:

   ```bash
   export ASCEND_HOME_PATH=/path/to/cann
   source ${ASCEND_HOME_PATH}/set_env.sh
   ```

2. View the help

   ```bash
   bash build.sh --help             # overview
   bash build.sh --pkg -h           # build section
   bash build.sh --install -h       # same as above
   bash build.sh -u -h              # test report section
   bash build.sh --clean -h         # clean section
   ```

   `--soc=SOC` / `--soc SOC`, `--ops=a,b` / `--ops a,b` and `-j16` / `-j 16` are all accepted. The two sections below describe every option; pick what fits your scenario.

---

## Build

Without `-u` the script takes the build path. The root `build.sh` is the main entry; `ascendc/build.sh` and the top-level CMake are covered in [Other Entries](#other-entries) at the end of this section.

### Build Command Format

Taking build-and-install as the example. In that scenario `--soc`, `--ops` and `-j` are optional; a sensible thread count speeds up compilation.

```bash
bash build.sh --install [--soc=${soc_version}] [--ops=${op_list}] [-j${n}]
```

### Three Entries

The three entries are mutually exclusive; choose one per invocation.

**Build only the OPP `.run` package**

```bash
bash build.sh --pkg
```

Calls `ascendc/build.sh --pkg` to produce the OPP `.run` under `output/<SOC>/`. Does not build the Python whl and does not install into CANN.

**Build without installing (default)**

```bash
bash build.sh
```

Builds the OPP `.run` and the Python whl into `output/`, without running the `.run` installer and without `pip install`. Top-level `bash build.sh` runs `build_opp_run` before `build_python_wheel`; `ascendc/pybind_extensions.py::_vendor_opapi` preferentially uses the headers under `ascendc/_work/CustomOp_*/build_out/autogen` and `ascendc/_work/CustomOp_*/build_out/op_host/libcust_opapi.so`, so the default top-level build does not require the OPP to be installed first. Only when `python3 setup.py bdist_wheel` is run outside the top-level flow and the workspace has no valid OPP artifacts does it fall back to the installed CANN vendor directory, which must then contain the required headers and library.

**Build and install**

```bash
bash build.sh --install
```

Builds the OPP and the whl, installs them into CANN / Python and runs the whl smoke test. After installation, later shells still need the custom opapi shared library path:

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
```

### Option Reference

`bash build.sh --help` lists every build option.

| Option | Required | Description |
|--------|----------|-------------|
| `--pkg` | optional | Build only the OPP `.run`; no whl, no installation. Cannot be used with `--install` or `-u`. |
| `--install` | optional | Build the `.run` and the whl, then install and run the smoke test. Cannot be used with `--pkg` or `-u`. |
| (default, none of the above) | — | Build the `.run` and the whl into `output/` without installing. |
| `--soc=${soc}` | optional | NPU model, default `Ascend910B1`. One model per build. |
| `--ops=${op_list}` | optional | AscendC operators to build, comma or space separated. Defaults to all. |
| `-j${n}`, `--jobs <N>` | optional | Number of build threads, default is the CPU core count (for example `-j16`). |
| `--clean` | optional | Clean `ascendc/_work` and `output/` before building, then continue with the selected entry. |
| `--skip-tests` | optional | Only meaningful with `--install`: install but skip the whl smoke test. |
| `-h`, `--help` | optional | Print help; combined with `--pkg` / `--install` / `--clean` it prints the matching section. |

### Examples

```bash
bash build.sh --pkg                                 # OPP .run only
bash build.sh                                       # .run and whl, no installation
bash build.sh --install                             # build and install
bash build.sh --clean                               # clean first, then build (still no installation)
bash build.sh --ops=complex_mul,complex_mul_grad    # build only the given operators
bash build.sh --install --skip-tests                # install and skip the smoke test
bash build.sh --soc=Ascend910B1 -j16 --install      # pick SOC and parallelism, then install
```

### Artifact Locations

All build artifacts land under `output/` in the repository root:

```text
output/
├── <SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run   # OPP operator package
└── wheels/oil_gas_ops_prospect-*.whl                # Python whl
```

Setting `OIL_GAS_OPS_OUTPUT_DIR` changes that location for `build.sh`, `ascendc/build.sh --pkg` and `tests/run_whl_tests.sh` alike.

| Deliverable | Name | Specification |
|-------------|------|---------------|
| Python distribution name | `oil-gas-ops-prospect` | PEP 508 |
| import package name | `oil_gas_ops_prospect` | PEP 8 |
| whl | `oil_gas_ops_prospect-<ver>-<py>-<abi>-<plat>.whl` | PEP 427 (`-`/`.` normalized to `_`) |
| OPP operator package | `oil_gas_ops_prospect_opp_<os>_<arch>.run` | CANN `<pkg>_opp_<os>_<arch>.run` |
| environment variable prefix | `OIL_GAS_OPS_` | — |

The CANN vendor directory remains the conventional `opp/vendors/customize/`.

To validate an already installed whl without rebuilding the AscendC operators:

```bash
bash tests/run_whl_tests.sh
```

### Other Entries

For day-to-day work the root `build.sh` is enough. The two entries below are for step-by-step debugging or for building only the ophost tiling unit tests. All commands below are run from the repository root.

**`ascendc/build.sh`**: the AscendC-side step entry. Packages the OPP `.run` only, or runs the ophost UT only.

```bash
bash ascendc/build.sh --pkg [options]        # compile and package the OPP .run
bash ascendc/build.sh -u --ophost [options]  # compile and run the ophost unit tests
bash ascendc/build.sh clean                  # clean ascendc/_work and output/
bash ascendc/build.sh list-ops               # list the operators in the manifest
```

| Option | Required | Description |
|--------|----------|-------------|
| `-v`, `--soc-version <SOC>` | optional | msopgen SOC model, default `Ascend910B1`; a comma-separated list is supported |
| `--ops <ops...>` | optional | operator directory names, repeatable or comma separated, defaults to all |
| `-j`, `--jobs <N>` | optional | build parallelism, default is the CPU core count |
| `-i`, `--install-path <path>` | optional | CANN installation path, auto-detected by default |
| `--vendor-name <name>` | optional | OPP installation directory `opp/vendors/<name>/`, default `customize` |
| `--package-name <name>` | optional | replaces the `custom` prefix in the `.run` name, default `oil_gas_ops_prospect` |

```bash
bash ascendc/build.sh --pkg -v Ascend910B1 --ops complex_mul complex_mul_grad -j 16
python3 ascendc/build.py -v Ascend910B1 --ops complex_mul complex_mul_grad   # equivalent Python entry
```

**Top-level CMake**: covers only the ophost-side tiling unit tests. The AscendC operators themselves are not built by the top-level CMake; they must go through msopgen project generation and then `bash build.sh --pkg` or `bash ascendc/build.sh --pkg`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

| CMake Option | Default | Description |
|--------------|---------|-------------|
| `BUILD_OPHOST_UT` | `ON` | whether to build the ophost tiling unit tests |
| `CMAKE_BUILD_TYPE` | `Release` | build type |
| `OIL_GAS_GTEST_VERSION` | `1.14.0` | googletest version |
| `OIL_GAS_GTEST_URL` | gitcode mirror URL | googletest source archive URL, used only when gtest is not found locally |

Versions and dependencies are centralized in [`version.cmake`](../../../version.cmake): `OIL_GAS_OPS_PROSPECT_VERSION` (must match `pyproject.toml` / `setup.py`), `OIL_GAS_OPS_PROSPECT_CANN_VERSION`, `OIL_GAS_OPS_PROSPECT_SOC_VERSIONS`, plus the build-time and run-time dependency variables.

---

## Test Report

`-u` enters test mode: no packaging, no installation. Categories and case paths are defined by [`tests/README.md`](../../../tests/README.md) (Chinese). This mode can write two kinds of artifacts:

| Artifact | When it is generated | Path |
|----------|----------------------|------|
| Execution summary | every `-u` run | `test_result/test_report.md` (plus a timestamped copy) and `test_result/test_results.json` |
| Open-source test report | with `--opensource-report` | `test_result/oil-gas-ops-prospect仓开源测试报告.md`, also overwriting the same-named file under `docs/zh` |

### Test Command Format

Without a subset flag everything runs (including the large FFT shapes and the open-source gate). `--ops`, `--report` and `--opensource-report` are optional.

```bash
bash build.sh -u [--ut|--st|--precision|--perf|...] [--ops=${op_list}] [--opensource-report]
```

Without an NPU, the on-device items of a full `-u` run are **SKIPPED** rather than failed. Explicitly passing `--ut` / `--st` / `--precision` / `--perf` / `--examples` without an NPU is a **FAIL**.

Subset switches such as `--ut` / `--st` / `--precision` / `--perf` enter test mode by themselves, so `bash build.sh --ut` is equivalent to `bash build.sh -u --ut`. The CI job `JOB_llt` uses cmake/ctest and does not call `build.sh -u`.

### Option Reference

`bash build.sh -u -h` prints the test section help.

| Option | Required | Category | Description |
|--------|----------|----------|-------------|
| `-u` | enters test mode | all | runs every category below when no subset is given |
| `--all` | optional | all | explicit full run, same as bare `-u` |
| `--ut` | optional | **UT** | UT-host (tiling) + UT-device (small on-device launch); needs an NPU |
| `--ophost` | optional | UT-host | runs only the tiling googletest |
| `--st` | optional | **ST** | multi-shape C++ aclnn + golden; needs an NPU |
| `--precision` | optional | **precision** | NPU Python comparison |
| `--perf` | optional | **performance** | NPU Python timing |
| `--examples` | optional | UT-device | runs only the small C++ launches; needs an NPU |
| `--triton` | optional | Triton CPU | CPU fallback pytest |
| `--composite` | optional | Composite CPU | CPU pytest of the pure PyTorch composed operators |
| `--aot` | optional | Triton AOT | kernel compilability check |
| `--ops=${op_list}` | optional | filter | applies to UT-device / ST / precision / performance |
| `--report=<path>` | optional | — | path of the Markdown execution summary |
| `--opensource-report[=path]` | optional | — | generates the open-source test report from this `-u` run's JSON; used alone it only reads the existing JSON and runs no tests |
| `--noexec` | optional | UT-host | compile only, do not execute |
| `--cov` | optional | UT-host | enables `--coverage`; adds `--cov` for Triton when pytest-cov is installed |
| `--gtest_filter=pattern` | optional | UT-host | filters gtest |
| `--clean` | optional | — | cleans `ascendc/_work` and `output/` before testing |

`--noexec` / `--cov` / `--gtest_filter` / `--report` are only valid together with test mode. `--opensource-report` cannot be combined with `--pkg` or `--install`.

### Examples

```bash
bash build.sh -u                                 # everything
bash build.sh -u --ut                            # UT (tiling + on-device)
bash build.sh -u --st --precision --perf
bash build.sh -u --ophost                        # tiling only
bash build.sh -u --perf --ops=complex_mul
bash build.sh -u --report=test_result/my.md      # custom execution summary path
bash build.sh -u --opensource-report             # run tests and generate the open-source test report
bash build.sh --opensource-report                # report only, from the last -u JSON
```

### Execution Summary

Every `bash build.sh -u` run writes:

```text
test_result/
├── test_report_<timestamp>.md    # summary of this run
├── test_report.md                # stable copy of the latest summary
└── test_results.json             # data source for the open-source test report
```

`--report=<path>` changes only the Markdown summary path, not the JSON location.

### Open-Source Test Report

The community-template open-source test report is **not** produced by a bare `-u` run; it requires an explicit `--opensource-report`.

| Command | Behavior |
|---------|----------|
| `bash build.sh -u --opensource-report` | runs the full test set (documentation / packaging entries / installation traces / API smoke, plus the large FFT shapes), then **generates** the report from this run's JSON |
| `bash build.sh --opensource-report` | reads the existing `test_result/test_results.json` only, runs no tests |

By default it writes `test_result/oil-gas-ops-prospect仓开源测试报告.md` and overwrites the same-named file under `docs/zh`. If anything FAILs, the conclusion is "release not recommended" and stale numbers are never carried over.

For the category table, suite directories and which tests a new operator must carry, see [`tests/README.md`](../../../tests/README.md) (Chinese).
