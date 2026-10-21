# Debugging and Profiling

English | [简体中文](../../zh/debug/op_debug_prof.md)

The AscendC operators in this repository run on the **AI Core** (`ComplexMul` / `ComplexMulGrad` are AIV; the bf16 path of `FusedSoftmaxGrad` is MIX). This page describes the entry points available here for locating problems, checking precision and collecting performance data.

Precision, performance and device memory must be verified on a real **910B (NPU)**:

```bash
bash build.sh -u
```

---

## Debugging (AI Core Operators)

When an operator fails to execute or produces wrong values, inspect the host log and the kernel intermediate results separately.

### 1. Host-Side Logs

After the program exits, host debug logs are by default in:

```text
$HOME/ascend/log/debug/plog/plog-pid_*.log
```

For the paths and categories see CANN 9.0 [Viewing Logs](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/maintenref/logreference/logreference_0002.html).

Enabling [`ASCEND_SLOG_PRINT_TO_STDOUT`](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0121.html) prints host application logs to stdout (`1` on, `0` off). Once enabled, logs are no longer written to log files:

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

For a more verbose level set [`ASCEND_GLOBAL_LOG_LEVEL`](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html) (it only affects debug logs: `0` DEBUG, `1` INFO, `2` WARNING, `3` ERROR):

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```

For the full list of environment variables see the [Environment Variable Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html).

On the Python path, enabling synchronous execution makes the traceback point at the real failure site:

```python
import torch_npu
torch_npu.npu.set_sync_debug_mode(1)
```

When an aclnn call fails, use [`aclGetRecentErrMsg`](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_0028.html) to get the most recent ACL error description (reading it clears it, so call it immediately after the failure). The `LaunchOp` of the sample `examples/aclnn/complex_mul/test_aclnn_complex_mul.cpp` currently only prints the return code; while investigating, change it to:

```cpp
auto ret = aclnnComplexMulGetWorkspaceSize(
    t.inT[0], t.inT[1], t.inT[2], t.inT[3], t.wrT, t.wiT, &workspaceSize, &executor);
if (ret != ACL_SUCCESS) {
    const char* msg = aclGetRecentErrMsg();
    printf("aclnnComplexMulGetWorkspaceSize failed. ERROR: %d\n%s\n",
           ret, msg == nullptr ? "" : msg);
    return ret;
}
```

The header is already available: `examples/_common/aclnn_example.h` includes `acl/acl.h`.

### 2. Kernel Debugging

`AscendC::printf` / `DumpTensor` slow the kernel down, so enable them only while debugging. For the API details see [printf](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0193.html) and [DumpTensor](https://www.hiascend.com/document/detail/en/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0192.html).

The samples here call `aclInit(nullptr)` (see `Init` in `examples/_common/aclnn_example.h`), which does not export kernel dumps. To debug, place an `acl.json` next to the executable and change `aclInit(nullptr)` to `aclInit("acl.json")`:

```json
{
  "dump": {
    "dump_kernel_data": "printf,tensor",
    "dump_path": "./kernel_dump"
  }
}
```

Output path priority: `ASCEND_DUMP_PATH` > `ASCEND_WORK_PATH` > the `dump_path` in the configuration. The exported bin files are not readable as text; parse them with [show_kernel_debug_data](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0102.html). The `printf` content ends up in `parser.log` and `DumpTensor` output in `dump_data/`.

- printf: prints scalars. In `ComplexMul`, right after the tiling data is read (the `complex_mul` entry in `op_kernel/complex_mul.cpp`), you can temporarily add:

```c++
ComplexMulTilingData tilingData;
OgOpsKernel::CopyTilingData(&tilingData, tiling);
AscendC::printf("totalElems=%ld tileFloats=%ld coreNum=%ld blockIdx=%d\n",
                tilingData.totalElems, tilingData.tileFloats, tilingData.coreNum,
                static_cast<int>(GetBlockIdx()));
```

- DumpTensor: prints a LocalTensor in UB. In `ComplexMulKernel::ProcessTile`, before `wr` is written out (`op_kernel/complex_mul_impl.h`), you can temporarily add the following. The third argument is the element count; keep the per-core dump total below 1MB:

```c++
out = outQue_.DeQue<float>();
const uint32_t dumpN = static_cast<uint32_t>(cur < 32 ? cur : 32);
DumpTensor(out, 0, dumpN);
CopyOutPad(wrGm_[gmOff], out, cur);
```

For hangs, GM / UB out-of-bounds access and similar cases, [msDebug](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900beta2/devaids/optool/atlasopdev_16_0062.html) offers step-by-step debugging.

---

## Common Issues in This Repository

### The Custom Operator Is Not Called

Symptom: `OIL_GAS_OPS_COMPLEX_MUL_IMPL=ascendc` (or `OIL_GAS_OPS_FUSED_BIAS_SOFTMAX_IMPL=ascendc` for the fused operators) is set, but the results or the timings still look like the PyTorch reference implementation.

Cause: a failed AscendC call falls back silently by default. To forbid the fallback:

```bash
export OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL=1
# fused attention operators:
export OIL_GAS_OPS_REQUIRE_ASCENDC_FUSED_BIAS_SOFTMAX=1
```

For the complete set of environment variables see the [API List](../api_list.md#environment-variables).

| Root cause | How to check |
|------------|--------------|
| the pybind extension is not built or not installed | `python3 -c "import complex_mul_npu"` (`fused_bias_softmax_npu` for the fused operators) |
| the OPP package is not installed | `ls ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib` |
| the shared library path is missing | `echo $LD_LIBRARY_PATH`, it must contain `opp/vendors/customize/op_api/lib` |
| the tensors are not on the NPU | `x.device.type == "npu"` |
| dtype / shape violates an operator constraint | see the [Operator List](../op_list.md) |

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
```

### The aclnn API Cannot Be Found

The aclnn name of a custom operator follows the OpType. Here `ComplexMul` maps to `aclnnComplexMul`. Changing the OpType requires updating the bindings under `ascendc/pybind/csrc/` accordingly.

### The `.run` Package Has No Effect After Installation

The OPP installs into `opp/vendors/<vendor_name>/`, `customize` by default. With multiple vendors, make sure `opp/vendors/config.ini` contains `customize`.

```bash
bash output/Ascend910B1/oil_gas_ops_prospect_opp_*.run --quiet
```

### Build Error: Missing CANN Internal Framework Headers

Host code ported from the original CANN repositories often depends on `op_host/tiling_base.h`, `err/ops_err.h`, `log/log.h` and similar headers, which do not exist in an msopgen custom operator project. For how to handle this see [Operator Development Guide · Notes on Porting CANN Framework Operators](../develop/operator_development_guide.md#10-notes-on-porting-cann-framework-operators).

### A Triton Operator Runs on CPU Without Any Error

`real_rfft` / `real_irfft` fall back to `torch.fft`, and `complex_reciprocal` falls back to torch `1 / z`, when not on an NPU or when Triton is unavailable. To confirm that the NPU kernel is used, check the tensor device and whether `triton-ascend` is installed. `complex_reciprocal` also falls back for any dtype other than `complex64` / `float32`.

---

## Performance Collection

When latency or memory looks wrong, use `msprof op` to inspect kernel duration, core count and pipeline utilization. The commands below were run on a real 910B with CANN 9.0.0 using the `ComplexMul` sample of this repository. For other operators, replace the executable with `examples/aclnn/<op>/output/test_aclnn_<op>` and the simulator `--kernel-name` with that operator's OpType.

- **On-device collection**: kernel duration, block count and pipeline utilization on a real NPU.
- **Pipeline simulation**: per-core instruction pipelines; finer grained than on-device collection.

That sample uses a tiny `{2,2}` tensor, so the on-device `Task Duration` is only a few microseconds. It validates the collection path and must not be read as a business performance number. For representative shapes, use the Python `--perf` path at the end of this section.

For the meaning of the commands and result files see [Collecting Performance Data](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0008.html).

### Method 1: On-Device Performance Collection

Install the customize vendor first, then build the sample. `run.sh` wipes `build/` and `output/`, re-runs cmake/make and **immediately executes a functional check**:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
bash examples/aclnn/complex_mul/run.sh
```

Then collect performance data in the output directory (this is the second launch; 5 warm-up runs by default):

```bash
cd examples/aclnn/complex_mul/output
msprof op ./test_aclnn_complex_mul
```

The result directory is a sibling `OPPROF_<timestamp>_*`. Measured summary from this repository:

```text
Op Name: ComplexMul_923973e39f803f5bc6ddb804acabf320_0
Op Type: vector
Task Duration(us): 4.640093
Block Dim: 48
Device Id: 0
```

`Task Duration` is the duration of the current kernel and `Block Dim` is the number of cores used. The measured result directory (the timestamp and random suffix differ every time):

```text
OPPROF_<timestamp>_<random>/
├── OpBasicInfo.csv              # kernel name, Op Type, total duration, Block Dim, device ID
├── ArithmeticUtilization.csv    # per-block cube/vector compute utilization (for vector look at aiv_time / aiv_vec_ratio)
├── PipeUtilization.csv          # duration and ratio per pipeline (VEC / MTE2 / MTE3 / SCALAR)
├── Memory.csv                   # GM ↔ UB bandwidth and transfer volume
├── MemoryL0.csv                 # L0A/L0B/L0C bandwidth (mostly NA on pure vector cores)
├── MemoryUB.csv                 # UB read/write bandwidth
├── L2Cache.csv                  # L2 hit / miss
├── ResourceConflictRatio.csv    # bank / MTE conflicts and wait ratios
└── dump/
    ├── op_basic_info.txt        # text summary matching OpBasicInfo
    └── pc_start_addr.txt        # kernel PC start address
```

The bf16 path of `FusedSoftmaxGrad` is MIX, so its `Op Type` may not be pure vector; that operator has not been re-measured with `msprof op` yet.

Python precision / performance comparison (relative error and `ms/call`, including business shapes):

```bash
bash build.sh -u --precision --perf --ops=complex_mul
# or (tries to install the OPP .run under output/, then builds pybind and runs the comparison)
cd tests/ascendc/complex_mul && bash run_test.sh
```

### Method 2: Simulated Pipeline Collection

This repository defaults to `SOC_VERSION=Ascend910B1`. Before simulating, add the simulator library to `LD_LIBRARY_PATH` (adjust for your CANN installation path and SOC directory). Measured here, `--kernel-name=ComplexMul` matches the same kernel name as the on-device run, so there is no need to use the entry symbol `complex_mul`:

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/Ascend910B1/lib:${LD_LIBRARY_PATH}
cd examples/aclnn/complex_mul/output
msprof op simulator --output=$PWD/pipeline_auto --kernel-name=ComplexMul ./test_aclnn_complex_mul
```

Results land in `$PWD/pipeline_auto/OPPROF_*` (CANN 9.0.0 does not emit `visualize_data.bin`). Measured structure:

```text
pipeline_auto/OPPROF_<timestamp>_<random>/
├── dump/
│   ├── pc_start_addr.txt        # kernel PC start address
│   └── object_dump.txt          # simulation dump path and kernel name
└── simulator/
    ├── trace.json               # Chrome Tracing events for all cores (look here first)
    └── coreN.veccore0/          # one per vector core; N matches Block Dim (0..47 for this sample)
        ├── trace.json           # single-core pipeline timeline
        ├── coreN.veccore0_instr_exe.csv   # instruction level: pipeline, cycles, duration
        └── coreN.veccore0_code_exe.csv    # per-source-line cost; header only when not built with -g
```

Import `simulator/trace.json` into Chrome `chrome://tracing` or MindStudio Insight. To see which instruction a core is stuck on, open the matching `*_instr_exe.csv`.

---

## Performance Switches in This Repository

### Triton FFT Tiling

`OIL_GAS_OPS_FFT_BLOCK_M` controls the row tile size of the DFT matmul (default 512, bounded by the ≤128KB fp32 accumulator in the 910B L0C). The implementation shrinks it to fit and automatically retries with half the value on overflow:

```bash
for m in 128 256 512; do
  OIL_GAS_OPS_FFT_BLOCK_M=$m python3 tests/triton/real_fft/verify_triton_fft_real_128_3d.py
done
```

The `--block-m` flag of `verify_triton_fft_real_128_3d.py` overrides that environment variable.

### Triton Complex Reciprocal Tile

`OIL_GAS_OPS_RECIP_BLOCK` sets the number of complex elements per tile in `complex_reciprocal` (default 2048, rounded down to a power of two and clamped to `[16, 2048]`; larger tiles overflow the Vector Core UB). `verify_complex_reciprocal.py --block-sweep` sweeps the candidate values:

```bash
OIL_GAS_OPS_RECIP_BLOCK=1024 python3 tests/triton/complex_reciprocal/verify_complex_reciprocal.py
```

### AscendC Tiling

The tiling algorithm of each operator lives in `ascendc/operators/<op>/op_host/<op>_tiling_core.h`. After changing it, run the ophost UT first, then check performance on a 910B:

```bash
bash build.sh -u --ophost
bash build.sh -u --perf --ops=complex_mul
```
