# Real FFT Triton Ascend Kernels

Triton-Ascend real-valued FFT kernels: the DFT basis is applied as `tl.dot` GEMMs so the
transform runs on the cube unit. Used by spectral operators in seismic imaging and
inversion workloads.

- `triton_fft_real.py`: real-input `rfft` and real-output `irfft` wrappers with CPU reference fallback and NPU Triton kernels.
- `verify_triton_fft_real_128_3d.py`: three-way verification/benchmark helper for `torch.fft` on CPU, native `torch-npu`, and fused Triton NPU kernels.
- `test_triton_fft_real.py`: pytest coverage for CPU fallback, Fourier basis correctness, and opt-in NPU tests.

## Commands

在仓库根目录 `oil-gas-ops-prospect/` 下执行（容器内一般为 `/root/projects/oil-gas-ops-prospect`）。

Run the default pytest coverage. This covers CPU fallback and Fourier basis correctness, and skips NPU-only cases when NPU is unavailable:

```bash
cd /root/projects/oil-gas-ops-prospect
python -m pytest triton-ascend/real_fft/test_triton_fft_real.py -q
```

Two cases cross-check these kernels against the downstream model code they replace.
This operator library is domain-scoped and ships no model code, so both are skipped
by default. A downstream project opts in:

```bash
export OIL_GAS_OPS_MODEL_REPO=/path/to/model-repo      # enables the cross-checks
export OIL_GAS_OPS_MODEL_CONFIG=/abs/path/config.yaml  # config for shape inference
export OIL_GAS_OPS_FFT_BACKBONE=my_pkg.backbone:MyNet  # optional, overrides the default
```

Run pytest with the opt-in 128^3 NPU backward and memory test:

```bash
cd /root/projects/oil-gas-ops-prospect
OIL_GAS_OPS_RUN_128_FFT_TEST=1 \
python -m pytest triton-ascend/real_fft/test_triton_fft_real.py -s -q
```

Use `-s` when you want pytest to show the intermediate precision, memory, and latency lines while the test is running. Without `-s`, pytest may capture stdout and only print the final pass/fail summary.

Print the shape a run will use. Give it directly with `--shape` / `--axes` — this is
the standalone form and needs no model config:

```bash
python triton-ascend/real_fft/verify_triton_fft_real_128_3d.py \
  --shape 1 41 41 41 2048 --axes 1 \
  --print-shape-only
```

Two built-in shape presets cover the profiled spectral workloads:

```bash
# profiled 3D spectral latent tensor (2, 41, 41, 41, 2048) on axis 1
python triton-ascend/real_fft/verify_triton_fft_real_128_3d.py \
  --run-latent-3d-profiled --print-shape-only

# 128^3 channels-first spectral-conv smoke tensor (1, 35, 128, 128, 128) on axes 4/3/2
python triton-ascend/real_fft/verify_triton_fft_real_128_3d.py \
  --run-spectral-3d-smoke --print-shape-only
```

Run full three-way verification for a preset shape. This checks `rfft` and `irfft` forward/backward precision, latency, and peak memory across `torch_cpu`, `torch_npu`, and `fused_npu`:

```bash
python triton-ascend/real_fft/verify_triton_fft_real_128_3d.py \
  --run-latent-3d-profiled \
  --device-id 0 \
  --rounds 3 \
  --output tmp/fft_latent_3d_profiled.json
```

Each round prints intermediate comparison results directly:

```text
axis=1 round=0
  torch_cpu: elapsed_ms=...
  torch_npu vs torch_cpu: rfft_fwd_max=(real,imag) rfft_bwd_max=... irfft_max=(fwd,bwd_real,bwd_imag) peak_mb=... elapsed_ms=...
  fused_npu vs torch_cpu: rfft_fwd_max=(real,imag) rfft_bwd_max=... irfft_max=(fwd,bwd_real,bwd_imag) peak_mb=... elapsed_ms=...
  fused_npu vs torch_npu: rfft_fwd_max=(real,imag) rfft_bwd_max=... irfft_max=(fwd,bwd_real,bwd_imag) speedup=...x memory_delta_mb=...
```

Run full NPU verification with an explicit 128^3 channels-last input shape:

```bash
python triton-ascend/real_fft/verify_triton_fft_real_128_3d.py \
  --shape 1 128 128 128 1 \
  --axes 1 2 3 \
  --device-id 0 \
  --rounds 3 \
  --output tmp/fft_128_3d.json
```

Infer the shape from a downstream model YAML config instead of passing it explicitly.
The config belongs to the downstream project, so pass an absolute path (or set
`OIL_GAS_OPS_MODEL_REPO` and use a path relative to it):

```bash
python triton-ascend/real_fft/verify_triton_fft_real_128_3d.py \
  --from-model-config /abs/path/to/model-config.yaml \
  --device-id 0 \
  --rounds 3 \
  --output tmp/fft_from_model_config.json
```
