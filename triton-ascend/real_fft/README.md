# Real FFT Triton Ascend Kernels

Triton-Ascend real-valued FFT kernels: the DFT basis is applied as `tl.dot` GEMMs so the
transform runs on the cube unit. Used by spectral operators in seismic imaging and
inversion workloads.

Python 接口：`oil_gas_ops_prospect.real_fft.real_rfft` / `real_irfft` / `real_rfft2` / `real_irfft2`
（另有内部辅助 `make_basis`）。

| 算子 | 数学定义（`norm="ortho"`） | 输入 → 输出 | 关键约束 |
|------|---------------------------|-------------|----------|
| `real_rfft` | \(X[k]=n^{-1/2}\sum_t x[t]e^{-2\pi i k t/n}\)，\(k=0,\ldots,n//2\) | `x` → `(xr, xi)`，`dim` 维变为 `n//2+1` | 只支持 `ortho`；NPU kernel 为 `float32`；非 NPU 回退 `torch.fft` |
| `real_irfft` | 与 `torch.fft.irfft(..., norm="ortho")` 等价，负频按共轭补全 | `(xr, xi), n` → `y` | `xr.shape == xi.shape`；`xr.shape[dim] == n//2+1` |
| `real_rfft2` | 与 `torch.fft.rfft2` 等价；由两个恒等式在一维 `real_rfft` 上组合 | `x` → `(xr, xi)`，`dim[1]` 变为 `w//2+1` | `norm` 支持 `backward`（默认）/ `ortho`；组合接口，无新 kernel |
| `real_irfft2` | 与 `torch.fft.irfft2` 等价，是 `real_rfft2` 的逆 | `(xr, xi), s` → `y` | `xr.shape == xi.shape`；`dim[1]` 为半谱轴 |

`real_rfft2` / `real_irfft2` 是**组合接口**（`triton_fft_real.py` 之外的新文件 `fft2d.py`），
不引入新的 Triton kernel：

- 正向先在 `dim[1]` 做一维 `real_rfft`（实输入），再在 `dim[0]` 用
  `FFT(a+ib)=FFT(a)+i·FFT(b)` 拆两次实数 FFT 得到全谱；
- 反向用 `IFFT(z)=conj(FFT(conj(z)))` 做完整复数逆，再沿 `dim[1]` 做一维 `real_irfft`；
- 一维算子只给半谱，进恒等式前按 `X[n-k]=conj(X[k])` 补全，补全长度为 `n-(n//2+1)`；
  沿末轴补全时先把该轴搬到 0 再做镜像——910B 上沿 stride-1 轴的 `flip` 比沿非末轴慢
  两个数量级（实测 11.1 ms → 0.54 ms），`movedim` 本身很便宜；
- 内部按 `ortho` 计算，`norm="backward"` 通过整体乘/除 `sqrt(h*w)` 换算；
- 非末轴且长度超过内部阈值（320）时先把轴搬到末轴，避免 strided kernel 编译失败；
  这层兜底对正向、反向都内置；
- 非 NPU / Triton 不可用 / 非 `float32` 回退 `torch.fft.rfft2` / `irfft2`。

### 什么时候值得用（910B2 实测）

判据是**行数** `rows = numel / (h*w)`，即两个变换轴之前的全部维度展平后的行数；
rank 本身不决定收益。`dim=(-2,-1)`、`s=(N,N)`、交替多轮取中位数：

| N \ rows | 1 | 4 | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|---|---|
| 144 | 0.28× | 0.26× | 0.25× | 0.26× | 0.26× | **1.04×** |
| 288 | 0.30× | 0.25× | 0.26× | 0.25× | **1.30×** | **1.94×** |
| 576 | 0.20× | 0.22× | 0.22× | 0.59× | **1.17×** | **1.16×** |

**约 256 行起反超；行数少时明显更慢**（`rows≤16` 时 0.20~0.30×，`rows=64` 时
0.26~0.59×）。只增加维度而不增加行数没有收益。

反超并非来自变换本身：在 `rows≤16` 的档位上原生 `torch.fft.rfft2` 快 3~5 倍，
变换规模不变而只是行数变多时原生才开始落后。差异来自**输出表示**：

| 项 | 原生 | 本接口 |
|---|---|---|
| 返回类型 | 一个 `complex64` 张量 | 两个 `float32` 实数张量 |
| 布局（910B 实测） | **非连续**，`stride=(…, 1, N)` | 连续，`stride=(…, w//2+1, 1)` |

910B 上「`complex64` × 非连续布局」的逐元素算子会退化，同 shape 下：

| 操作 | 耗时 |
|---|---|
| `complex64` 连续 × 连续 | 0.06 ms |
| `float32` 非连续 × 非连续 | 0.27 ms |
| **`complex64` 非连续 × 非连续** | **57.2 ms** |

单是 `complex64`、或单是非连续都不慢，叠加才慢约 200 倍。原生 `rfft2` 恰好同时给了这两点，
所以调用方一旦在频域做逐点乘加，成本就会盖过 FFT 本身。

**因此：如果调用方本来就要在频域做乘加，不换 FFT、只把复数乘改写成 `real`/`imag` 上的实数乘，
也能拿到同样的收益。** 本接口的价值是把「实数表示 + 二维变换 + 非末轴搬移兜底」打包成一个
入口，而不是发明了更快的 FFT。

行数落在收益区、且每步有多次二维变换时，相对仅用一维入口还有 1.6~1.8× 的额外收益
（二维入口一次调用取代正反向各两次一维调用，并省掉重复的搬轴）。

完整数学定义、输入输出与约束见 [`docs/zh/op_list.md`](../../docs/zh/op_list.md)；签名与环境变量见 [`docs/zh/api_list.md`](../../docs/zh/api_list.md)。

This directory holds the implementation only; all test cases live under the
repository-level `tests/` directory.

- `triton_fft_real.py`: real-input `rfft` and real-output `irfft` wrappers with CPU reference fallback and NPU Triton kernels.
- [`tests/triton/real_fft/verify_triton_fft_real_128_3d.py`](../../tests/triton/real_fft/verify_triton_fft_real_128_3d.py): three-way verification/benchmark helper for `torch.fft` on CPU, native `torch-npu`, and fused Triton NPU kernels.
- [`tests/triton/real_fft/test_triton_fft_real.py`](../../tests/triton/real_fft/test_triton_fft_real.py): pytest coverage for CPU fallback, Fourier basis correctness, and opt-in NPU tests.
- [`tests/triton/real_fft/test_triton_fft2d.py`](../../tests/triton/real_fft/test_triton_fft2d.py): 二维组合接口的代数正确性（可替换一维原语，因此无需 NPU）、`s` 语义、autograd 与搬轴兜底。
- [`tests/triton/real_fft/verify_real_fft2.py`](../../tests/triton/real_fft/verify_real_fft2.py): 真机精度（对拍 float64 CPU）与性能（对拍 `torch.fft.rfft2` / `irfft2`）。

## Commands

在仓库根目录执行：

Run the default pytest coverage. This covers CPU fallback and Fourier basis correctness, and skips NPU-only cases when NPU is unavailable:

```bash
python3 -m pytest tests/triton/real_fft/test_triton_fft_real.py -q
python3 -m pytest tests/triton/real_fft/test_triton_fft2d.py -q     # 二维组合接口（CPU 可跑）
```

真机上验证二维接口：

```bash
python3 tests/triton/real_fft/verify_real_fft2.py --mode precision
python3 tests/triton/real_fft/verify_real_fft2.py --mode perf
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
OIL_GAS_OPS_RUN_128_FFT_TEST=1 \
python3 -m pytest tests/triton/real_fft/test_triton_fft_real.py -s -q
```

Use `-s` when you want pytest to show the intermediate precision, memory, and latency lines while the test is running. Without `-s`, pytest may capture stdout and only print the final pass/fail summary.

Print the shape a run will use. Give it directly with `--shape` / `--axes` — this is
the standalone form and needs no model config:

```bash
python tests/triton/real_fft/verify_triton_fft_real_128_3d.py \
  --shape 1 41 41 41 2048 --axes 1 \
  --print-shape-only
```

Two built-in shape presets cover the profiled spectral workloads:

```bash
# profiled 3D spectral latent tensor (2, 41, 41, 41, 2048) on axis 1
python tests/triton/real_fft/verify_triton_fft_real_128_3d.py \
  --run-latent-3d-profiled --print-shape-only

# 128^3 channels-first spectral-conv smoke tensor (1, 35, 128, 128, 128) on axes 4/3/2
python tests/triton/real_fft/verify_triton_fft_real_128_3d.py \
  --run-spectral-3d-smoke --print-shape-only
```

Run full three-way verification for a preset shape. This checks `rfft` and `irfft` forward/backward precision, latency, and peak memory across `torch_cpu`, `torch_npu`, and `fused_npu`:

```bash
python tests/triton/real_fft/verify_triton_fft_real_128_3d.py \
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
python tests/triton/real_fft/verify_triton_fft_real_128_3d.py \
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
python tests/triton/real_fft/verify_triton_fft_real_128_3d.py \
  --from-model-config /abs/path/to/model-config.yaml \
  --device-id 0 \
  --rounds 3 \
  --output tmp/fft_from_model_config.json
```
