# 接口列表

对外接口统一由 Python 包 `oil_gas_ops_prospect` 暴露。源码树里没有这个目录：`setup.py` 在打包时把 `ascendc/pybind/` 与 `triton-ascend/real_fft/` 组装到同一包名下；裸 checkout 由 `scripts/local_pkg.py` 提供同一导入路径。

```text
oil_gas_ops_prospect
├── complex_mul           # complex_mul, torch_complex_mul, torch_complex_mul_grad
├── fused_bias_softmax    # fused_bias_softmax, fused_softmax_grad, attention_core, torch_*
└── real_fft              # real_rfft, real_irfft, make_basis
```

数学定义、shape / dtype 约束与精度阈值见[算子列表](op_list.md)。

---

## `oil_gas_ops_prospect.complex_mul`

### `complex_mul`

```python
from oil_gas_ops_prospect.complex_mul import complex_mul

def complex_mul(
    ur: torch.Tensor,
    ui: torch.Tensor,
    vr: torch.Tensor,
    vi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]
```

复数逐点乘 \((u_r, u_i) \cdot (v_r, v_i) \rightarrow (w_r, w_i)\)。`torch.autograd.Function` 封装，反向由 OPP 算子 `ComplexMulGrad` 承担；Python **没有** 单独的 `complex_mul_grad` 公开函数。

| 参数 | 类型 | 说明 |
|------|------|------|
| `ur`, `ui` | `torch.Tensor` | 第一个复数的实部、虚部。AscendC 路径要求 `float32`、同 shape、连续 |
| `vr`, `vi` | `torch.Tensor` | 第二个复数的实部、虚部，与 `ur` 同 shape |

返回 `(wr, wi)`，shape 与输入相同。

**实现选择**：默认走 PyTorch 参考实现。只有在张量位于 NPU 且 `OIL_GAS_OPS_COMPLEX_MUL_IMPL` 设为 `ascendc` 或 `auto` 时才调用 AscendC 融合算子。调用失败时默认静默回退到 PyTorch，除非设置了 `OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL=1`。`IMPL=ascendc` 且张量不在 NPU、同时要求自定义算子时抛 `RuntimeError`。

### 参考实现

同模块另导出两个纯 PyTorch 函数，供对拍与 CPU 测试：

```python
def torch_complex_mul(
    ur: torch.Tensor, ui: torch.Tensor, vr: torch.Tensor, vi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]

def torch_complex_mul_grad(
    gwr: torch.Tensor, gwi: torch.Tensor,
    ur: torch.Tensor, ui: torch.Tensor, vr: torch.Tensor, vi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

公式与约束同[算子列表 · ComplexMul / ComplexMulGrad](op_list.md#complexmul)。二者都不走 NPU 自定义算子。

---

## `oil_gas_ops_prospect.fused_bias_softmax`

### `fused_bias_softmax`

```python
from oil_gas_ops_prospect.fused_bias_softmax import fused_bias_softmax

def fused_bias_softmax(
    x: torch.Tensor,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
) -> torch.Tensor
```

\(y = \mathrm{softmax}(x + \mathrm{bias1} + \mathrm{bias2},\, -1)\)。

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `torch.Tensor` | logits，`[B, H, Q, S]`，`fp32` / `fp16` / `bf16` |
| `bias1` | `torch.Tensor` 或 `None` | `[B, H, 1, S]` 广播或 `[B, H, Q, S]` 全尺寸；`None` 表示不加 |
| `bias2` | `torch.Tensor` 或 `None` | `[B, H, Q, S]`；`None` 表示不加 |

返回与 `x` 同 shape / dtype 的 softmax 概率。NPU 融合路径**原地写回 `x`**（返回张量与 `x` 共享 storage）；参考实现非原地。

**实现选择**：默认走 PyTorch。张量在 NPU 且 `OIL_GAS_OPS_FUSED_BIAS_SOFTMAX_IMPL` 为 `ascendc` / `auto` 时走 AscendC。失败默认静默回退，除非 `OIL_GAS_OPS_REQUIRE_ASCENDC_FUSED_BIAS_SOFTMAX=1`。

### `fused_softmax_grad`

```python
from oil_gas_ops_prospect.fused_bias_softmax import fused_softmax_grad

def fused_softmax_grad(
    probs: torch.Tensor,
    grad_output: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor
```

先算 \(g = \mathrm{grad\_output} \cdot V^{\top}\)，再做 softmax JVP，得到 \(\partial L / \partial \mathrm{logits}\)。

| 参数 | 类型 | 说明 |
|------|------|------|
| `probs` | `torch.Tensor` | 前向 softmax 概率，`[B, H, Q, S]`（`Q ≠ S` 合法） |
| `grad_output` | `torch.Tensor` | \(\partial L / \partial o\)，`[B, H, Q, D]` |
| `values` | `torch.Tensor` | \(V\)，`[B, H, S, D]` |

三输入同 dtype（`fp32` / `fp16` / `bf16`）。返回与 `probs` 同 shape。NPU 融合路径**原地覆写 `probs`**；参考实现非原地。实现开关与 `fused_bias_softmax` 共用。

### `attention_core`

```python
from oil_gas_ops_prospect.fused_bias_softmax import attention_core

def attention_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
) -> torch.Tensor
```

openfold `AttentionCoreFunction.apply` 等价封装，**不是**独立 OPP 算子，而是前反向组合：

```text
forward:   logits = q @ k^T
           probs  = softmax(logits + bias1 + bias2, -1)   # fused_bias_softmax
           o      = probs @ v
backward:  grad_v = probs^T @ go                          # 先于 probs 覆写，与原版 CUDA 时序一致
           g_logits = softmax_jvp(probs, go, v)           # fused_softmax_grad，原地覆写 probs
           grad_bias* = 按 bias 广播维对 g_logits 求和
           grad_q = g_logits @ k
           grad_k = (q^T @ g_logits)^T
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `q` | `torch.Tensor` | query，`[B, H, Q, D]` |
| `k` | `torch.Tensor` | key，`[B, H, S, D]` |
| `v` | `torch.Tensor` | value，`[B, H, S, D]` |
| `bias1` | `torch.Tensor` 或 `None` | 典型为 `[B, H, 1, S]`（mask bias） |
| `bias2` | `torch.Tensor` 或 `None` | 典型为 `[B, H, Q, S]`（triangle bias） |

返回 `o`，shape `[B, H, Q, D]`。支持 autograd；NPU + `ascendc`/`auto` 时前反向分别走两个融合算子。用法与 openfold 一致：`o = attention_core(q, k, v, mask_bias, triangle_bias)`。

### 参考实现

```python
def torch_fused_bias_softmax(
    x: torch.Tensor, bias1: torch.Tensor | None = None, bias2: torch.Tensor | None = None,
) -> torch.Tensor

def torch_fused_softmax_grad(
    probs: torch.Tensor, grad_output: torch.Tensor, values: torch.Tensor,
) -> torch.Tensor
```

均在 fp32 域计算、非原地，写回原 dtype。供对拍与 CPU 测试。

---

## `oil_gas_ops_prospect.real_fft`

### `real_rfft`

```python
from oil_gas_ops_prospect.real_fft import real_rfft

def real_rfft(
    x: torch.Tensor,
    dim: int,
    norm: str = "ortho",
    block_n: int | None = None,
    block_m: int | None = None,
    block_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

沿 `dim` 执行实数正向 FFT，返回 `(real, imag)` 两个实数张量，`dim` 轴长度变为 `n // 2 + 1`。支持 autograd。

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `torch.Tensor` | 输入实数张量，`dim` 维长度为 \(n\) |
| `dim` | `int` | 变换所沿的维度 |
| `norm` | `str` | **只支持 `"ortho"`**，其他值抛 `ValueError` |
| `block_n`、`block_m`、`block_k` | `int` 或 `None` | Triton kernel 分块；`None` 由实现选择。`block_m` 还可由 `OIL_GAS_OPS_FFT_BLOCK_M` 覆盖（默认 512） |

NPU + Triton 可用时走融合 kernel（计算/输出为 `float32`）；否则回退 `torch.fft.rfft`。

### `real_irfft`

```python
from oil_gas_ops_prospect.real_fft import real_irfft

def real_irfft(
    xr: torch.Tensor,
    xi: torch.Tensor,
    n: int,
    dim: int,
    norm: str = "ortho",
    block_k: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
) -> torch.Tensor
```

从半谱实部 `xr`、虚部 `xi` 重建长度为 `n` 的实数信号。支持 autograd。

| 参数 | 类型 | 说明 |
|------|------|------|
| `xr`, `xi` | `torch.Tensor` | 半谱实部、虚部，**shape 必须相同**，否则抛 `ValueError` |
| `n` | `int` | 重建后 `dim` 维长度 |
| `dim` | `int` | 变换所沿的维度；NPU 路径要求 `xr.shape[dim] == n // 2 + 1` |
| `norm` | `str` | **只支持 `"ortho"`** |
| `block_k`、`block_m`、`block_n` | `int` 或 `None` | Triton kernel 分块 |

非 NPU / Triton 不可用时回退 `torch.fft.irfft`。

### `make_basis`

```python
from oil_gas_ops_prospect.real_fft import make_basis

def make_basis(
    n: int,
    device: torch.device,
    norm: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

构造并按 `(n, device, norm)` 缓存长度为 `n` 的稠密 Fourier 基。`norm` 必须为 `"ortho"`。

返回 `(rfft_cos, rfft_sin, irfft_cos, irfft_sin)`，均为 `float32`，shape 为 `(n, n // 2 + 1)`。一般无需直接调用，由 `real_rfft` / `real_irfft` 内部使用。

---

## aclnn 接口

安装 OPP `.run` 包后，CANN 侧走标准两段式入口（`GetWorkspaceSize` + 执行）。仓内 pybind 即按此直调：

| 接口 | OpType | 张量（顺序） | 说明 |
|------|--------|--------------|------|
| `aclnnComplexMul` / `aclnnComplexMulGetWorkspaceSize` | `ComplexMul` | `ur, ui, vr, vi → wr, wi` | 全部 `float32` / `ND`，同 shape |
| `aclnnComplexMulGrad` / `aclnnComplexMulGradGetWorkspaceSize` | `ComplexMulGrad` | `gwr, gwi, ur, ui, vr, vi → gur, gui, gvr, gvi` | 全部 `float32` / `ND`，同 shape |
| `aclnnFusedBiasSoftmax` / `aclnnFusedBiasSoftmaxGetWorkspaceSize` | `FusedBiasSoftmax` | `x, bias1, bias2 → y` | `bias1`/`bias2` 可为 `nullptr`；`y` 可与 `x` 别名 |
| `aclnnFusedSoftmaxGrad` / `aclnnFusedSoftmaxGradGetWorkspaceSize` | `FusedSoftmaxGrad` | `probs, gradOutput, values → out` | `out` 可与 `probs` 别名 |

自定义算子的 opapi 动态库位于 `${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib`，调用前需将其加入 `LD_LIBRARY_PATH`。绑定实现见 [`ascendc/pybind/csrc/`](../../ascendc/pybind/csrc/)。

Triton 算子没有 aclnn 入口。

---

## 环境变量

自定义环境变量统一使用 `OIL_GAS_OPS_` 前缀。

### 算子行为

| 环境变量 | 取值 | 默认 | 说明 |
|----------|------|------|------|
| `OIL_GAS_OPS_COMPLEX_MUL_IMPL` | `torch` / `ascendc` / `auto` | `torch` | `complex_mul` 的实现选择 |
| `OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL` | `0` / `1` | `0` | 置 `1` 时禁止回退到 PyTorch，AscendC 调用失败直接抛异常 |
| `OIL_GAS_OPS_FUSED_BIAS_SOFTMAX_IMPL` | `torch` / `ascendc` / `auto` | `torch` | `fused_bias_softmax` / `fused_softmax_grad` / `attention_core` 的实现选择 |
| `OIL_GAS_OPS_REQUIRE_ASCENDC_FUSED_BIAS_SOFTMAX` | `0` / `1` | `0` | 置 `1` 时禁止回退到 PyTorch，AscendC 调用失败直接抛异常 |
| `OIL_GAS_OPS_FFT_BLOCK_M` | 正整数 | `512` | `real_rfft` / `real_irfft` 的 DFT-matmul 行分块大小，用于性能调优（可扫 128 / 256 / 512） |

### 构建与产物

| 环境变量 | 取值 | 默认 | 说明 |
|----------|------|------|------|
| `ASCEND_HOME_PATH` | 路径 | `/usr/local/Ascend/cann-9.0.0` | CANN 安装根目录 |
| `OIL_GAS_OPS_OUTPUT_DIR` | 路径 | `<repo>/output` | 覆盖构建产物目录 |
| `OIL_GAS_OPS_SKIP_EXT` | `0` / `1` | `0` | 置 `1` 时 `setup.py` 跳过 pybind 扩展编译 |

### 测试与验证

| 环境变量 | 取值 | 默认 | 说明 |
|----------|------|------|------|
| `OIL_GAS_OPS_RUN_PKG` | 路径 | 自动探测 | `tests/ascendc/*/run_test.sh` 指定要安装的 OPP `.run` 包 |
| `OIL_GAS_OPS_WHL_REQUIRE_NPU_EXT` | `0` / `1` | `0` | 置 `1` 时 whl 冒烟测试强制要求已编译的 NPU 扩展存在，否则该用例跳过 |
| `OIL_GAS_OPS_WHL_TEST` | `0` / `1` | `0` | 由 `tests/run_whl_tests.sh` / `bash build.sh --install` 置 `1`，声明本次确实在验证已安装的 whl；此时不把仓内源码挂到 `oil_gas_ops_prospect` 导入名下，若解析到仓内路径则硬失败 |
| `OIL_GAS_OPS_RUN_LATENT3D_FFT_TEST` | `0` / `1` | `0` | 置 `1` 时运行 `(2,41,41,41,2048)` 大 shape 的 NPU FFT 重型用例 |
| `OIL_GAS_OPS_RUN_128_FFT_TEST` | `0` / `1` | `0` | 置 `1` 时运行 `tests/triton/real_fft` 的 128³ NPU 反向与显存用例 |

### 与外部模型仓对拍（可选）

`triton-ascend/real_fft` 的部分测试支持与外部模型仓的实现交叉验证，默认全部跳过，需显式开启：

| 环境变量 | 取值 | 说明 |
|----------|------|------|
| `OIL_GAS_OPS_MODEL_REPO` | 路径 | 外部模型仓根目录，设置后启用交叉验证 |
| `OIL_GAS_OPS_MODEL_CONFIG` | 路径 | 用于 shape 推导的配置文件（绝对路径） |
| `OIL_GAS_OPS_FFT_BACKBONE` | `<module>:<class>` | 覆盖默认的 backbone 导入路径 |
| `OIL_GAS_OPS_SPECTRAL_CONV` | `<module>:<class>` | 指定被对拍的 spectral conv 实现 |

详见 [`triton-ascend/real_fft/README.md`](../../triton-ascend/real_fft/README.md)。
