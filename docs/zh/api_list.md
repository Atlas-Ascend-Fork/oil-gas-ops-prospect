# 接口列表

简体中文 | [English](../en/api_list.md)

对外接口统一由 Python 包 `oil_gas_ops_prospect` 暴露。源码树里没有这个目录：`setup.py` 在打包时把 `ascendc/pybind/`、`triton-ascend/real_fft/`、`triton-ascend/complex_reciprocal/`、`triton-ascend/interpolate/` 与 `composite-torch/conv3d_custom/` 显式映射到同一包名下；直接用仓内源码跑、没装 whl 时，由 `scripts/local_pkg.py` 自动发现并提供同一导入路径。

```text
oil_gas_ops_prospect
├── complex_mul           # complex_mul, torch_complex_mul, torch_complex_mul_grad
├── fused_bias_softmax    # fused_bias_softmax, fused_softmax_grad, attention_core, torch_*
├── real_fft              # real_rfft, real_irfft, make_basis
├── complex_reciprocal   # complex_reciprocal（z 或 xr, xi 两种形式）
├── interpolate           # interpolate, bilinear, trilinear
└── conv3d_custom         # Conv3d；functional 子模块： conv3d
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

返回与 `x` 同 shape / dtype 的 softmax 概率。成功进入 NPU 融合路径时，仅当传入张量连续，返回值才与调用者原始 `x` 共享 storage 并原地写回；非连续输入会由包装层 `contiguous()` 复制，输出与该副本别名，不会覆写调用者原始非连续张量。参考实现非原地。

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

三输入同 dtype（`fp32` / `fp16` / `bf16`）。返回与 `probs` 同 shape。成功进入 NPU 融合路径时，仅当传入张量连续，返回值才与调用者原始 `probs` 共享 storage 并原地覆写；非连续输入会由包装层 `contiguous()` 复制，输出与该副本别名，不会覆写调用者原始非连续张量。参考实现非原地。实现开关与 `fused_bias_softmax` 共用。

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
           g_logits = softmax_jvp(probs, go, v)           # AscendC 融合分支原地覆写传入扩展的连续内部 probs
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

## `oil_gas_ops_prospect.complex_reciprocal`

### `complex_reciprocal`

```python
from oil_gas_ops_prospect.complex_reciprocal import complex_reciprocal

def complex_reciprocal(
    z: torch.Tensor,
    xi: torch.Tensor | float | None = None,
    *,
    stable: bool = True,
    block_size: int | None = None,
) -> torch.Tensor
```

complex64 倒数，按输入形式自动分发：

- `complex_reciprocal(z)`：`1/z`，`z` 为 complex64 张量。NPU 上由单个 Triton kernel（Vector Core）完成；CPU 或无 triton 环境回退 `torch` 的 `1 / z`（含 autograd）。
- `complex_reciprocal(xr, xi)`：`1 / (xr + i*xi)`，实虚部分开给出，省去先构造 complex64 的 Ai_Cpu 算子；分母常见形态为「实张量 + 常数虚部」（如 `1/(a - i*c)`）。`xr` / `xi` 至少一个为 float32 张量（可互相广播），另一个可为张量或标量；输出 complex64。

| 参数 | 类型 | 说明 |
|------|------|------|
| `z` | `torch.Tensor` | 单参形式：complex64 张量；双参形式：float32 实部张量/标量 |
| `xi` | `torch.Tensor` / `float` / `None` | 双参形式的虚部（可广播）；单参形式不传 |
| `stable` | `bool` | `True`（默认）用 `t=\|a\|+\|b\|` 缩放避免 `a^2+b^2` 上下溢；`False` 朴素公式（带宽高约 10%，`\|z\|` 超出 ~[1e-19, 1.8e19] 会错） |
| `block_size` | `int` 或 `None` | 每 tile 复数元素数，向下取整到 2 的幂并钳制到 `[16, 2048]`；`None` 时取 `OIL_GAS_OPS_RECIP_BLOCK`，未设置则用默认值 `2048` |

---

## `oil_gas_ops_prospect.interpolate`

### `interpolate`

```python
from oil_gas_ops_prospect.interpolate import interpolate

def interpolate(
    x: torch.Tensor,
    size: int | Sequence[int] | None = None,
    scale_factor: float | Sequence[float] | None = None,
    mode: str = "trilinear",
    align_corners: bool = False,
    antialias: bool = False,
    recompute_scale_factor: bool | None = None,
) -> torch.Tensor
```

2D/3D 线性插值。NPU 上 5D `float32` `trilinear` 走 Cube GEMM 路径（每个空间轴一次批量矩阵乘，支持 autograd）；4D 与其余签名回退 `F.interpolate`。

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `torch.Tensor` | 4D `[B,C,H,W]` 或 5D `[B,C,D,H,W]`；NPU 快路径要求 `float32` |
| `size` | `int` / 序列 / `None` | 目标空间尺寸；4D 给 2 元组，5D 给 3 元组 |
| `scale_factor` | `float` / 序列 / `None` | 传此参数时回退 `F.interpolate` |
| `mode` | `str` | 4D 用 `"bilinear"`，5D 用 `"trilinear"`；其它模式回退 |
| `align_corners` | `bool` | 两种取值都支持 |
| `antialias` | `bool` | 置 `True` 时回退原生 |
| `recompute_scale_factor` | `bool` / `None` | 仅在回退路径使用 |

便捷入口：

```python
from oil_gas_ops_prospect.interpolate import bilinear, trilinear

y2 = bilinear(x2, size=(256, 256))
y3 = trilinear(x3, size=(160, 160, 160))
```

#### 行为与约束

- NPU 快路径仅支持 5D `float32` `trilinear`；4D、fp16/bf16/fp64 回退 `F.interpolate`。
- 输出小于 128 MiB 时使用原生 kernel；该阈值和 1 GiB 单次 launch 分块预算为内部实现细节，不提供公开环境变量。
- 支持一阶 autograd；反向为转置权重的三次轴 GEMM。4D 使用原生 autograd。
- 验证记录见 [`tests/triton/interpolate/report/`](../../tests/triton/interpolate/report/)。

环境变量见[环境变量 · 算子行为](#算子行为)。

---

## `oil_gas_ops_prospect.conv3d_custom`

### `Conv3d`

```python
from oil_gas_ops_prospect.conv3d_custom import Conv3d

class Conv3d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

纯 PyTorch 组合实现，在下述支持参数域内与 `torch.nn.Conv3d` 数学一致；`forward(x)` 返回 \((N, C_{out}, D_{out}, H_{out}, W_{out})\)。当前构造器不提供 `nn.Conv3d` 的 `device` / `dtype` 关键字。

- **参数范围**：`kernel_size`、`stride`、数值 `padding`、`dilation` 支持 `int` 或 3 元素序列（元组或列表）；`groups` 为 `int`。字符串 `padding="same"` / `"valid"` 受支持，会在卷积前换算为显式补边；`"same"` 要求 stride 为 1，否则抛 `ValueError`。
- **state_dict 互通**：key 与 `nn.Conv3d` 一致，可双向 `load_state_dict`（strict）。
- **路径选择**：检测到 `torch.npu` 接口时 `use_custom` 默认开启，否则回退 `F.conv3d`；测试可强制 `module.use_custom = True` 在 CPU 上驱动自定义路径。
- **反向与 dtype**：前/反向由内部 `_Conv3dFunction` 组织；数学等价性仅针对上述支持参数域及仓内覆盖的组合。dtype 范围为底层 PyTorch 后端支持且经本仓测试的 dtype，含复数输入（CPU complex128/complex64 与 NPU complex64）——复数反向对 weight 取共轭，权重梯度由实数卷积梯度重组。
- **`padding_mode`**：模块支持 `zeros` / `reflect` / `replicate` / `circular`；非 `zeros` 模式先 `F.pad` 再卷积。

### `conv3d`

```python
from oil_gas_ops_prospect.conv3d_custom.functional import conv3d

def conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    groups: int = 1,
) -> torch.Tensor
```

纯 PyTorch 函数式兼容实现，始终经内部 `_Conv3dFunction`（`torch.autograd.Function`）组织前/反向。`stride`、数值 `padding`、`dilation` 支持 `int` 或 3 元素序列（元组或列表），`groups` 为 `int`；字符串 `padding="same"` / `"valid"` 会在卷积前换算为显式补边（`"same"` 要求 stride 为 1）。在这些受支持参数及底层 PyTorch 后端支持且经本仓测试的 dtype 范围内，与 `F.conv3d` 数学一致。nn.Module 形态与 NPU 回退逻辑见 `Conv3d`。

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
| `OIL_GAS_OPS_RECIP_BLOCK` | 正整数（2 的幂） | `2048` | `complex_reciprocal` 的 tile 大小（向下取整到 2 的幂，钳制 `[16, 2048]`）；显式传入的 `block_size` 参数优先 |
| `OIL_GAS_OPS_INTERPOLATE_IMPL` | `auto` / `triton` / `torch` | `auto` | `interpolate` 的实现选择；`auto` 对大 5D fp32 输出走 Cube，4D/小输出/非 fp32 回退原生 |
| `OIL_GAS_OPS_FFT_BLOCK_M` | 正整数 | `512` | `real_rfft` / `real_irfft` 的 DFT-matmul 行分块大小，用于性能调优（可扫 128 / 256 / 512）。每进程首次使用时读取一次并缓存，需在启动 Python 前设置 |

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
