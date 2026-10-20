# 算子列表

当前仓库交付 4 个 AscendC/CANN OPP 算子、3 个 Triton-Ascend Python 算子和 1 个 Composite-Torch（纯 PyTorch）算子。每个算子单独说明数学定义、输入输出与约束；Python / aclnn 签名见[接口列表](api_list.md)。

## 总览

| 名称 / OpType | 后端 | aclnn / Python | 源码 | 功能 | dtype / format | 支持 SoC |
|---------------|------|----------------|------|------|----------------|----------|
| `ComplexMul` | AscendC | `aclnnComplexMul` / `complex_mul` | [`ascendc/operators/complex_mul/`](../../ascendc/operators/complex_mul/) | 复数逐点乘 | `float32` / `ND` | Ascend 910B |
| `ComplexMulGrad` | AscendC | `aclnnComplexMulGrad`（反向由 `complex_mul` 的 autograd 调用） | [`ascendc/operators/complex_mul_grad/`](../../ascendc/operators/complex_mul_grad/) | `ComplexMul` 的反向 | `float32` / `ND` | Ascend 910B |
| `FusedBiasSoftmax` | AscendC | `aclnnFusedBiasSoftmax` / `fused_bias_softmax` | [`ascendc/operators/fused_bias_softmax/`](../../ascendc/operators/fused_bias_softmax/) | `softmax(x + bias1 + bias2, -1)`，原地写回 | `fp32` / `fp16` / `bf16` / `ND` | Ascend 910B |
| `FusedSoftmaxGrad` | AscendC | `aclnnFusedSoftmaxGrad` / `fused_softmax_grad` | [`ascendc/operators/fused_softmax_grad/`](../../ascendc/operators/fused_softmax_grad/) | 注意力 softmax JVP，原地覆写 probs | `fp32` / `fp16` / `bf16` / `ND` | Ascend 910B |
| `real_rfft` | Triton-Ascend | `oil_gas_ops_prospect.real_fft.real_rfft` | [`triton-ascend/real_fft/`](../../triton-ascend/real_fft/) | 实数正向 FFT：`x → (xr, xi)` | NPU kernel 为 `float32` | Ascend 910B |
| `real_irfft` | Triton-Ascend | `oil_gas_ops_prospect.real_fft.real_irfft` | [`triton-ascend/real_fft/`](../../triton-ascend/real_fft/) | 半谱重建实数：`(xr, xi), n → y` | NPU kernel 为 `float32` | Ascend 910B |
| `Conv3d` / `conv3d` | Composite-Torch | `oil_gas_ops_prospect.conv3d_custom.Conv3d` / `.conv3d` | [`composite-torch/conv3d_custom/`](../../composite-torch/conv3d_custom/) | `nn.Conv3d` / `F.conv3d` 等价替换（自定义前/反向） | 任意（同 `nn.Conv3d`） | 任意（纯 PyTorch） |

命名约定：

- C++ 侧 OpType 为 PascalCase（`ComplexMul`），aclnn 为 `aclnnComplexMul`。
- Python 侧公开函数为蛇形命名（`complex_mul`），不加 `npu_` 前缀。
- AscendC kernel 入口通过 `opFile.value` / `opInterface.value` 固定为蛇形文件名（如 `complex_mul`），OpType 改名不影响 kernel 源码，但 aclnn 接口名随 OpType 变化。

---

## AscendC 算子

交付物为 CANN OPP `.run` 算子包，安装后暴露 `aclnn` 接口，并通过 `ascendc/pybind/` 在 Python 侧调用。

### ComplexMul

复数逐点乘。AIV-only 逐元素算子，支持动态 rank / 动态 shape。

#### 数学定义

将输入拆成两个复数 \(u = u_r + i u_i\)、\(v = v_r + i v_i\)，输出 \(w = u \cdot v\)：

\[
\begin{aligned}
w_r &= u_r v_r - u_i v_i \\
w_i &= u_r v_i + u_i v_r
\end{aligned}
\]

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `ur`, `ui` | 第一个复数的实部、虚部 | 任意 rank，彼此相同 | `float32` / `ND` |
| 输入 | `vr`, `vi` | 第二个复数的实部、虚部 | 与 `ur` 相同 | `float32` / `ND` |
| 输出 | `wr`, `wi` | 乘积的实部、虚部 | 与 `ur` 相同 | `float32` / `ND` |

#### 约束

- 四个输入与两个输出必须 **同 shape、同 dtype、`float32`、`ND`**。
- 输入需连续（Python 绑定会 `contiguous()`）。
- **AIV-only**：按元素总数在 AIV 核间均分（32B 对齐），核内再按 UB 预算分拍。前向 UB 模型为 13 份 tile（4 路 inQue + 2 路 outQue 的 double buffer + 1 份 tmp）。
- 切分算法在 `op_host/complex_mul_tiling_core.h`，由 [`tests/ut/op_host/complex_mul/test_complex_mul_tiling.cpp`](../../tests/ut/op_host/complex_mul/test_complex_mul_tiling.cpp) 覆盖。
- Python 默认走 PyTorch 参考实现；仅当张量在 NPU 且 `OIL_GAS_OPS_COMPLEX_MUL_IMPL` 为 `ascendc` / `auto` 时调用本算子。
- 精度验收：相对误差默认阈值 `1e-5`（见 [`tests/ascendc/complex_mul/`](../../tests/ascendc/complex_mul/)）。

Python 侧没有单独的 `complex_mul_grad` 函数；反向由 `complex_mul` 的 `torch.autograd.Function` 调用 `ComplexMulGrad`。

---

### ComplexMulGrad

`ComplexMul` 的反向。AIV-only 逐元素算子。

#### 数学定义

对 \(w = u \cdot v\)，已知 \(\partial L / \partial w = g_w = g_{w_r} + i g_{w_i}\)：

\[
\begin{aligned}
g_{u_r} &= g_{w_r} v_r + g_{w_i} v_i \\
g_{u_i} &= -g_{w_r} v_i + g_{w_i} v_r \\
g_{v_r} &= g_{w_r} u_r + g_{w_i} u_i \\
g_{v_i} &= -g_{w_r} u_i + g_{w_i} u_r
\end{aligned}
\]

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `gwr`, `gwi` | \(\partial L / \partial w\) 的实部、虚部 | 与前向输入相同 | `float32` / `ND` |
| 输入 | `ur`, `ui`, `vr`, `vi` | 前向保存的输入 | 与 `gwr` 相同 | `float32` / `ND` |
| 输出 | `gur`, `gui`, `gvr`, `gvi` | \(\partial L / \partial u\)、\(\partial L / \partial v\) | 与 `gwr` 相同 | `float32` / `ND` |

#### 约束

- 六个输入与四个输出必须 **同 shape、同 dtype、`float32`、`ND`**。
- **AIV-only**。反向 UB 模型为 21 份 tile（6 路输入 + 4 路输出的 double buffer + tmp）。
- 切分算法在 `op_host/complex_mul_grad_tiling_core.h`，与前向共用 `og_elementwise_tiling.h`，由同一份 ophost UT 覆盖。
- C++ / aclnn 可单独调用；Python 只通过 `complex_mul(...).backward()` 触发。

---

### FusedBiasSoftmax

openfold 注意力前向融合：把 bias 累加与沿最后一维的 softmax 合成单 kernel，原地写回 `x`。对标 `attn_core_inplace_cuda.forward_` 的三次发射（bias1 / bias2 / in-place softmax）。

#### 数学定义

\[
y = \mathrm{softmax}(x + b_1 + b_2,\; \mathrm{dim}=-1)
\]

其中 \(b_1\)、\(b_2\) 均可缺省（视为 0）。非 `fp32` 输入在 fp32 域累加并做 softmax，再写回原 dtype。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `x` | logits | `[B, H, Q, S]` | `fp32` / `fp16` / `bf16`，`ND` |
| 输入（可选） | `bias1` | 可广播偏置（如 mask bias） | `[B, H, 1, S]` 或 `[B, H, Q, S]` | 与 `x` 相同 |
| 输入（可选） | `bias2` | 全尺寸偏置（如 triangle bias） | `[B, H, Q, S]` | 与 `x` 相同 |
| 输出 | `y` | softmax 概率 | 与 `x` 相同 | 与 `x` 相同 |

NPU 融合路径上 **`y` 与 `x` 共享同一块 device 内存**（原地写回）。PyTorch 参考实现 `torch_fused_bias_softmax` 非原地。

#### 约束

- **必须 4D**。`S = width` 满足 \(1 \le S \le 4096\)；`Q = height`、`B`、`H` 均需大于 0。
- `bias1` 的 `B/H/S` 必须与 `x` 对齐，`Q` 维只能是 `1`（广播）或等于 `x` 的 `Q`（全尺寸）。
- `bias2` 若存在必须与 `x` **全尺寸相同**。
- 存在的 bias 必须与 `x` **同 dtype**。
- **AIV-only**。按行（`B*H*Q`）在 AIV 核间均分，每行 `S` 对齐到 32B；高阶 SoftMax 预留 32KB UB。
- tiling key：`0=fp32`，`1=fp16`，`2=bf16`。
- 切分算法在 `op_host/fused_bias_softmax_tiling_core.h`，由 [`tests/ut/op_host/fused_bias_softmax/test_fused_bias_softmax_tiling.cpp`](../../tests/ut/op_host/fused_bias_softmax/test_fused_bias_softmax_tiling.cpp) 覆盖。
- 精度验收（[`tests/ascendc/fused_bias_softmax/`](../../tests/ascendc/fused_bias_softmax/)）：fp32 `max_diff ≤ 1e-5`；bf16 契约式 `atol=2e-2, rtol=4e-3`。

---

### FusedSoftmaxGrad

openfold 注意力反向融合：先做 \(g = go \cdot V^\top\)，再沿最后一维做 softmax Jacobian-vector product，原地覆写 `probs`。对标 `attn_core_inplace_cuda.backward_`，替换 `bmm + _softmax_backward_data + copy_`。

#### 数学定义

\[
\begin{aligned}
g &= \mathrm{grad\_output} \cdot V^{\top} \\
\mathrm{out} &= P \odot \bigl(g - \mathrm{rowsum}(P \odot g)\bigr)
\end{aligned}
\]

其中 \(P\) 为前向 softmax 概率（最后一维和为 1）。计算在 **fp32 域**完成，再写回原 dtype。`out` 即 \(\partial L / \partial \mathrm{logits}\)。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `probs` | 前向 softmax 概率 \(P\) | `[B, H, Q, S]` | `fp32` / `fp16` / `bf16`，`ND` |
| 输入 | `gradOutput` | \(\partial L / \partial o\) | `[B, H, Q, D]` | 与 `probs` 相同 |
| 输入 | `values` | 注意力的 \(V\) | `[B, H, S, D]` | 与 `probs` 相同 |
| 输出 | `out` | \(\partial L / \partial \mathrm{logits}\) | 与 `probs` 相同 | 与 `probs` 相同 |

NPU 融合路径上 **`out` 与 `probs` 共享同一块 device 内存**（原地覆写）。PyTorch 参考实现 `torch_fused_softmax_grad` 非原地。

`Q ≠ S` 合法（openfold MSA / 三角注意力形态）。

#### 约束

- **必须 4D**。`B/H` 三输入对齐；`gradOutput` 与 `values` 的末维 `D` 必须一致且大于 0；`Q`、`S` 大于 0。
- 三输入必须 **同 dtype**，支持 `fp32` / `fp16` / `bf16`。
- 路径分发（tiling key，与 kernel `TILING_KEY_IS` 一致）：

  | key | 路径 | 条件 |
  |-----|------|------|
  | 0 | 纯向量（AIV） | `fp32` |
  | 1 | 纯向量（AIV） | `fp16` |
  | 2 | 纯向量（AIV） | `bf16` 且 `S%16 ≠ 0` 或 `D%16 ≠ 0` |
  | 3 | MIX_AIC_1_2（AIC 矩乘 → workspace → AIV JVP） | `bf16` 且 `S%16==0` 且 `D%16==0` |

- `fp32` / `fp16` 不走 cube：Ascend 910B fixpipe C 为 16bit 结构上限，`g` 以 half 落盘的量化误差会超出 fp32 契约（`rtol=1e-4`）。
- 向量路径要求 \(V\) 的 plane（`S * pad(D) * 4` 字节）加上约 48KB 余量能放入 AIV UB，否则 tiling 失败。
- MIX 路径 workspace = 系统预留 16MB + `B*H*Q*S*sizeof(half)` 的 `g` 缓冲。
- 切分算法在 `op_host/fused_softmax_grad_tiling_core.h`，由 [`tests/ut/op_host/fused_softmax_grad/test_fused_softmax_grad_tiling.cpp`](../../tests/ut/op_host/fused_softmax_grad/test_fused_softmax_grad_tiling.cpp) 覆盖。
- 精度验收与前向同一目录：fp32 `max_diff ≤ 1e-5`；bf16 `atol=2e-2, rtol=4e-3`。MIX 路径的 half-`g` 量化误差经 `rowsum(P⊙g)` 放大，与 \(\sum P^2\) 成正比，生产输入应为 softmax 分布。

`attention_core(q, k, v, bias1, bias2)` 是上述两算子按 openfold `AttentionCoreFunction.apply` 语义的组合封装，不是独立 OPP 算子。数学定义见[接口列表](api_list.md#attention_core)。

---

## Triton-Ascend 算子

交付物为 Python whl 内的 Triton kernel，支持 autograd。NPU 上把稠密 Fourier 基矩阵当作 `tl.dot` GEMM 跑在 cube 上；非 NPU 或 Triton 不可用时回退到 `torch.fft`，接口语义保持一致。

### real_rfft

沿指定维做实数正向 FFT，返回半谱的实部与虚部。

#### 数学定义

对长度为 \(n\) 的实序列（`norm="ortho"`）：

\[
X[k] = n^{-1/2} \sum_{t=0}^{n-1} x[t]\, e^{-2\pi i k t / n},\quad k = 0,1,\ldots,\lfloor n/2 \rfloor
\]

实现上构造并缓存基矩阵（`make_basis`）：

\[
\begin{aligned}
\mathrm{rfft\_cos}[t,k] &= n^{-1/2}\cos(2\pi k t / n) \\
\mathrm{rfft\_sin}[t,k] &= -n^{-1/2}\sin(2\pi k t / n)
\end{aligned}
\]

然后 \(X_r = x \cdot \mathrm{rfft\_cos}\)，\(X_i = x \cdot \mathrm{rfft\_sin}\)。频率轴长度为 \(n//2+1\)。

反向（autograd）用同一组基把 \((\partial L/\partial X_r,\, \partial L/\partial X_i)\) 映回 \(\partial L/\partial x\)，与 `torch.fft.rfft(..., norm="ortho")` 对齐。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `x` | 实数信号 | 任意 rank，`dim` 维长度为 \(n\) | NPU kernel 计算/输出为 `float32` |
| 输入 | `dim` | 变换所沿的维度 | 标量 | `int` |
| 输入 | `norm` | 归一化 | 只接受 `"ortho"` | `str` |
| 输入 | `block_n` / `block_m` / `block_k` | Triton GEMM 分块，`None` 由实现选择 | 可选 | `int` |
| 输出 | `xr`, `xi` | 半谱实部、虚部 | `dim` 维变为 \(n//2+1\)，其余维不变 | `float32` |

#### 约束

- **仅支持 `norm="ortho"`**，其他值抛 `ValueError`。
- NPU + Triton 可用时走融合 kernel；否则回退 `torch.fft.rfft`，CPU 上可做精度自检。
- 基矩阵按 `(n, device, norm)` 缓存。
- `block_m` 默认 512，可用 `OIL_GAS_OPS_FFT_BLOCK_M` 覆盖；实现会按 910B L0C 容量收缩，溢出时自动减半重试。
- 支持 autograd。

---

### real_irfft

从半谱实部、虚部重建长度为 \(n\) 的实数信号，共轭对称的负频由实现按 `torch.fft.irfft` 补全。

#### 数学定义

`norm="ortho"` 下与 `torch.fft.irfft(torch.complex(xr, xi), n=n, dim=dim, norm="ortho")` 等价。基矩阵侧：

\[
\begin{aligned}
w[0] &= 1,\quad w[k] = 2\ (1 \le k < (n+1)//2),\quad \text{Nyquist（}n\text{ 为偶数）}=1 \\
\mathrm{irfft\_cos}[t,k] &= n^{-1/2}\, w[k]\, \cos(2\pi k t / n) \\
\mathrm{irfft\_sin}[t,k] &= n^{-1/2}\, w[k]\, \sin(2\pi k t / n)
\end{aligned}
\]

\[
y[t] = \sum_{k=0}^{n//2} \bigl( X_r[k]\,\mathrm{irfft\_cos}[t,k] - X_i[k]\,\mathrm{irfft\_sin}[t,k] \bigr)
\]

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `xr`, `xi` | 半谱实部、虚部 | 彼此相同；`dim` 维必须为 \(n//2+1\) | NPU kernel 为 `float32` |
| 输入 | `n` | 重建后的实数长度 | 标量 | `int` |
| 输入 | `dim` | 变换所沿的维度 | 标量 | `int` |
| 输入 | `norm` | 归一化 | 只接受 `"ortho"` | `str` |
| 输入 | `block_k` / `block_m` / `block_n` | Triton GEMM 分块，`None` 由实现选择 | 可选 | `int` |
| 输出 | `y` | 重建的实数信号 | `dim` 维长度为 \(n\)，其余维不变 | 与 `xr` 相同 |

#### 约束

- **仅支持 `norm="ortho"`**，其他值抛 `ValueError`。
- `xr.shape` 必须等于 `xi.shape`，否则抛 `ValueError`。
- NPU 路径上若 `xr.shape[dim] != n//2+1` 抛 `ValueError`。
- 非 NPU / Triton 不可用时回退 `torch.fft.irfft`。
- 与 `real_rfft` 共用基矩阵缓存与 `OIL_GAS_OPS_FFT_BLOCK_M`。
- 支持 autograd。

`make_basis(n, device, norm)` 是内部辅助，一般不必直接调用；签名见[接口列表](api_list.md#make_basis)。

实现与调试用法详见 [`triton-ascend/real_fft/README.md`](../../triton-ascend/real_fft/README.md)。

---

### complex_reciprocal

complex64 张量的倒数，地震反演等负载用它替代 NPU 上走 Ai_Cpu 的 `aclnnReciprocal`。

#### 数学定义

对 \(z = a + ib\)：

\[
w = \frac{1}{z} = \frac{\bar{z}}{|z|^2} = \frac{a - ib}{a^2 + b^2}
\]

- **stable 路径**（默认）：用 \(t = |a| + |b|\) 缩放，\(w = \frac{\mathrm{sign}(a) - i\,\mathrm{sign}(b)}{t \cdot (1 + (b/a)^2)}\) 等价变形，避免 \(a^2+b^2\) 上下溢；
- **naive 路径**：直接计算 \(a^2+b^2\)，带宽更高（约 10%），但 \(|z|\) 超出约 \([10^{-19}, 1.8\times10^{19}]\) 时会错。

反向：\(\partial(1/z)/\partial z = -1/z^2 = -w^2\)，kernel 支持交错与实虚分离两种梯度输出。

双参形式 `complex_reciprocal(xr, xi)` 计算 \(1/(xr + i\,xi)\)，`xr`/`xi` 为可广播的 float32 张量或标量，省去先构造 complex64 的 Ai_Cpu 算子。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `z` | 复数张量 | 任意 | `complex64` |
| 输出 | `w` | 倒数 | 与 `z` 相同 | `complex64` |
| 输入 | `xr`, `xi` | 实部 / 虚部（可广播，至少一个为张量） | 任意 | `float32` 或标量 |
| 输出 | `w` | `1/(xr + i*xi)` | 广播结果 | `complex64` |

#### 约束

- NPU kernel 仅 `complex64` / `float32`；CPU 或其他 dtype 回退 torch 参考实现（含 autograd）。
- 元素数超过 `2^31` 时走 int64 offset 分支（kernel 自动选择，AOT 已覆盖）。
- `block_size` 每 tile 复数元素数（2 的幂，钳制 `[16, 2048]`），可由 `OIL_GAS_OPS_RECIP_BLOCK` 覆盖。
- 支持 autograd（自定义 backward kernel）与双入口（交错 / 实虚分离）。

实现与说明详见 [`triton-ascend/complex_reciprocal/README.md`](../../triton-ascend/complex_reciprocal/README.md)。

---

## Composite-Torch 算子

交付物为 Python whl 内的纯 PyTorch 组合模块（`torch.autograd.Function` 组织前/反向，无自定义 kernel），CPU / NPU 行为一致。

### Conv3d

`torch.nn.Conv3d` / `F.conv3d` 的等价替换。NPU 可用（`torch_npu` 已导入）时模块走 `conv3d` 自定义前/反向，否则回退 `F.conv3d`；weight/bias 为顶层参数，state_dict 的 key 与 `nn.Conv3d` 完全一致。

#### 数学定义

前向与 `F.conv3d` 完全一致（3D cross-correlation）。反向对已知 \(\partial L/\partial \text{out}\) 的三个梯度：

\[
\begin{aligned}
\text{grad\_bias}[o] &= \sum_{n,d,h,w} \text{grad\_out}[n,o,d,h,w] \\
\text{grad\_weight} &= \texttt{torch.nn.grad.conv3d\_weight}(\text{input}, \text{grad\_out}, \ldots) \\
\text{grad\_in} &= \texttt{F.conv3d}\bigl(\widetilde{\text{grad\_out}},\ \text{flip}(W)^{\top},\ \text{pad}=d(K{-}1)-p,\ \ldots\bigr)
\end{aligned}
\]

其中 \(\widetilde{\text{grad\_out}}\) 为 grad_output 的零膨胀（stride>1 时元素间插 \(s-1\) 个零，`output_padding` 补齐到 input 尺寸），\(\text{flip}(W)^{\top}\) 为空间翻转 + in/out channel 转置（groups>1 时组内转置）；padding 为负时裁剪膨胀张量两侧。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `x` | 输入特征 | \((N, C_{in}, D, H, W)\) | 任意（同 `nn.Conv3d`） |
| 输出 | `output` | 输出特征 | \((N, C_{out}, D_{out}, H_{out}, W_{out})\) | 与输入相同 |

构造参数 `in_channels / out_channels / kernel_size / stride / padding / dilation / groups / bias / padding_mode` 与 `nn.Conv3d` 一致，支持 int 或 3 元组。

#### 约束与行为

- `use_custom` 跟随 NPU 可用性（`torch_npu` 已导入即开启），**不限制 stride / dilation**；测试可强制 `module.use_custom = True` 在 CPU 上驱动自定义路径。
- `padding_mode != 'zeros'` 时先 `F.pad` 再以 `conv_padding=0` 进入自定义 Function，梯度经 `F.pad` 的 autograd 正确回传。
- 支持双阶反向（二阶混合导数与原生一致）。
- 与原生的数值关系：前向逐位相同；反向 float64 最大绝对误差 ≤ 1.78e-14（数学等价）；NPU float32 上 grad_weight 因不同内核累加顺序存在 5.8e-4 ~ 9.3e-3 差异（float32 正常范围）。
- NPU 上原生 `aclnnConvolutionBackward` 对大 kernel depthwise（kernel ≥ 7 + 大空间）会因 L1 buffer 超限崩溃，本实现分步构造反向可正常完成。

#### 验证

- 精度 / 性能用例：[`tests/composite/conv3d_custom/`](../../tests/composite/conv3d_custom/)（主要取值场景组合、11 种实际训练调用模式、性能基准）。
- 数学等价性与 NPU 实机精度已验证：float64 三梯度与原生 autograd 一致（阈值 1e-12 内）；NPU float32 实际训练尺寸全模式对比通过，20 epochs 训练 loss 正常收敛。

实现与使用说明详见 [`composite-torch/conv3d_custom/README.md`](../../composite-torch/conv3d_custom/README.md)。
