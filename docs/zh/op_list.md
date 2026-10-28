# 算子列表

简体中文 | [English](../en/op_list.md)

当前仓库交付 6 个 AscendC/CANN OPP 算子、5 个 Triton-Ascend Python 算子和 1 个 Composite-Torch（纯 PyTorch）算子。每个算子单独说明数学定义、输入输出与约束；Python / aclnn 签名见[接口列表](api_list.md)。

## 总览

| 名称 / OpType | 后端 | aclnn / Python | 源码 | 功能 | dtype / format | 支持 SoC |
|---------------|------|----------------|------|------|----------------|----------|
| `ComplexMul` | AscendC | `aclnnComplexMul` / `complex_mul` | [`ascendc/operators/complex_mul/`](../../ascendc/operators/complex_mul/) | 复数逐点乘 | `float32` / `ND` | Ascend 910B |
| `ComplexMulGrad` | AscendC | `aclnnComplexMulGrad`（反向由 `complex_mul` 的 autograd 调用） | [`ascendc/operators/complex_mul_grad/`](../../ascendc/operators/complex_mul_grad/) | `ComplexMul` 的反向 | `float32` / `ND` | Ascend 910B |
| `FusedBiasSoftmax` | AscendC | `aclnnFusedBiasSoftmax` / `fused_bias_softmax` | [`ascendc/operators/fused_bias_softmax/`](../../ascendc/operators/fused_bias_softmax/) | `softmax(x + bias1 + bias2, -1)`，连续输入成功进入融合路径时原地写回 | `fp32` / `fp16` / `bf16` / `ND` | Ascend 910B |
| `FusedSoftmaxGrad` | AscendC | `aclnnFusedSoftmaxGrad` / `fused_softmax_grad` | [`ascendc/operators/fused_softmax_grad/`](../../ascendc/operators/fused_softmax_grad/) | 注意力 softmax JVP，连续输入成功进入融合路径时覆写 probs | `fp32` / `fp16` / `bf16` / `ND` | Ascend 910B |
| `Conv3DGelu` | AscendC | `aclnnConv3dGeluCustom` / `conv3d_gelu` | [`ascendc/operators/conv3d_gelu/`](../../ascendc/operators/conv3d_gelu/) | `gelu_tanh(conv3d(x, weight, bias))` | `float32` / `NCDHW` | Ascend 910B |
| `Conv3DGeluGrad` | AscendC | `aclnnConv3dGeluGrad`（由 `conv3d_gelu` 的反向调用） | [`ascendc/operators/conv3d_gelu_grad/`](../../ascendc/operators/conv3d_gelu_grad/) | `dz = dy * gelu'(z)`。反向先重算 conv3d，再调本算子，再做 conv3d 反向 | `float32` / `ND` | Ascend 910B |
| `real_rfft` | Triton-Ascend | `oil_gas_ops_prospect.real_fft.real_rfft` | [`triton-ascend/real_fft/`](../../triton-ascend/real_fft/) | 实数正向 FFT：`x → (xr, xi)` | NPU kernel 为 `float32` | Ascend 910B |
| `real_irfft` | Triton-Ascend | `oil_gas_ops_prospect.real_fft.real_irfft` | [`triton-ascend/real_fft/`](../../triton-ascend/real_fft/) | 半谱重建实数：`(xr, xi), n → y` | NPU kernel 为 `float32` | Ascend 910B |
| `complex_reciprocal` | Triton-Ascend | `oil_gas_ops_prospect.complex_reciprocal.complex_reciprocal` | [`triton-ascend/complex_reciprocal/`](../../triton-ascend/complex_reciprocal/) | 复数倒数 `1/z`（`z` 或 `xr, xi`） | NPU kernel 为 `complex64` | Ascend 910B |
| `complex_exp` | Triton-Ascend | `oil_gas_ops_prospect.complex_exp.complex_exp` | [`triton-ascend/complex_exp/`](../../triton-ascend/complex_exp/) | 复数指数 `e^z` | NPU kernel 为 `complex64` | Ascend 910B |
| `interpolate` | Triton-Ascend | `oil_gas_ops_prospect.interpolate.interpolate` | [`triton-ascend/interpolate/`](../../triton-ascend/interpolate/) | 2D/3D 线性插值；5D fp32 走 Cube GEMM，4D 走原生 | 5D `float32`；4D 由原生实现支持 | Ascend 910B |
| `Conv3d` / `conv3d` | Composite-Torch | `oil_gas_ops_prospect.conv3d_custom.Conv3d` / `.conv3d` | [`composite-torch/conv3d_custom/`](../../composite-torch/conv3d_custom/) | 支持参数域内的 `nn.Conv3d` / `F.conv3d` 兼容实现（自定义前/反向） | 底层 PyTorch 后端支持且经本仓测试的 dtype | PyTorch 后端支持；已测 CPU / Ascend 910B |

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

openfold 注意力前向融合：把 bias 累加与沿最后一维的 softmax 合成单 kernel；底层 aclnn 输出可与 `x` 别名。对标 `attn_core_inplace_cuda.forward_` 的三次发射（bias1 / bias2 / in-place softmax）。

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

底层 aclnn 接口允许 **`y` 与 `x` 别名**。Python 包装仅在传入 `x` 连续且成功进入 NPU 融合路径时与调用者原始 `x` 共享 storage 并原地写回；非连续输入会先经 `contiguous()` 复制，`y` 与副本别名，不会覆写原始非连续张量。PyTorch 参考实现 `torch_fused_bias_softmax` 非原地。

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

openfold 注意力反向融合：先做 \(g = go \cdot V^\top\)，再沿最后一维做 softmax Jacobian-vector product；底层 aclnn 输出可与 `probs` 别名。对标 `attn_core_inplace_cuda.backward_`，替换 `bmm + _softmax_backward_data + copy_`。

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

底层 aclnn 接口允许 **`out` 与 `probs` 别名**。Python 包装仅在传入 `probs` 连续且成功进入 NPU 融合路径时与调用者原始 `probs` 共享 storage 并原地覆写；非连续输入会先经 `contiguous()` 复制，`out` 与副本别名，不会覆写原始非连续张量。PyTorch 参考实现 `torch_fused_softmax_grad` 非原地。

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

### Conv3DGelu

Conv3d 与 tanh 近似 GELU 的 cube+vector 融合。对齐的是 NPU 上 `F.gelu` / `nn.GELU` 实际执行的 tanh 近似，也就是 CPU 的 `F.gelu(approximate="tanh")`。CPU `nn.GELU()` 默认是 erf，不是本算子的真值。

#### 数学定义

\[
y = \text{gelu}_{\tanh}(\text{conv3d}(x, W, b))
\]

\[
\text{gelu}_{\tanh}(x) = \frac{x}{1 + \exp(-1.5957691216057308\,(x + 0.044715\, x^3))}
\]

1×1、stride=1、padding=0，且输入通道不少于 8、输出通道不超过 64 时，GELU 直接写进用户输出。其余形状沿输出深度每次只算一个平面，私有格式缓冲只跟这一平面走。反向不保存预激活，由 `Conv3DGeluGrad` 在一次调用里重算 conv3d、做 GELU 反向，再做 conv3d 反向。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `x` | 特征图 | `NCDHW` | `float32` |
| 输入 | `filter` | 卷积核 | `(C_out, C_in/groups, Kd, Kh, Kw)` | `float32` |
| 输入 | `bias` | 可选偏置 | `(C_out,)` | `float32` |
| 输出 | `y` | GELU 之后的结果 | `NCDHW` | `float32` |

#### 约束

- 只支持 `float32`、`groups=1`。`groups` 不是 1，或 `stride` / `padding` / `dilation` 的长度不是 1 或 3，会抛 `ValueError`。`padding_mode` 不是 `zeros` 时由 Python 先做 pad。
- 精度验收相对 CPU `F.gelu(F.conv3d(...), approximate="tanh")`，且关闭 `torch_npu.npu.conv.allow_hf32`，相对误差 `< 1e-5`。见 [`tests/ascendc/conv3d_gelu/`](../../tests/ascendc/conv3d_gelu/)。
- 必测 shape 与原 `oil_gas_ops` conv3d_gelu 用例一致，包含 `(2, 32, 160, 160, 160)`、kernel `3×3×3`。

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

- **stable 路径**（默认）：用 \(t = |a| + |b|\) 缩放，记 \(\hat{a} = a/t,\ \hat{b} = b/t\)，按 \(w = \frac{\hat{a} - i\,\hat{b}}{t\,(\hat{a}^2 + \hat{b}^2)}\) 计算；该式与 \(\frac{a - ib}{a^2+b^2}\) 代数等价，但避免了 \(a^2+b^2\) 上下溢；
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
- 元素数达到 `2^30` 起走 int64 offset 分支（kernel 自动选择，AOT 已覆盖）；单个张量元素数超过 `2^31 - 1` 直接抛 `ValueError`，因为块内 int32 运算本身会回绕。
- `block_size` 每 tile 复数元素数，默认 `2048`；会向下取整到 2 的幂并钳制到 `[16, 2048]`，可由 `OIL_GAS_OPS_RECIP_BLOCK` 覆盖。
- 支持 autograd（自定义 backward kernel）与双入口（交错 / 实虚分离）。

实现与说明详见 [`triton-ascend/complex_reciprocal/README.md`](../../triton-ascend/complex_reciprocal/README.md)。

---

### complex_exp

complex64 张量的指数，替代 NPU 上走 Ai_Cpu 的 `aclnnExp`（f32 同算子走 AiCore 为"良"级，complex64 触发 AiCpu 后劣化约 2500 倍）。

#### 数学定义

对 \(z = a + ib\)：

\[
w = e^z = e^a (\cos b + i\sin b)
\]

反向：\(f(z) = e^z\) 全纯，\(f'(z) = e^z = w\)，torch 复数 autograd 约定下
\(\text{grad}_z = g \cdot \overline{w}\)，kernel 从保存的前向输出直接计算。
支持二阶及更高阶导：当反向本身处于 grad 模式（`create_graph=True` / `gradcheck`）时，
`backward` 改用等价的 torch 组合式 `g * conj(w)`，由 autograd 继续求导。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `z` | 复数指数 | 任意 | `complex64` |
| 输出 | `w` | 指数 | 与 `z` 相同 | `complex64` |

#### 约束

- NPU kernel 仅 `complex64`；CPU 或其他 dtype 回退 `torch.exp`（含 autograd）。
- 数值语义对齐 `cexpf`（即 `torch.exp(complex64)` 的实现），包括「`inf * 0` 不得把虚部 0 变成 NaN」和「实部在溢出处仍可能有限」这两种情况：
  - `b == 0`（含 `-0.0`）直接返回 `exp(a) + i*b`，纯实数上溢得到 `inf+0j` 而不是 `inf+nanj`。
  - `a > ln(FLT_MAX)`（约 88.72）时实部可能仍然有限（如 `89+1j` → `2.43e38+infj`）。kernel 用 `s = exp(a/4)` 连乘四次，不让 `exp(a)` 先溢出，结果与 torch 一致。
  - `b` 为 `±inf` 或 NaN 时按 `cexpf` 特判：`a == -inf` 返回 `0`，`a == +inf` 返回 `inf`，其余 NaN。
  - 下溢与常规有限值沿用 float32 `exp/cos/sin` 组合。
- **已知限制**：triton-ascend 后端的 `tl.sin` / `tl.cos`（以及 CANN `libdevice` 版本）在 `|b| >= 2^25`（33554432）后没有参数归约，结果从该量级起与 `torch.exp` 不一致。本算子目标场景的相位角较小；虚部数值极大时请改用 `torch.exp`。
- 元素数超过 `2^31` 不支持；`2^30` 起走 int64 offset 分支（前向与反向都由 kernel 自动选择，AOT 已覆盖）。
- `block_size` 每 tile 复数元素数（2 的幂，钳制 `[16, 2048]`），可由 `OIL_GAS_OPS_EXP_BLOCK` 覆盖。
- Triton kernel 有固定的 host 派发/launch 开销（约 0.26ms/次）；元素数很少（约低于 `3×10^4`）时可能慢于 `torch.exp`。该算子面向大张量（实测 12x~160x），极小输入建议直接调用 `torch.exp`。
- 验证记录见 [`tests/triton/complex_exp/report/`](../../tests/triton/complex_exp/report/)。

实现与说明详见 [`triton-ascend/complex_exp/README.md`](../../triton-ascend/complex_exp/README.md)。

---

### interpolate

2D/3D 线性插值。5D `float32` `trilinear` 将每个空间轴的一次批量 GEMM 交给 Cube；4D `bilinear` 与其余签名回退 `F.interpolate`。

#### 数学定义

对单个轴，线性插值可写成矩阵乘：

\[
Y = X W,\qquad W\in\mathbb{R}^{n_{\text{in}}\times n_{\text{out}}}
\]

其中 \(W\) 的每一列只有两个非零元素（`align_corners=False` 时）：

\[
W_{i_0,j}=1-w_j,\qquad W_{i_1,j}=w_j
\]

3D 三线性插值是三个轴的可分离组合：

\[
Y = \bigl((X W_w) W_h\bigr) W_d
\]

实现上按 `W -> H -> D` 顺序执行三次批量 GEMM：
`M = batch × 其余空间维`，`K = 输入轴长度`，`N = 输出轴长度`。
每个 pass 使用 `num_aicore` 的 persistent grid，每个核循环处理多个 tile；
shape / stride 参数走 `tl.constexpr` 特化。
权重矩阵由 PyTorch 自身对单位阵做一次 `F.interpolate` 生成，因此与原生实现的 fp32 权重一致。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `x` | 输入特征 | 4D `[B,C,H,W]` 或 5D `[B,C,D,H,W]` | NPU kernel 为 `float32` |
| 输入 | `size` | 目标空间尺寸 | 2 或 3 元组 | `int` |
| 输入 | `mode` | 插值模式 | `"bilinear"`（4D）或 `"trilinear"`（5D） | `str` |
| 输入 | `align_corners` | 对齐方式 | `False` / `True` | `bool` |
| 输出 | `y` | 插值结果 | `size` 替换空间维，其余不变 | 与 `x` 相同 |

#### 约束

- NPU 快路径仅 5D `float32` `trilinear`，`align_corners` 两种取值都支持；4D、`nearest`、`bicubic`、`scale_factor`、`antialias`、fp16/bf16/fp64 均回退 `F.interpolate`。
- 仅当 `size` 精确给出目标尺寸时启用快路径；`scale_factor` 走原生。
- 输出小于 128 MiB 时使用原生 kernel；该阈值和单次 launch 的 1 GiB 分块预算均为内部实现细节，不提供公开环境变量。
- 支持一阶 autograd；反向是三个轴上的转置权重 GEMM。4D 使用原生 `F.interpolate` autograd。
- 验证记录见 [`tests/triton/interpolate/report/`](../../tests/triton/interpolate/report/)。

实现、环境变量与调试说明详见 [`triton-ascend/interpolate/README.md`](../../triton-ascend/interpolate/README.md)。


---

## Composite-Torch 算子

交付物为 Python whl 内的纯 PyTorch 组合模块（`torch.autograd.Function` 组织前/反向，无自定义 kernel），CPU / NPU 行为一致。

### Conv3d

在下述支持参数域内兼容 `torch.nn.Conv3d` / `F.conv3d`。检测到 `torch.npu` 接口时模块默认走 `conv3d` 自定义前/反向，否则回退 `F.conv3d`；weight/bias 为顶层参数，state_dict key 与 `nn.Conv3d` 一致。

#### 数学定义

在支持参数域内，前向采用与 `F.conv3d` 相同的 3D cross-correlation 数学定义。反向对已知 \(\partial L/\partial \text{out}\) 的三个梯度：

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
| 输入 | `x` | 输入特征 | \((N, C_{in}, D, H, W)\) | 底层 PyTorch 后端支持且经本仓测试的 dtype |
| 输出 | `output` | 输出特征 | \((N, C_{out}, D_{out}, H_{out}, W_{out})\) | 与输入相同 |

`kernel_size`、`stride`、数值 `padding`、`dilation` 支持 `int` 或 3 元素序列（元组或列表），`groups` 为 `int`。模块的 `padding_mode` 支持 `zeros` / `reflect` / `replicate` / `circular`。字符串 `padding="same"` / `"valid"` 受支持，会在卷积前换算为显式补边；`"same"` 要求 stride 为 1，否则抛 `ValueError`。当前 `Conv3d` 构造器没有 `nn.Conv3d` 的 `device` / `dtype` 关键字。

#### 约束与行为

- `use_custom` 在检测到 `torch.npu` 接口时默认开启；测试可强制 `module.use_custom = True` 在 CPU 上驱动自定义路径。
- 参数只接受上述受支持类型与取值；数学一致性结论仅适用于该参数域和仓内覆盖的组合。
- `padding_mode != 'zeros'` 时先 `F.pad`，再以 `conv_padding=0` 进入自定义 Function，梯度经 `F.pad` 的 autograd 回传。
- weight/bias 的 state_dict key 与 `nn.Conv3d` 互通。
- dtype 取决于底层 PyTorch 后端能力，并须经过本仓对应 CPU / NPU 测试后才能列为支持。
- 支持复数输入：反向对翻转后的 weight 取共轭，复数权重梯度由实部/虚部各自的实数 `conv3d_weight` 结果重组（`torch.nn.grad.conv3d_weight` 本身只接受实数张量）。
- 在仓内覆盖的特定大 kernel depthwise NPU 形态中，分步反向可避开原生反向曾触发的 L1 buffer 限制；该观察不扩展为所有大 kernel 或任意形态的保证。

#### 验证

- 精度 / 性能用例：[`tests/composite/conv3d_custom/`](../../tests/composite/conv3d_custom/)（仓内列出的参数组合、训练调用形态与性能基准）。
- 已有用例覆盖 CPU float64/float32 组合、NPU float32 的指定形态，以及 CPU complex128/complex64 与 NPU complex64（[`test_conv3d_custom_complex.py`](../../tests/composite/conv3d_custom/test_conv3d_custom_complex.py)，含 CPU complex128 的 `gradcheck` / `gradgradcheck` 双阶反向）；结论仅适用于实际执行并通过的 case，不代表未列出的参数或 dtype。
- 复数 NPU 用例在 `torch.npu.conv.allow_hf32 = False` 下执行（结束后恢复原值）。默认 HF32 设置下部分梯度相对误差约 `2e-4`，不按完整 fp32 阈值验收。

实现与使用说明详见 [`composite-torch/conv3d_custom/README.md`](../../composite-torch/conv3d_custom/README.md)。
