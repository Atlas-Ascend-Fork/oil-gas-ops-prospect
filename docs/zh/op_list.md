# 算子列表

简体中文 | [English](../en/op_list.md)

当前仓库交付 9 个 AscendC/CANN OPP 算子、8 个 Triton-Ascend Python 算子和 3 个 Composite-Torch（纯 PyTorch）算子。每个算子单独说明数学定义、输入输出与约束；Python / aclnn 签名见[接口列表](api_list.md)。

## 总览

| 名称 / OpType | 后端 | aclnn / Python | 源码 | 功能 | dtype / format | 支持 SoC |
|---------------|------|----------------|------|------|----------------|----------|
| `ComplexMul` | AscendC | `aclnnComplexMul` / `complex_mul` | [`ascendc/operators/complex_mul/`](../../ascendc/operators/complex_mul/) | 复数逐点乘 | `float32` / `ND` | Ascend 910B |
| `ComplexMulGrad` | AscendC | `aclnnComplexMulGrad`（反向由 `complex_mul` 的 autograd 调用） | [`ascendc/operators/complex_mul_grad/`](../../ascendc/operators/complex_mul_grad/) | `ComplexMul` 的反向 | `float32` / `ND` | Ascend 910B |
| `FusedBiasSoftmax` | AscendC | `aclnnFusedBiasSoftmax` / `fused_bias_softmax` / `fused_bias_softmax_` | [`ascendc/operators/fused_bias_softmax/`](../../ascendc/operators/fused_bias_softmax/) | `softmax(x + bias1 + bias2, -1)`；Python API 有独立输出和原地两种形式 | `fp32` / `fp16` / `bf16` / `ND` | Ascend 910B |
| `FusedSoftmaxGrad` | AscendC | `aclnnFusedSoftmaxGrad` / `fused_softmax_grad` / `fused_softmax_grad_` | [`ascendc/operators/fused_softmax_grad/`](../../ascendc/operators/fused_softmax_grad/) | 注意力 softmax JVP；Python API 有独立输出和原地两种形式 | `fp32` / `fp16` / `bf16` / `ND` | Ascend 910B |
| `Conv3DGelu` | AscendC | `aclnnConv3dGeluCustom` / `conv3d_gelu` | [`ascendc/operators/conv3d_gelu/`](../../ascendc/operators/conv3d_gelu/) | `gelu_tanh(conv3d(x, weight, bias))` | `float32` / `NCDHW` | Ascend 910B |
| `Conv3DGeluGrad` | AscendC | `aclnnConv3dGeluGrad`（由 `conv3d_gelu` 的反向调用） | [`ascendc/operators/conv3d_gelu_grad/`](../../ascendc/operators/conv3d_gelu_grad/) | `dz = dy * gelu'(z)`。反向先重算 conv3d，再调本算子，再做 conv3d 反向 | `float32` / `ND` | Ascend 910B |
| `CustomGRUV2` | AscendC | `aclnnCustomGRUV2` / `custom_gru_v2` | [`ascendc/operators/custom_gru_v2/`](../../ascendc/operators/custom_gru_v2/) | 单层 GRU 融合 kernel（gate 行序 [r; z; n] 与 PyTorch nn.GRU 一致，tanh 用 2·sigmoid(2x)−1 稳定形式），单向/双向，反向由 Python autograd 承担 | `float32` / `ND` | Ascend 910B |
| `ROIAlignRotated` | AscendC | `aclnnROIAlignRotated` / `roi_align_rotated` | [`ascendc/operators/roi_align_rotated/`](../../ascendc/operators/roi_align_rotated/) | 旋转 ROIAlign：`input[N,C,H,W] + rois[K,6] -> out[K,C,PH,PW]` | `float32` / `ND` | Ascend 910B |
| `ROIAlignRotatedGrad` | AscendC | `aclnnROIAlignRotatedGrad`（反向由 `roi_align_rotated` 的 autograd 调用） | [`ascendc/operators/roi_align_rotated_grad/`](../../ascendc/operators/roi_align_rotated_grad/) | `ROIAlignRotated` 的反向 | `float32` / `ND` | Ascend 910B |
| `real_rfft` | Triton-Ascend | `oil_gas_ops_prospect.real_fft.real_rfft` | [`triton-ascend/real_fft/`](../../triton-ascend/real_fft/) | 实数正向 FFT：`x → (xr, xi)` | NPU kernel 为 `float32` | Ascend 910B |
| `real_irfft` | Triton-Ascend | `oil_gas_ops_prospect.real_fft.real_irfft` | [`triton-ascend/real_fft/`](../../triton-ascend/real_fft/) | 半谱重建实数：`(xr, xi), n → y` | NPU kernel 为 `float32` | Ascend 910B |
| `real_rfft2` | Triton-Ascend | `oil_gas_ops_prospect.real_fft.real_rfft2` | [`triton-ascend/real_fft/`](../../triton-ascend/real_fft/) | 二维实数正向 FFT：`x → (xr, xi)`（组合接口，无新 kernel） | NPU kernel 为 `float32` | Ascend 910B |
| `real_irfft2` | Triton-Ascend | `oil_gas_ops_prospect.real_fft.real_irfft2` | [`triton-ascend/real_fft/`](../../triton-ascend/real_fft/) | 二维半谱重建实数：`(xr, xi), s → y`（组合接口，无新 kernel） | NPU kernel 为 `float32` | Ascend 910B |
| `complex_reciprocal` | Triton-Ascend | `oil_gas_ops_prospect.complex_reciprocal.complex_reciprocal` | [`triton-ascend/complex_reciprocal/`](../../triton-ascend/complex_reciprocal/) | 复数倒数 `1/z`（`z` 或 `xr, xi`） | NPU kernel 为 `complex64` | Ascend 910B |
| `complex_exp` | Triton-Ascend | `oil_gas_ops_prospect.complex_exp.complex_exp` | [`triton-ascend/complex_exp/`](../../triton-ascend/complex_exp/) | 复数指数 `e^z` | NPU kernel 为 `complex64` | Ascend 910B |
| `interpolate` | Triton-Ascend | `oil_gas_ops_prospect.interpolate.interpolate` | [`triton-ascend/interpolate/`](../../triton-ascend/interpolate/) | 2D/3D 线性插值；5D fp32 走 Cube GEMM，4D 走原生 | 5D `float32`；4D 由原生实现支持 | Ascend 910B |
| `spectral_mlp` | Triton-Ascend | `oil_gas_ops_prospect.spectral_mlp.spectral_mlp` | [`triton-ascend/spectral_mlp/`](../../triton-ascend/spectral_mlp/) | 频域低秩门控两层 MLP（复数，实虚分离） | NPU kernel 为 `float32` | Ascend 910B |
| `Conv3d` / `conv3d` | Composite-Torch | `oil_gas_ops_prospect.conv3d_custom.Conv3d` / `.conv3d` | [`composite-torch/conv3d_custom/`](../../composite-torch/conv3d_custom/) | 支持参数域内的 `nn.Conv3d` / `F.conv3d` 兼容实现（自定义前/反向） | 底层 PyTorch 后端支持且经本仓测试的 dtype | PyTorch 后端支持；已测 CPU / Ascend 910B |
| `complex_addcmul` | Composite-Torch | `oil_gas_ops_prospect.complex_addcmul.complex_addcmul` | [`composite-torch/complex_addcmul/`](../../composite-torch/complex_addcmul/) | 复数三输入逐元素 `input + value*tensor1*tensor2`，规避 NPU 上 `torch.addcmul` 缺失复数 dtype | `complex64` / `complex128` | PyTorch 后端支持；已测 Ascend 910B |
| `complex_index_select` | Composite-Torch | `oil_gas_ops_prospect.complex_index_select.complex_index_select` | [`composite-torch/complex_index_select/`](../../composite-torch/complex_index_select/) | 复数版 `torch.index_select`，补齐 NPU 反向报错并绕开前向 AiCPU 慢路径 | `complex64` / `complex128` | PyTorch 后端支持；已测 Ascend 910B |

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

底层 aclnn 接口支持独立输出与输入别名两种形式。Python `fused_bias_softmax` 始终非原地并支持非连续输入；`fused_bias_softmax_` 只接受连续输入并直接写回 `x`。PyTorch 参考实现也不修改输入。

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

底层 aclnn 接口支持独立输出与输入别名两种形式。Python `fused_softmax_grad` 始终非原地并支持非连续输入；`fused_softmax_grad_` 只接受连续输入并直接覆写 `probs`。PyTorch 参考实现也不修改输入。

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

### CustomGRUV2

单层 GRU 的 NPU 融合 kernel，gate 行序 `[r; z; n]` 与 PyTorch `nn.GRU` 一致。前向写出全部时刻的隐状态序列 `y` 与最后时刻隐状态 `output_h`；反向不写自定义 kernel，由 Python 侧 autograd（PyTorch 原生算子构成的计算图）承担。tanh 用 `2·sigmoid(2x)−1` 稳定形式。

#### 数学定义

对时刻 \(t\)（\(h_{t-1} \to h_t\)，门行序 `[r; z; n]`）：

\[
\begin{aligned}
r_t &= \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}) \\
z_t &= \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}) \\
n_t &= \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn})) \\
h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1}
\end{aligned}
\]

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `x` | 输入序列 | `[B, T, F]` | `float32` / `ND` |
| 输入 | `weight_ih` | 输入-隐层权重（按门序 `[r; z; n]` 堆叠） | `[3H, F]`；双向 `[2, 3H, F]` | `float32` / `ND` |
| 输入 | `weight_hh` | 隐-隐层权重 | `[3H, H]`；双向 `[2, 3H, H]` | `float32` / `ND` |
| 输入（可选） | `bias_ih` | 输入侧偏置 | `[3H]`；双向 `[2, 3H]` | `float32` / `ND` |
| 输入（可选） | `bias_hh` | 隐状态侧偏置（aclnn 侧恒传，`None` 时由 Python 包装层补零） | `[3H]`；双向 `[2, 3H]` | `float32` / `ND` |
| 输出 | `y` | 各时刻隐状态序列 | `[num_dirs, B, T, H]` | `float32` / `ND` |
| 输出 | `output_h` | 最后时刻隐状态 | `[num_dirs, B, H]` | `float32` / `ND` |

#### 约束

- 只支持 `float32`、`ND`。
- gate 行序为 `[r; z; n]`，与 `torch.nn.GRU` 一致（`weight_ih` 布局与 `nn.GRU.weight_ih_l0` 相同）。
- tanh 用 `2·sigmoid(2x)−1` 稳定形式计算。
- 精度验收相对 `nn.GRU`：前向阈值 `0.003`（`tests/ascendc/custom_gru_v2/unit/test_forward_precision.py:67`）；`gradcheck` 阈值 `atol=1e-4, rtol=1e-3, eps=1e-6`（`tests/ascendc/custom_gru_v2/unit/test_manual_backward_gradcheck.py:154`）；稳定性为同输入 10 次 `torch.equal` 完全一致（`tests/ascendc/custom_gru_v2/test_plan.md:234`）。
- 性能定额两 shape：`B=1, T=1000, H=24`（单长序列）与 `B=32, T=100, H=24`（批量），见 `tests/ascendc/custom_gru_v2/performance/bench_forward.py`（`test_perf_fwd_001_single_long_seq` / `test_perf_fwd_002_batch`）。
- 测试入口：[`tests/ascendc/custom_gru_v2/`](../../tests/ascendc/custom_gru_v2/)（`run_test.sh` / `run_all_tests.sh`）。
- 性能瓶颈定位与后续优化计划见 [`tests/reports/custom_gru_v2_precision_perf.md`](../../tests/reports/custom_gru_v2_precision_perf.md)（§性能定位与后续优化计划）。

---

### ROIAlignRotated

detectron2 旋转 ROIAlign。AIV-only，前向按输出元素空间 64B slab 切分。

#### 数学定义

对第 \(k\) 个 ROI、通道 \(c\)、输出 bin \((ph,pw)\)：

\[
\begin{aligned}
c_x &= x^{ctr}_k s - 0.5,\quad c_y = y^{ctr}_k s - 0.5 \\
w_r &= w_k s,\quad h_r = h_k s,\quad \theta = a_k\pi/180 \\
y &= y_y\cos\theta - x_x\sin\theta + c_y \\
x &= y_y\sin\theta + x_x\cos\theta + c_x
\end{aligned}
\]

其中 \(y_y = -h_r/2 + ph\cdot h_r/PH + (i_y+0.5)h_r/(PH\cdot g_h)\)，\(x_x\) 同理；
\(g_h,g_w\) 为 `sampling_ratio`（0 时取 \(\lceil h_r/PH\rceil\)、\(\lceil w_r/PW\rceil\)）。
输出为 \(g_h g_w\) 个采样点双线性插值的平均。边界语义严格对齐 detectron2 CUDA：
\(y<-1\) 或 \(y>H\) 或 \(x<-1\) 或 \(x>W\) 返回 0；\(y/x<0\) 夹到 0；越界低索引夹到 \(H-1/W-1\)。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `input` | 特征图 | `[N,C,H,W]` | `float32` / `ND` |
| 输入 | `rois` | `(batch_idx, x_ctr, y_ctr, width, height, angle_deg)` | `[K,6]` | `float32` / `ND` |
| 属性 | `spatial_scale` / `pooled_height` / `pooled_width` / `sampling_ratio` | ROI 缩放与输出 bin | 标量 | — |
| 输出 | `out` | pooled 特征 | `[K,C,PH,PW]` | `float32` / `ND` |

#### 约束

- `input` 4D、`rois` 第 2 维为 6；`pooled_h/pooled_w > 0`，`sampling_ratio >= 0`，`spatial_scale > 0`。
- AIV-only；输出按 64B slab 切分，单 slab 最大 16KB。
- 精度验收：fp32 `max_abs ≤ 1e-5`；独立 golden 为 detectron2 `ROIAlignRotated_cpu.cpp`。
- 测试：[`tests/ascendc/roi_align_rotated/`](../../tests/ascendc/roi_align_rotated/)、[`tests/st/aclnnROIAlignRotated/`](../../tests/st/aclnnROIAlignRotated/)。

---

### ROIAlignRotatedGrad

`ROIAlignRotated` 的反向。AIV-only，按 `grad_input` 元素空间 64B slab 独占切分。

#### 数学定义

\[
g_{n,c,y_l,x_l} \mathrel{+}= \frac{g^{out}_{k,c,ph,pw} w_1}{g_h g_w},\quad \ldots
\]

四个邻域权重 \(w_1..w_4\) 与前向相同；越界索引直接跳过。反向不使用 `atomicAdd`：
每个输出元素只属于一个 slab，核内 UB 累加后整片回写，结果确定。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `grad_output` | \(\partial L/\partial out\) | `[K,C,PH,PW]` | `float32` / `ND` |
| 输入 | `rois` | 同前向 | `[K,6]` | `float32` / `ND` |
| 属性 | `spatial_scale` / `pooled_height` / `pooled_width` / `sampling_ratio` / `batch_size` / `channels` / `height` / `width` | 前向形状与属性 | 标量 | — |
| 输出 | `grad_input` | \(\partial L/\partial input\) | `[N,C,H,W]` | `float32` / `ND` |

#### 约束

- 切分覆盖性由 ophost UT 断言：每个 grad_input 元素恰好被一个核处理一次。
- 精度验收：fp32 `max_abs ≤ 1e-5`；测试同前向目录。

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

---

### real_rfft2

二维实数正向 FFT。**组合接口**：不引入新 kernel，语义与 `torch.fft.rfft2` 一致，返回半谱的实部、虚部两个实数张量。

#### 数学定义

\[
X[k_1,k_2] = \sum_{t_1}\sum_{t_2} x[t_1,t_2]\, e^{-2\pi i (k_1 t_1/h + k_2 t_2/w)},\quad k_2 = 0,\ldots,\lfloor w/2\rfloor
\]

与 `torch.fft.rfft2(x, dim=(d_1,d_2), norm=norm)` 等价；`d_2` 轴输出长度 \(w//2+1\)，`d_1` 轴保持全谱长度。

实现基于两个只要求「实序列 FFT」的恒等式，全部落在已有的一维 `real_rfft` 上：

\[
\mathrm{FFT}(a+ib) = \mathrm{FFT}(a) + i\,\mathrm{FFT}(b),\qquad
\mathrm{IFFT_{ortho}}(z) = \overline{\mathrm{FFT_{ortho}}(\bar z)}
\]

一维 `real_rfft` 只给半谱，进入恒等式前按 \(X[n-k]=\overline{X[k]}\) 补成全谱，补全长度 \(n-(n//2+1)\)。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `x` | 实数信号 | 任意 rank；NPU 路径要求 `float32` | `float32` |
| 输入 | `s` | 两个变换轴的信号长度 | 可选，`None` 沿用输入尺寸 | `int` 序列 |
| 输入 | `dim` | 两个变换轴 | 必须互不相同，半谱在 `dim[1]` | `int` 序列 |
| 输入 | `norm` | 归一化 | `"backward"`（默认）或 `"ortho"` | `str` |
| 输入 | `block_n` / `block_m` / `block_k` | 透传给一维 kernel 的分块参数 | 可选 | `int` |
| 输出 | `xr`, `xi` | 半谱实部、虚部 | `dim[1]` 变为 \(w//2+1\)，`dim[0]` 保持 | `float32` |

#### 约束

- `norm` 只接受 `"backward"` / `"ortho"`，其他值抛 `ValueError`。
- `dim` 必须恰有两个互不相同且在下标范围内的元素。
- `s` 的裁断/零填充语义与 `torch.fft` 一致。
- 非 NPU / Triton 不可用 / 非 `float32` 时回退 `torch.fft.rfft2`。
- 内部按 `ortho` 计算，`backward` 通过整体乘 \(\sqrt{hw}\) 换算。
- 非末轴且长度超过内部阈值（320）时先把轴搬到末轴，避免 strided kernel 编译失败；
  末轴做半谱补全前也会先把该轴搬到 0，避开 910B 上 stride-1 `flip` 的慢路径。
- 组合实现 kernel 数多于原生。910B2 实测按行数 `numel/(h*w)` 分档：行数 ≲64 时慢 3~5 倍，
  约 256 行起反超（N=288 时 1.30×、N=576 时 1.17×）。收益来自实数输出表示与连续布局，
  不是变换本身更快；详见 [`triton-ascend/real_fft/README.md`](../../triton-ascend/real_fft/README.md)。
- 支持 autograd。

---

### real_irfft2

二维实数逆 FFT，是 `real_rfft2` 的逆变换，语义与 `torch.fft.irfft2` 一致。

#### 数学定义

按相反顺序组合：先在 `dim[0]` 上做完整复数逆（共轭反转 + 一次正向 + 共轭），再在 `dim[1]` 上做一维 `real_irfft`。`backward` 归一化通过除以 \(\sqrt{hw}\) 换算。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `xr`, `xi` | 半谱实部、虚部 | 彼此相同；`dim[1]` 为 \(w//2+1\) | `float32` |
| 输入 | `s` | 重建后的两个轴长度 | 可选；`None` 时由谱形状反推 | `int` 序列 |
| 输入 | `dim` | 两个变换轴 | 互不相同 | `int` 序列 |
| 输入 | `norm` | 归一化 | `"backward"`（默认）或 `"ortho"` | `str` |
| 输入 | `block_k` / `block_m` / `block_n` | 透传给一维 kernel 的分块参数 | 可选 | `int` |
| 输出 | `y` | 实数信号 | `dim[0]` 为 \(h\)，`dim[1]` 为 \(w\) | 与 `xr` 相同 |

#### 约束

- `xr.shape` 必须等于 `xi.shape`，否则抛 `ValueError`。
- `dim[1]` 必须是半谱轴；`s` 对半谱轴同样按 torch 语义裁断/零填充。
- 非 NPU / Triton 不可用 / 非 `float32` 时回退 `torch.fft.irfft2`。
- 沿途非末轴 `irfft` 也内置搬轴兜底，长度超过阈值不会触发 kernel 编译失败。
- 支持 autograd。

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


### spectral_mlp

对 rFFT 之后的实虚分离频谱做两层低秩 MLP：第一层 `d → rank → dm` 带 GELU，
第二层 `dm → rank → d` 不带激活。两层的矩阵乘、逐模数门控、偏置与 GELU 融合在
一个 kernel 内完成，两层之间的激活不落 GM。

#### 数学定义

记 `x = xr + i·xi`，`G` 个分组各有一套权重：

\[
\begin{aligned}
X_{1r} &= x_r U_1, & X_{1i} &= x_i U_1 \\
\text{sum}_1 &= X_{1r}\odot g_{1r} + X_{1i}\odot g_{1i}, &
\text{diff}_1 &= X_{1r}\odot g_{1r} - X_{1i}\odot g_{1i} \\
h_{1r} &= \text{diff}_1 V_1 + b_{1r}, & h_{1i} &= \text{sum}_1 V_1 + b_{1i} \\
o_{1r} &= \mathrm{GELU}(h_{1r}), & o_{1i} &= \mathrm{GELU}(h_{1i}) \\
X_{2r} &= o_{1r} U_2, & X_{2i} &= o_{1i} U_2 \\
y_r &= (X_{2r}\odot g_{2r} - X_{2i}\odot g_{2i}) V_2 + b_{2r}, &
y_i &= (X_{2r}\odot g_{2r} + X_{2i}\odot g_{2i}) V_2 + b_{2i}
\end{aligned}
\]

`U1/U2` 收缩特征维，`V1/V2` 收缩低秩维，门控按 `(L, G, rank)` 逐模数、逐分组
生效。GELU 采用 `x / (1 + exp(-1.5957691·(x + 0.044715·x³)))`。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `xr` / `xi` | 频谱实部 / 虚部 | `(B, L, *spatial, G, d)` | `float32` |
| 输入 | `up1` | 第 1 层上投影 | `(G, d, rank)` | `float32` |
| 输入 | `down1` | 第 1 层下投影 | `(G, dm, rank)` | `float32` |
| 输入 | `up2` | 第 2 层上投影 | `(G, dm, rank)` | `float32` |
| 输入 | `down2` | 第 2 层下投影 | `(G, d, rank)` | `float32` |
| 输入 | `gate1_r` / `gate1_i` | 第 1 层门控 | `(L, G, rank)` | `float32` |
| 输入 | `gate2_r` / `gate2_i` | 第 2 层门控 | `(L, G, rank)` | `float32` |
| 输入 | `bias1_r` / `bias1_i` | 第 1 层偏置 | `(G, dm)` | `float32` |
| 输入 | `bias2_r` / `bias2_i` | 第 2 层偏置 | `(G, d)` | `float32` |
| 输入 | `block_m` / `block_m_bwd` | `S` 方向 tile 大小（取 2 的幂并按 UB 钳制） | 标量 | `int` |
| 输出 | `yr` / `yi` | 输出实部 / 虚部 | 同 `xr` | `float32` |

#### 约束

- NPU kernel 仅支持 `float32`；非 NPU、Triton 不可用或任一输入不是 `float32`
  时回退等价的 PyTorch 实现（`einsum` + 同一 GELU 近似）。
- `d / dm / rank` 在 kernel 内向上取到不小于 16 的 2 的幂；空间维可任意组合，
  `S` 在内部展平。
- 支持一阶 autograd；前向不保存中间量，反向从输入重算两层的上投影与激活。
- **显存**：两层之间的激活不落 GM。业务形态 `B=1, L=64, S=128×128, G=8, d=16, dm=16, rank=8`
  下，不再同时驻留的中间量为 `16·B·L·S·G·(rank+dm)` 字节 ≈ 3.0 GB（`64³` 时 ≈ 0.39 GB）。
  前向不保存中间量，训练单步峰值显存 7.17 GB → 4.12 GB（-3.04 GB，-42.5%，基线为原生
  `einsum + npu_gelu`）。
- **性能**：基线为同设备 fp32 的两层 `einsum` + 逐点门控 + 偏置 + `npu_gelu`。
  同形态下前向 28.7 ms → 12.8 ms（2.24×，耗时 -55.3%）；前向+反向
  189.1 ms → 165.0 ms（1.15×，-12.8%）。前向不保存任何中间量，反向从输入
  重算 `x1/h1/o1/x2`；默认 `block_m=512` / `block_m_bwd=256`。
- 验证记录见 [`tests/triton/spectral_mlp/report/`](../../tests/triton/spectral_mlp/report/)。

实现与调优说明详见 [`triton-ascend/spectral_mlp/README.md`](../../triton-ascend/spectral_mlp/README.md)。

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
| 输入 | `x` | 输入特征 | \((N, C_{in}, D, H, W)\) 或无 batch 的 \((C_{in}, D, H, W)\) | 底层 PyTorch 后端支持且经本仓测试的 dtype |
| 输出 | `output` | 输出特征 | \((N, C_{out}, D_{out}, H_{out}, W_{out})\) 或无 batch 的 \((C_{out}, D_{out}, H_{out}, W_{out})\) | 未启用 autocast 时与输入相同；已测 CPU autocast 遵循 PyTorch 目标 dtype |

`kernel_size`、`stride`、数值 `padding`、`dilation` 支持 `int` 或 3 元素序列（元组或列表），`groups` 为 `int`。模块的 `padding_mode` 支持 `zeros` / `reflect` / `replicate` / `circular`。字符串 `padding="same"` / `"valid"` 受支持，会在卷积前换算为显式补边；`"same"` 要求 stride 为 1，否则抛 `ValueError`。当前 `Conv3d` 构造器没有 `nn.Conv3d` 的 `device` / `dtype` 关键字。

#### 约束与行为

- `use_custom` 在检测到 `torch.npu` 接口时默认开启；测试可强制 `module.use_custom = True` 在 CPU 上驱动自定义路径。
- 支持含 batch 的 5D 与无 batch 的 4D 输入，输出保留输入维数，输入梯度保留输入 shape。
- 参数只接受上述受支持类型与取值；数学一致性结论仅适用于该参数域和仓内覆盖的组合。
- `padding_mode != 'zeros'` 时先 `F.pad`，再以 `conv_padding=0` 进入自定义 Function，梯度经 `F.pad` 的 autograd 回传。
- weight/bias 的 state_dict key 与 `nn.Conv3d` 互通。
- dtype 取决于底层 PyTorch 后端能力，并须经过本仓对应 CPU / NPU 测试后才能列为支持。
- 支持复数输入：反向对翻转后的 weight 取共轭，复数权重梯度由实部/虚部各自的实数 `conv3d_weight` 结果重组（`torch.nn.grad.conv3d_weight` 本身只接受实数张量）。
- 在仓内覆盖的特定大 kernel depthwise NPU 形态中，分步反向可避开原生反向曾触发的 L1 buffer 限制；该观察不扩展为所有大 kernel 或任意形态的保证。

#### 验证

- 精度 / 性能用例：[`tests/composite/conv3d_custom/`](../../tests/composite/conv3d_custom/)（仓内列出的参数组合、训练调用形态与性能基准）。
- 已有用例覆盖 CPU float64/float32 组合、NPU float32 的指定形态，以及 CPU complex128/complex64 与 NPU complex64（[`test_conv3d_custom_complex.py`](../../tests/composite/conv3d_custom/test_conv3d_custom_complex.py)，含 CPU complex128 的 `gradcheck` / `gradgradcheck` 双阶反向）；结论仅适用于实际执行并通过的 case，不代表未列出的参数或 dtype。
- 在 PyTorch 2.7.1+cpu 中，float32 输入/参数配合 CPU autocast（`dtype=torch.bfloat16`）的前向及输入/权重/偏置梯度与原生 `F.conv3d` 对拍，原叶子梯度保持 float32。此处仅说明实际测试的 CPU 模式，不据此推定任意低精度输入或 NPU AMP 支持。 本模式随机梯度的 L2 相对误差阈值为 `1e-2`，普通 float32 验收仍为 `1e-5`。
- 复数 NPU 用例在 `torch.npu.conv.allow_hf32 = False` 下执行（结束后恢复原值）。默认 HF32 设置下部分梯度相对误差约 `2e-4`，不按完整 fp32 阈值验收。

实现与使用说明详见 [`composite-torch/conv3d_custom/README.md`](../../composite-torch/conv3d_custom/README.md)。

### complex_addcmul

复数三输入逐元素算子，纯 PyTorch 组合实现，用于规避 NPU 上 `torch.addcmul` 不支持复数 dtype 的限制。

#### 数学定义

\[
\text{out} = \text{input} + \text{value} \cdot \text{tensor1} \cdot \text{tensor2}
\]

复数乘按 \((a+bi)(c+di) = (ac-bd) + (ad+bc)i\) 展开，在「三路输入均为复数」的输入域内与 `torch.addcmul` 数值等价。

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `input` | 被加项 | 任意 rank | `complex64` / `complex128` |
| 输入 | `tensor1`, `tensor2` | 两个乘数，支持广播 | 可与 `input` 不同 | `complex64` / `complex128`，不与 `input` 相同时按 `torch.promote_types` 提升 |
| 输入 | `value` | 乘数标量，默认 `1.0` | 标量 | 实数或复数标量，或 0 维张量（不支持 `requires_grad`） |
| 输出 | — | 三者广播后 shape | 广播结果 | 三者按 `torch.promote_types` 提升 |

#### 约束

- 三个输入必须都是复数；实数输入应直接用 `torch.addcmul`，本算子抛 `TypeError`。原生 `torch.addcmul` 在 CPU 上还允许「复数被加项 × 实数乘数」等混合形态，该输入域**不在**本算子支持范围内。
- 三路 dtype 可不同（如 `complex64` 与 `complex128` 混合），按 `torch.promote_types` 提升后计算。
- 支持 PyTorch 广播规则；反向按标准广播规则把梯度归约回各输入原 shape。
- 复数乘法委托 [`complex_mul`](#complexmul)。`ComplexMul` 只接受 `float32`，因此仅 `complex64` 在 NPU 且 `OIL_GAS_OPS_COMPLEX_MUL_IMPL=ascendc` 时落到 AscendC 算子；`complex128` 始终走 torch 表达式。
- 精度：对拍 CPU 原生 `torch.addcmul`，前向/反向最大绝对误差 `complex64` ≤ 1e-5、`complex128` ≤ 2e-14（浮点舍入级，非逐位一致）。

实现与使用说明详见 [`composite-torch/complex_addcmul/README.md`](../../composite-torch/complex_addcmul/README.md)。

上游补齐 `aclnnAddcmul` 的复数 dtype 后本算子可移除，相关需求见
[cann/oil-gas-ops-prospect#29](https://gitcode.com/cann/oil-gas-ops-prospect/issues/29)。

---

### complex_index_select

复数版 `torch.index_select`，纯 PyTorch 组合实现，用于补齐 NPU 上复数路径的两处缺陷：原生反向（`aclnnIndexAdd`）不支持复数 dtype 直接报错；原生 complex64 前向派发到 AiCPU 标量核（`GatherV2`），大形状比 float32 的 `GatherV3` 慢两个数量级。

#### 数学定义

\[
\text{out}[i_0,\dots,i_{\text{dim}}=k,\dots] = \text{input}[i_0,\dots,i_{\text{dim}}=\text{index}[k],\dots]
\]

沿 `dim` 按 `index` 抽取行，与 `torch.index_select(input, dim, index)` 数值等价。反向对每个输入位置累加其被选中的梯度：

\[
\text{grad\_input}[\ldots, j, \ldots] = \sum_{k:\ \text{index}[k]=j} \text{grad\_out}[\ldots, k, \ldots]
\]

#### 输入输出

| 方向 | 名称 | 含义 | shape | dtype |
|------|------|------|-------|-------|
| 输入 | `input` | 被抽取的复数张量 | 任意 rank（0 维除外）、任意布局 | `complex64` / `complex128` |
| 输入 | `dim` | 抽取的轴 | 标量 | `int`（支持负数） |
| 输入 | `index` | 行索引，可重复、可为空 | 一维 `[M]` | `int32` / `int64` |
| 输出 | — | 抽取结果 | `input` 的 `dim` 轴替换为 `M` | 与 `input` 相同 |

#### 约束

- 输入必须为复数；实数 gather 在任何后端都原生可用，应直接用 `torch.index_select`，本算子抛 `TypeError`。
- 连续输入且 `dim < input.dim() - 1` 时走 flat 路径：`view(float)` 后行宽翻倍、行数不变，gather 字节访问模式与「复数感知 kernel」相同，带宽与 float32 基线一致；末轴 `dim`（含 1-D 张量）或非连续输入走 real/imag 路径（实部、虚部各 gather 一次再重组），仍远快于原生 AiCPU 路径。
- 反向是 `zeros + index_add_`（在实数视图上），与原生 `index_select` 的反向同构；重复索引按出现次数累加，一阶梯度与原生一致。二阶导不可用，与原生一致（反向公式不含可微保存张量）。
- 精度：对拍 CPU 原生 `torch.index_select`，前向为纯 gather 逐位一致；反向最大绝对误差 `complex64` ≤ 1e-5、`complex128` ≤ 2e-14（910B2 实测含重复索引约 2.5e-7）。
- 性能（910B2 / CANN 9.0.0 / torch_npu 2.7.1，complex64 前向 vs 原生，公开 API 实测口径）：频率体 `(4,64,80,144,144)×dim=1×M=16` 约 115~135×（173~189ms → 1.4ms），行聚合 `(65536,1024)×dim=0×M=16384` 约 124~144×，长条 `(2^20,4)×dim=0×M=2^18` 约 113~124×，中块 `(16384,256)×dim=0×M=8192` 约 30~39×（该规模 Python 派发开销占比高），1-D 兜底 `(2^20,)×M=2^17` 约 22~23×（real/imag 路径）；kernel 级带宽证据（`GatherV3` AiCore 47.3µs vs `GatherV2` AiCPU 9.66ms）见 ISSUE #61。反向（前向+反向合计，原生对复数报错无基准）与同字节量 float32 原生基线对比：大形状（≥268MB 搬移量）实测 0.99~1.00×，中块（33MB）0.68×——差异为常数级 Python 派发与 view 开销（fwd 单独实测 0.112ms vs 0.028ms），随数据量摊薄，非带宽损失；该比值可由 `verify_complex_index_select.py --mode perf` 的 `fwdbwd-baseline` 行复现。
- 索引值越界 / 为负的行为跟随底层后端，与原生一致。
- 验证记录见 ISSUE #61 与 `tests/composite/complex_index_select/`。

实现与使用说明详见 [`composite-torch/complex_index_select/README.md`](../../composite-torch/complex_index_select/README.md)。
