# 频域低秩门控两层 MLP Triton-Ascend 算子

对 rFFT 之后的实虚分离频谱做两层低秩 MLP：第一层 `d → rank → dm` 带 GELU，
第二层 `dm → rank → d` 不带激活。两层之间的小矩阵乘、逐模数门控、偏置与
GELU 全部融合进**一个 kernel**，中间激活不落 GM。

用于地震反演/全波形反演里沿单个频率轴的通道低秩变换（每个分组一套共享权重、
每个频率模数与分组一组门控）。

Python 接口：`oil_gas_ops_prospect.spectral_mlp.spectral_mlp`

## 数学定义

输入 `xr, xi` 为频谱实部/虚部，形状 `(B, L, *spatial, G, d)`；`G` 个分组各
有一套权重，门控按 `(L, G, rank)` 逐模数、逐分组生效：

```text
X1r = xr · U1                 X1i = xi · U1
sum1 = X1r ⊙ g1r + X1i ⊙ g1i
diff1 = X1r ⊙ g1r − X1i ⊙ g1i
h1r = diff1 · V1 + b1r        h1i = sum1 · V1 + b1i
o1r = GELU(h1r)               o1i = GELU(h1i)

X2r = o1r · U2                X2i = o1i · U2
sum2 = X2r ⊙ g2r + X2i ⊙ g2i
diff2 = X2r ⊙ g2r − X2i ⊙ g2i
yr  = diff2 · V2 + b2r        yi  = sum2 · V2 + b2i
```

- 两次收缩都按分组 `g` 做：`U1/U2` 收缩特征维 `d`，`V1/V2` 收缩低秩维 `rank`。
- 实部/虚部先各自乘门控，再按 `diff/sum` 重组后做第二次投影，这是复数线性层
  在实虚分离表示下的写法。
- GELU 采用 `x / (1 + exp(-1.5957691 · (x + 0.044715 · x³)))`，Python 回退路径
  与 kernel 使用同一公式。

## 接口

```python
from oil_gas_ops_prospect.spectral_mlp import spectral_mlp

yr, yi = spectral_mlp(
    xr, xi,
    up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
    up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
    block_m=512, block_m_bwd=256,
)
```

| 输入 | 形状 | 说明 |
| --- | --- | --- |
| `xr` / `xi` | `(B, L, *spatial, G, d)` | 频谱实部 / 虚部，`float32` |
| `up1` | `(G, d, rank)` | 第 1 层上投影 |
| `down1` | `(G, dm, rank)` | 第 1 层下投影 |
| `up2` | `(G, dm, rank)` | 第 2 层上投影 |
| `down2` | `(G, d, rank)` | 第 2 层下投影 |
| `gate1_r` / `gate1_i` | `(L, G, rank)` | 第 1 层实部 / 虚部门控 |
| `gate2_r` / `gate2_i` | `(L, G, rank)` | 第 2 层实部 / 虚部门控 |
| `bias1_r` / `bias1_i` | `(G, dm)` | 第 1 层实部 / 虚部偏置 |
| `bias2_r` / `bias2_i` | `(G, d)` | 第 2 层实部 / 虚部偏置 |

输出 `(yr, yi)`，形状同 `xr`，`float32`。全部输入与参数支持一阶 autograd。

`block_m` / `block_m_bwd` 是每个 tile 在展平后的 `S` 轴上的长度，会向下取到
2 的幂并按 UB 预算钳制；默认 512 / 256（128³ 实测最优），换 shape 时优先扫这两个参数。

## 实现要点

- **einsum 当 GEMM**：每个分组把两次收缩写成 `tl.dot`，跑在 Cube 上。
- **单 kernel 融合**：一个逻辑 tile = 一个 `(b, l, g)` 三元组加上 `S` 轴的一段
  连续切片；tile 内门控与偏置是常量，可直接广播，整层无需落中间张量。
- **persistent grid**：grid = 物理 AI Core 数，核内循环处理逻辑 tile；`S` 的
  尾块按 mask 处理。
- **维度对齐**：`d / dm / rank` 在 kernel 内向上取到不小于 16 的 2 的幂，满足
  `tl.arange` 与 Cube 的 16 对齐要求；非 2 的幂的取值会被 padding 处理。
- **前向零中间量**：前向只读输入、写输出，不保存 `x1/h1/o1/x2`；反向用 6 个
  小 GEMM 从输入重算。训练显存只与输入/输出有关，前向也不再产生窄行
  （`(S, rank)`）的 GM 写。
- **梯度归约**：权重/门控/偏置梯度按 `(核, 槽位)` 写部分和，再由 host 求和。
  既避免跨核原子竞争，也把单槽顺序累加长度压到 1/8，bias/gate 这类跨 `S`
  全量规约的 fp32 舍入误差因此降到 1e-7 量级。

## 收益（910B2 空闲卡，业务形态 128³）

基线为同设备 fp32 的两层 `einsum` + 点门控 + 偏置 + `npu_gelu`：

| 项 | 基线 | 本算子 |
| --- | ---: | ---: |
| 前向 | 28.7 ms | **12.8 ms（2.24×）** |
| 前向+反向 | 189.1 ms | **165.0 ms（1.15×）** |
| 训练单步峰值显存 | 7168 MB | **4124 MB（-42.5%）** |
| 两层中间量（不再驻留） | — | **3.0 GB** |

## 回退与约束

- 非 NPU、Triton 不可用、或任一输入不是 `float32` 时，回退到等价的 PyTorch
  参考实现（`einsum` + 同一 GELU 近似），接口语义与 NPU 路径一致。
- NPU kernel 仅支持 `float32`；所有输入需同设备。
- 空间维可任意组合（`S` 内部展平），`B`、`L`、`G` 无特殊约束。
- `d / dm / rank` 建议不超过 256，且 padding 后同时驻留的 tile 不能超出 UB；
  `_safe_block_m` 会按 160KB 可用 UB 自动折减 `block_m`。

## 验证

精度、性能与显存记录见 [`tests/triton/spectral_mlp/report/`](../../tests/triton/spectral_mlp/report/)：

```bash
# CPU 回退 + NPU 前/反向精度（pytest）
python3 -m pytest tests/triton/spectral_mlp/test_spectral_mlp.py -q

# 精度 + 性能，原始结果写 results.json
python3 tests/triton/spectral_mlp/verify_spectral_mlp.py --mode all --img 128 \
  --iters 20 --repeats 5 --json tests/triton/spectral_mlp/report/results.json

# 训练单步峰值显存（两条路径分别起进程）
python3 tests/triton/spectral_mlp/bench_memory.py --mode ref
python3 tests/triton/spectral_mlp/bench_memory.py --mode op

# AOT 编译门禁（前向与反向）
python3 triton-ascend/tools/aot_compile_check.py --filter spectral_mlp
```
