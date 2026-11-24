# complex_index_select

复数版 `torch.index_select`，纯 PyTorch 组合实现（无自定义 kernel）：

```python
from torch import index_select                                                     # 原生：NPU 反向报错、前向 AiCPU 慢路径
from oil_gas_ops_prospect.complex_index_select import complex_index_select          # 本仓替换
```

## 何时用

NPU 上复数 `index_select` 有两处缺陷（910B2 / CANN 9.0.0 / torch_npu 2.7.1 实测）：

1. **反向直接报错**——autograd 反向落在 `aclnnIndexAdd`，dtype 支持列表不含复数：

```
RuntimeError: .../IndexAddKernelNpuOpApi.cpp...
NPU function error: call aclnnIndexAdd failed, error code is 161002
AclNN_Parameter_Error(EZ1001): Tensor self not implemented for DT_COMPLEX64,
should be in dtype support list [DT_FLOAT, DT_INT32, ...]
```

2. **前向派发到 AiCPU**——complex64 走 `GatherV2`（AI_CPU 标量核），float32 走 `GatherV3`（AI_VECTOR_CORE）。同形状 kernel 耗时相差约两个数量级（中块形状实测 9.66ms vs 47.3µs），大形状经公开 API 端到端慢 22~144 倍（见下表）。

CPU 后端完全正常。实数 dtype 在 NPU 上也正常且走快路径，**请直接用 `torch.index_select`**，本算子不转发、会抛 `TypeError`。

## 行为

语义与 `torch.index_select(input, dim, index)` 一致：

- `input` 任意 rank（0 维除外）、任意布局的复数张量（`complex64` / `complex128`）；
- `dim` 支持负数，归一化规则与 torch 一致；越界抛 `IndexError`；
- `index` 一维 `int32` / `int64`，可重复、可为空；越界 / 负值行为跟随底层后端，与原生一致；
- 支持 autograd：反向对每个输入位置累加其被选中的梯度（重复索引按出现次数累加）；
- 二阶导不可用，与原生一致（反向公式只消费 `grad_output`，不含可微保存张量）。

## 实现

两条路径按输入自动选择，CPU / NPU 行为一致：

- **flat 路径**（输入连续且 `dim < input.dim() - 1`）：complex64 在内存中是两个相邻
  float32，`view(torch.float32)` 后行宽翻倍、行数不变，沿 `dim` gather 的字节访问
  模式与「复数感知 kernel」完全相同，带宽与 float32 基线一致；反向在 float 视图上
  `zeros + index_add_`。`dim` 为末轴时该视图会拆散实虚对，不能用。
- **real/imag 路径**（末轴 `dim`、1-D 张量或非连续输入）：实部、虚部各 gather 一次再
  `torch.complex` 重组。stride-2 访问比 flat 路径慢，但仍远快于原生 AiCPU 路径。

反向用 `zeros + index_add_`，与原生 `index_select` 的反向同构（在实数视图上执行，
绕开 `aclnnIndexAdd` 的复数限制）。

注意：本实现的目标是复数 `index_select` 的**可用性**（反向从报错变为可用）与前向
**性能**（绕开 AiCPU），不是替代一个理想的融合 kernel。

## 精度

对拍 CPU 原生 `torch.index_select`（CPU 与 NPU 各测一轮）：

| dtype | 前向 max\|Δ\| | 反向 max\|Δ\| |
|---|---|---|
| `complex64` | 0（逐位一致） | ≤ 1e-5（910B2 实测含重复索引约 2.5e-7） |
| `complex128` | 0（逐位一致） | ≤ 2e-14 |

前向是纯 gather（无算术），逐位一致；反向差异为浮点累加顺序的舍入级。

## 性能（910B2 / CANN 9.0.0 / torch_npu 2.7.1，complex64 前向 vs 原生）

`verify_complex_index_select.py --mode perf` 的实测口径（公开 API 端到端，含 Python
派发开销；原生侧 AiCPU 耗时有波动，取多次复测区间）：

| 形态 | shape | dim / M | 原生 | 本算子 | 加速比 |
|---|---|---|---|---|---|
| 频率体（FWI 频率选择） | `(4, 64, 80, 144, 144)` | 1 / 16 | 173~189 ms | ~1.4 ms | ~115~135× |
| 行聚合大块 | `(65536, 1024)` | 0 / 16384 | ~30.5~33 ms | ~0.24 ms | ~124~144× |
| 长条 | `(2^20, 4)` | 0 / 2^18 | ~26~28 ms | ~0.23 ms | ~113~124× |
| 中块 | `(16384, 256)` | 0 / 8192 | ~6.0~6.7 ms | ~0.20 ms | ~30~39× |
| 1-D 兜底（real/imag 路径） | `(2^20,)` | 0 / 2^17 | ~9.3~10.1 ms | ~0.43 ms | ~22~23× |

- 中块等小形态处于 Python 派发开销主导区（每次调用约 0.1~0.15ms），加速比随原生
  耗时走，不能反映 kernel 带宽差异；kernel 级证据（中块形状 `GatherV3` AiCore
  47.3µs vs `GatherV2` AiCPU 9.66ms）见 ISSUE #61 报告。
- 反向（前向 + 反向合计，原生对复数报错无基准）：中块 ~0.6ms、行聚合 ~1.0~1.1ms、
  长条 ~0.6ms、1-D ~1.15ms、频率体 ~26ms；与同字节量 float32 原生 `index_select`
  前向+反向基线对比：大形状（≥268MB 搬移量）实测 0.99~1.00×，中块（33MB）0.68×
  ——差异为常数级 Python 派发与 view 开销（fwd 单独实测 0.112ms vs 0.028ms），
  随数据量摊薄，非带宽损失（该比值可由 `verify_complex_index_select.py --mode
  perf` 的 `fwdbwd-baseline` 行复现）。反向零张量按 float dtype 分配——
  NPU 上 complex dtype 的 fill 会落到 AiCPU 慢路径（512MB 实测 39ms vs float
  0.4ms）。

用例见 [`tests/composite/complex_index_select/`](../../tests/composite/complex_index_select/)：

- `test_complex_index_select.py`：pytest 用例，CPU 部分在任何环境都跑，NPU 部分自动 skip；
- `verify_complex_index_select.py`：`tests/suites.json` 的 `npu_python` 入口，支持 `--mode all|precision|perf`。

## 备注

上游补齐 `aclnnIndexAdd` 的复数 dtype（并让 complex64 前向离开 AiCPU）后，本算子可
直接移除。`TestNPU::test_native_backward_still_rejects_complex` 断言了「NPU 原生反向
对复数仍报错」，届时该用例会失败，作为移除信号。

相关需求：[cann/oil-gas-ops-prospect#61](https://gitcode.com/cann/oil-gas-ops-prospect/issues/61)。
