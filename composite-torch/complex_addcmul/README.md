# complex_addcmul

支持复数的三输入逐元素算子，纯 PyTorch 组合实现（无自定义 kernel）：

```python
from torch import addcmul                                            # 原生：NPU 上不支持复数
from oil_gas_ops_prospect.complex_addcmul import complex_addcmul    # 本仓替换
```

## 何时用

`torch.addcmul` 在 NPU 上对 `complex64` / `complex128` 直接报错：

```
RuntimeError: addcmul:.../AddcmulKernelNpuOpApi.cpp:59
NPU function error: call aclnnAddcmul failed, error code is 161002
AclNN_Parameter_Error(EZ1001): Tensor self not implemented for DT_COMPLEX64,
should be in dtype support list [DT_FLOAT, DT_INT32, ...]
```

CPU 后端正常，实数 dtype（`float32` / `float16` / `bfloat16` / 整数）在 NPU 上也正常。**只有复数 dtype 缺失**，本算子就是为这一情形提供的替代。

实数输入请直接调 `torch.addcmul`，本算子不转发、会抛 `TypeError`。

## 行为

语义与 `torch.addcmul(input, tensor1, tensor2, value=value)` 一致，但支持域限定为「三路输入均为复数」：

```
out = input + value * tensor1 * tensor2
```

- 三个输入均须为复数（`complex64` / `complex128`）；dtype 可以不同，按 `torch.promote_types` 提升；
- 原生 `torch.addcmul` 在 CPU 上还允许「复数 × 实数」等混合形态，该输入域本算子不支持，会抛 `TypeError`；
- 支持 PyTorch 广播规则，输出 shape 为三者广播结果；
- `value` 可以是实数或复数标量，也接受 0 维张量（张量不支持 `requires_grad`：本算子不对 `value` 求导，传了会抛 `TypeError`）；
- 支持 autograd，反向遵循 PyTorch 复数约定 `dL/dt = conj(value) * grad_output * conj(t)`，广播维度按标准规则归约回原 shape。

## 实现

复数乘法交给 [`complex_mul`](../../ascendc/pybind/complex_mul.py)：

- `complex_mul` 在 NPU 且 `OIL_GAS_OPS_COMPLEX_MUL_IMPL=ascendc`（或 `auto`）时调用 AscendC `ComplexMul`，否则用等价 torch 表达式；
- **`ComplexMul` 只接受 `float32`**，因此只有 `complex64` 会落到 AscendC 算子，`complex128` 始终走 torch 表达式。

算子内部先把三路输入广播到输出 shape 再 `contiguous()`，因为 `complex_mul` 要求六路同 shape 且连续。反向的共轭通过与 `complex_mul` 的虚部取负实现，避免构造 conj 视图（NPU 上带 conj 位的张量参与运算前会被强制物化）。

注意：本实现是多次 torch 算子组合（广播、`contiguous`、复数乘、标量乘、加、实虚重组），**不是**单个融合 kernel。它的目的是解决复数 `addcmul` 的可用性，而非替代一个理想的三输入融合算子。

## 精度

对拍 CPU 原生 `torch.addcmul`（多组 shape、广播、`value` 取值，CPU 与 NPU 各测一轮）：

| dtype | 前向 max\|Δ\| | 反向 max\|Δ\| |
|---|---|---|
| `complex64` | ≤ 1e-5 | ≤ 1e-5 |
| `complex128` | ≤ 2e-14 | ≤ 2e-14 |

均为浮点舍入级差异，**不是逐位一致**（复数乘的展开顺序与原生 kernel 不同）。

用例见 [`tests/composite/complex_addcmul/`](../../tests/composite/complex_addcmul/)：

- `test_complex_addcmul.py`：pytest 用例，CPU 部分在任何环境都跑，NPU 部分自动 skip；
- `verify_complex_addcmul.py`：`tests/suites.json` 的 `npu_python` 入口，支持 `--mode all|precision|perf`。

## 备注

上游补齐 `aclnnAddcmul` 的复数 dtype 后，本算子可直接移除。`TestNPU::test_native_addcmul_still_rejects_complex` 断言了「NPU 原生 `torch.addcmul` 对复数仍报错」，届时该用例会失败，作为移除信号。

相关需求：[cann/oil-gas-ops-prospect#29](https://gitcode.com/cann/oil-gas-ops-prospect/issues/29)。
