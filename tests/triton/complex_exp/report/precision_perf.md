# complex_exp 精度与性能结果

测试环境、覆盖范围与复现命令见 [`README.md`](README.md)；逐条原始数据见
[`results.json`](results.json)。测试日期 2026-10-28，Ascend 910B2 / CANN 9.0.0 /
torch 2.7.1+cpu / torch_npu 2.7.1.post4 / triton-ascend 3.2.0。

## 前向精度

参考为 CPU `float64` 的 `exp(z)`，误差按 `|got-ref| / max(|ref|, 1e-30)` 逐元素
取最大值。阈值 `max_rel <= 1e-5`。同时列出同设备 NPU 原生 `torch.exp` 作为对照，
用于判断误差来自 kernel 还是来自 complex64 本身的量化。

| shape | 元素数 | 方法 | max_abs | max_rel | 结果 |
| --- | ---: | --- | ---: | ---: | --- |
| `[13]` | 13 | triton | 1.047e-07 | 9.846e-08 | PASS |
| `[13]` | 13 | torch NPU exp | 1.091e-07 | 1.008e-07 | 对照 |
| `[1000001]` | 1.0e6 | triton | 1.000e-05 | 2.373e-07 | PASS |
| `[1000001]` | 1.0e6 | torch NPU exp | 1.000e-05 | 1.426e-07 | 对照 |
| `64×64×64` | 2.6e5 | triton | 1.566e-05 | 2.442e-07 | PASS |
| `64×64×64` | 2.6e5 | torch NPU exp | 4.071e-06 | 1.365e-07 | 对照 |
| `304×304×176` | 1.6e7 | triton | 1.464e-05 | 2.625e-07 | PASS |
| `304×304×176` | 1.6e7 | torch NPU exp | 8.233e-06 | 1.450e-07 | 对照 |
| `4096×4096` | 1.7e7 | triton | 1.444e-05 | 2.647e-07 | PASS |
| `4096×4096` | 1.7e7 | torch NPU exp | 1.330e-05 | 1.444e-07 | 对照 |

Triton 的 `max_rel` 与原生 `torch.exp` 同量级（`2.6e-7` 对 `1.4e-7`），差异主要
来自 `tl.exp` / `tl.cos` / `tl.sin` 与 Ai_Cpu `aclnnExp` 的末位舍入，不是语义偏差。
`max_abs` 的绝对值随元素数增长（符合随机采样的极值行为），10 万到 1700 万元素都
稳定在 `1.5e-5` 以内。

## 反向与二阶导精度

`grad_z = g·conj(exp(z))`，形状 `4096×4096`。参考分别是同设备 torch autograd
（fp32）与 CPU `float64` 闭式解 `g·conj(exp64(z))`。

| 项目 | 参考 | 阈值 | 实测 max_rel | 结果 |
| --- | --- | ---: | ---: | --- |
| 一阶反向 | 同设备 torch fp32 autograd | 1e-4 | 4.092e-07 | PASS |
| 一阶反向 | CPU `float64` 闭式解 | 1e-4 | 3.102e-07 | PASS |
| 二阶反向（`create_graph=True` 后再求导） | CPU `float64` | 1e-4 | 2.924e-07 | PASS |

二阶导在 `513×257` 上与 `float64` 双阶参考对齐；`pytest` 中还有一组
`test_npu_double_backward_matches_float64` 覆盖同样链路。

## 特殊值与语义对拍（pytest）

`test_npu_forward_pure_real_overflow_keeps_imag_zero`、
`test_npu_forward_overflow_band_matches_torch`、
`test_npu_forward_special_values_match_torch` 三个用例把实部/虚部分别逐元素比对，
NaN 视为相等、有限值给 `1e-6` 相对容差（三角函数末位差异），inf 与 0 的符号位
严格比对。覆盖的关键点：

| 输入 | 期望（`torch.exp` / `cexpf`） | 修复前 |
| --- | --- | --- |
| `89+0j`、`100+0j`、`inf+0j` | `inf+0j` | `inf+nanj` |
| `100+1e-30j` | `inf+2.688e13j` | `inf+infj` |
| `89+1j` | `2.426e38+infj`（实部有限） | `inf+infj` |
| `nan+0j` | `nan+0j` | `nan+nanj` |
| `inf+infj`、`inf+nanj` | `inf+nanj` | `nan+nanj` |

复现整份用例（`-q` 后加 `--tb=line` 可只看失败行）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python3 -m pytest tests/triton/complex_exp/test_complex_exp.py -q
```

本次测得 `52 passed, 1 warning in 7.88s`；唯一告警来自测试自身的
`sum().backward()` 内部 `torch.ones_like`，不是算子代码。

## 前向性能

同步计时，预热 5 次 + 20 次调用的墙钟平均，含 host 派发/launch 开销。
加速比 = 同设备原生 `torch.exp` 时延 / Triton 时延。等效带宽按每元素读 8 B + 写 8 B。

| shape | 元素数 | 原生 ms | Triton ms | Triton GB/s | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[13]` | 13 | 0.0451 | 0.4081 | ~0 | 0.11x |
| `[1000001]` | 1.0e6 | 12.3646 | 0.2854 | 56.1 | 43.33x |
| `64×64×64` | 2.6e5 | 3.7054 | 0.3040 | 13.8 | 12.19x |
| `304×304×176` | 1.6e7 | 250.1893 | 2.2043 | 118.1 | 113.50x |
| `4096×4096` | 1.7e7 | 254.8590 | 2.5376 | 105.8 | 100.43x |

- 13 元素时 Triton 慢 9 倍，是固定的约 0.4 ms host 开销所致，属已知取舍；
  该算子面向大张量。
- `64³` 只有 2.6e5 元素，加速比降到 12x，同样是固定开销占主导。
- 大张量上 Triton 稳定在 `1e2 GB/s` 量级，而原生 `aclnnExp` 走 Ai_Cpu，
  等效带宽只有约 `1 GB/s`。

## 反向性能

业务形态 `304×304×176`（1.6e7 元素），读 `w` + `g` 共 16 B/元素、写 `gz` 8 B/元素。

| 项目 | 实测 |
| --- | ---: |
| 前向 | 2.5826 ms |
| 前向 + 反向 | 3.6899 ms |
| 前向带反向的等效带宽 | 176.3 GB/s |

即反向 kernel 本身约 1.1 ms，与前向同量级（都是纯连续访存、无重算 `exp`）。

## 结论

- 前向、一阶反向、二阶反向都在阈值内（前向 `1e-5`，反向与二阶 `1e-4`），且与
  同设备原生实现的误差同量级。
- 特殊值语义已对齐 CPU `cexpf`，包括纯实数上溢不再产生 `inf+nanj`、`exp(a)`
  溢出带内实部不再被误判为上溢。`pytest` 共 52 个用例，NPU 上全部通过（52
  passed）；无 NPU 环境下 8 个 CPU 用例通过、44 个自动 skip。新增的 5 个
  语义/二阶导用例在修复前的实现上全部失败，修复后全部通过。
- AOT 编译门禁覆盖前向与反向的 int32/int64 共 4 个分支，全部通过。
- 性能结论只在上述共享卡环境下成立：加速比量级（大张量 100x 以上）稳定，绝对值
  随邻居负载波动。虚部 `|b| >= 2^25` 时结果不可用，见 `README.md` 的已知限制。
