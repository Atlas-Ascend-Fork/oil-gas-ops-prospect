# complex_exp 验证报告

本目录保存 `complex_exp` 算子的测试环境、精度与性能记录。正式的算子文档
（`docs/{zh,en}/op_list.md`、`api_list.md`、`triton-ascend/complex_exp/README.md`）
只描述功能、接口和约束，不重复这里的数值。

## 文件

| 文件 | 内容 |
| --- | --- |
| `README.md` | 测试环境、覆盖范围、复现命令与已知限制 |
| [`precision_perf.md`](precision_perf.md) | 精度与性能结果表、结论 |
| [`results.json`](results.json) | `verify_complex_exp.py --json` 的原始输出，含环境、容差、逐条精度/性能数据与最终 `status` |

## 测试环境

| 项目 | 内容 |
| --- | --- |
| 主机 | `bmswt001` |
| 设备 | Ascend 910B2，使用 NPU 0 |
| CANN | 9.0.0 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post4 |
| Triton | `triton-ascend` 3.2.0 |
| 测试日期 | 2026-10-28 |
| 测试入口 | `tests/triton/complex_exp/verify_complex_exp.py`、`tests/triton/complex_exp/test_complex_exp.py` |

**该卡与其他容器共享**：采集时 8 张卡的 AICore 占用率都在 94%–96%，每卡约
23 GB HBM 被其他进程占用。因此本目录的性能数值只用于确认量级和「相对同设备
`torch.exp` 的加速比」，绝对值会随邻居负载波动，不能当作独占卡的峰值带宽。
同一份输入重复测量时，`304×304×176` 的原生基线在 160 ms 至 292 ms 之间变化，
就是这个原因。

## 覆盖范围

精度（均以 CPU `float64` 为参考）：

- 前向：`13`、`1000001`、`64³`、`304×304×176`（业务形态，16.3 M 元素）、`4096²`，
  同时对比 Triton kernel 与 NPU 原生 `torch.exp`。
- 反向：`4096²` 的 `grad_z = g·conj(exp(z))`，对照组为同设备 torch autograd
  与 `float64` 闭式解。
- 二阶导：`513×257` 经 `create_graph=True` 后再求一次导，对 `float64`。
- 特殊值：`pytest` 用例覆盖纯实数上溢、`exp(a)` 溢出带（`a > ln(FLT_MAX)`）、
  `b = ±inf` / NaN 的 `cexpf` 特判表，以及第 0 维标量、空张量、非连续输入、
  conjugate-view 梯度、AOT 分支对拍。

未覆盖：

- `|b| >= 2^25` 的大相位角（后端 `tl.sin` / `tl.cos` 缺参数归约，见下文限制）。
- 混合精度（kernel 只接受 `complex64`，其余 dtype 回退 `torch.exp`）。

## 复现

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

# pytest：CPU 回退逐位一致 + NPU 前/反向精度 + 特殊值 + 二阶导 + AOT 分支对拍
python3 -m pytest tests/triton/complex_exp/test_complex_exp.py -q

# 精度 + 性能，原始结果写入 results.json
python3 tests/triton/complex_exp/verify_complex_exp.py --mode all --rtol 1e-5 \
  --json tests/triton/complex_exp/report/results.json

# 只跑精度（npu_python 套件协议，tests/suites.json）
python3 tests/triton/complex_exp/verify_complex_exp.py --mode precision --rtol 1e-5

# AOT 编译门禁：前向与反向各含 int32 / int64 偏移分支
python3 triton-ascend/tools/aot_compile_check.py --filter complex_exp
```

## 已知限制

- **大相位角无归约**：triton-ascend 后端的 `tl.sin` / `tl.cos`（以及 CANN
  `libdevice` 版本）在 `|b| >= 2^25`（33554432）之后不再做参数归约，结果与
  `torch.exp` 不一致。实测 `|b|` 在 `[2^25, 2^34)` 时误差从 `1.6e-3` 涨到
  `1e36` 量级；该后端不支持 fp64（`vcast` 无 float→double），无法在 kernel 内
  自行做归约。本算子目标场景（地震反演的相位因子）角度较小；虚部数值极大时
  请改用 `torch.exp`。
- **小张量有固定开销**：kernel 每次调用约 0.26–0.4 ms 的 host 派发/launch 成本，
  元素数低于约 `3×10^4` 时慢于 `torch.exp`（见 `precision_perf.md` 的 13 元素行）。
- **测试环境共享**：见上文的卡片占用说明，性能绝对值不宜作为独占卡结论。
