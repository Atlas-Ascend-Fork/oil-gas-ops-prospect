# interpolate 验证报告

本目录保存 `interpolate` 算子的精度、性能与测试环境记录。正式算子文档只描述
功能、接口和约束，不重复这里的数值。

## 测试环境

| 项目 | 内容 |
| --- | --- |
| 主机 | `bmswt015`（`172.18.0.117`） |
| 设备 | Ascend 910B2，单卡独占 |
| CANN | 9.0.0 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post4 |
| Triton-Ascend | `triton-ascend` 3.2.1 |
| 测试日期 | 2026-10-27 |
| 测试入口 | `tests/triton/interpolate/verify_interpolate.py` |

在该环境上先做了 **native-only 环境体检**：只调用 NPU 原生
`F.interpolate`，同一份 5D 输入连续 100 次与 CPU 参考对拍，全部
`max_abs = 4.77e-7`，输入前后 `xdiff = 0`。确认测试环境稳定后，才执行
Cube 路径测试。

## 覆盖范围

- 5D `float32` `trilinear`，`align_corners=False`。
- 前向：C=32/C=256 的 `33³→128³`、`41³→160³`。
- 反向：C=32/C=64 的 `33³→128³`、`41³→160³`。C=256 反向需要数 GiB
  梯度/workspace，未在默认验证中执行。
- 4D `bilinear` 走原生 `F.interpolate`，不是 Cube 路径。

小于默认 128 MiB 阈值的 shape 使用原生回退；验证脚本通过私有测试参数强制
Cube 时，也只在生产尺寸上执行。

## 复现

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

# 前向 + 反向精度，以及前向/反向/训练 step 性能
python3 tests/triton/interpolate/verify_interpolate.py --mode all --full \
  --json tests/triton/interpolate/report/results.json

# 只跑精度
python3 tests/triton/interpolate/verify_interpolate.py --mode precision
```

验证脚本默认通过私有测试参数强制走 Cube 路径并关闭 128 MiB 阈值；这些参数不是
算子公开环境变量。详细数值见 [`precision_perf.md`](precision_perf.md)。
