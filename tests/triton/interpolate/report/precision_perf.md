# interpolate 精度与性能结果

测试环境见 [`README.md`](README.md)。以下数值为同设备 `F.interpolate` 对比，
前向/反向容差均为 `max_abs <= 1e-5`。

## 前向精度

| shape | max_abs | 结果 |
| --- | ---: | --- |
| C=32 `33³→128³` | 4.77e-7 | PASS |
| C=32 `41³→160³` | 4.77e-7 | PASS |
| C=256 `33³→128³` | 4.77e-7 | PASS |
| C=256 `41³→160³` | 7.15e-7 | PASS |
| C=128 `65²→256²` bilinear | 0 | PASS（原生路径） |

4D `bilinear` 默认走原生 `F.interpolate`，因此该行不覆盖 Cube kernel；保留它
是为了记录 2D 行为，而不是把 0 误差归因于 Cube。

## 反向精度

| shape | grad max_abs | 结果 |
| --- | ---: | --- |
| C=32 `33³→128³` | 1.91e-6 | PASS |
| C=32 `41³→160³` | 2.86e-6 | PASS |
| C=64 `33³→128³` | 1.91e-6 | PASS |
| C=64 `41³→160³` | 3.82e-6 | PASS |

## 前向性能

910B2，persistent grid + 256x128 D-pass tile；预热 3 次、计时 10 次。
加速比 = 同设备原生 `F.interpolate` 时延 / Cube 时延。

| shape | output | native | Cube | 加速 |
| --- | --- | ---: | ---: | ---: |
| C=32 `33³→128³` | 268 MiB | 1.764 ms | 0.824 ms | 2.14x |
| C=32 `41³→160³` | 524 MiB | 3.325 ms | 1.411 ms | 2.36x |
| C=256 `33³→128³` | 2.1 GiB | 13.338 ms | 3.915 ms | 3.41x |
| C=256 `41³→160³` | 4.2 GiB | 26.905 ms | 10.944 ms | 2.46x |

2D `bilinear` 仅验证原生路径，不纳入 Cube 性能表；其 0.95x 是算子入口的包装开销，不代表 Cube 加速。

## 反向与训练 step 性能

| shape | bwd Cube | bwd native | bwd 加速 | fwd+bwd Cube | fwd+bwd native | 总加速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C=32 `33³→128³` | 2.247 ms | 3.539 ms | 1.58x | 2.771 ms | 5.302 ms | 1.91x |
| C=32 `41³→160³` | 2.914 ms | 5.783 ms | 1.98x | 4.333 ms | 8.970 ms | 2.07x |
| C=64 `33³→128³` | 4.328 ms | 7.068 ms | 1.63x | 5.310 ms | 10.389 ms | 1.96x |
| C=64 `41³→160³` | 5.629 ms | 11.672 ms | 2.07x | 8.385 ms | 18.212 ms | 2.17x |

C=256 反向因额外数 GiB 梯度/workspace 未跑；C=256 前向已在上表覆盖。

## 结论与限制

- 5D fp32 前向和反向精度均在 `1e-5` 容差内。
- 结果只在列出的 shape、dtype、CANN/triton 版本上验证；不同机器负载会有轻微波动。
- 2D 是原生路径，不算 Cube 路径的精度或性能收益。
- 环境体检：native-only 在 910B2 上连续 100 次与 CPU 一致，`xdiff=0`。
