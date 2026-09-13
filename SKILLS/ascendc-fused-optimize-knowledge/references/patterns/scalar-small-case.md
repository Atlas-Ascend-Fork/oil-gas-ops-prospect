# scalar-small-case

来源：cannbot `single-core-pipeline/scalar.md`。

## 现象

`[2,4,64,64]` fp32 约 0.15ms 量级。自定义 aclnn + 关闭 MatchArgs 的 host 开销和原生 `softmax` 同量级，倍数容易 <1x。

## 做法

- 核数 = `min(aivNum, totalLine, ceil(totalLine / rowsPerIter))`，小 payload 不要满 48 核。
- 保持少拍、大 tile。
- pybind 同 shape 复用 executor，跳过二次 `GetWorkspaceSize`；每次仍 Bind + launch。不要开 MatchArgs。
- 不要为小 case 上 Cube SoftMax、拆 kernel 或回退 `torch.softmax`。
- 目标：相对同设备 `softmax` **≥1.0x**。
