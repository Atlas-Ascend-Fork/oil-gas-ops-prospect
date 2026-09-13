# double-buffer-softmax

来源：cannbot `elementwise/double_buffer_design.md` + perf-optimize Step 4 memory bound。

## 现象

`TQue` depth=1 时 CopyIn → SoftMax → CopyOut 串行。`[8,8,256,256]` 每核约 5+ 拍，MTE2 等 Vector。

## 做法

1. UB 预算把 **入队张量**（x/y/全尺寸 bias）按 depth=2 计价；SoftMax 32K 与 fp32 中转仍 1 份。
2. `InitBuffer(..., 2, tileBytes)`；`TQue<..., 2>`。
3. Process 预取：先 `CopyIn(0)`，循环里 `CopyIn(i+1)` → `Compute(i)` → `CopyOut(i)`。
4. 1 拍形态（典型 `[2,4,64,128]`）只多占 UB、不增加拍数，不应变慢。

## 不要

- 为双缓冲把 1 拍形态切成很多小拍。
- 把 SoftMax 共享 32K 也翻倍（高阶 API 单份即可）。
- 不改 ophost 手算期望就改 `ComputeMaxByteLine`。
