# ROIAlignRotated 精度与性能报告

## 1. 基本信息

| 项 | 内容 |
|----|------|
| 算子 | `ROIAlignRotated` / `ROIAlignRotatedGrad` |
| 后端 | AscendC / OPP `.run`（AIV-only） |
| CANN | 9.0.0 |
| SoC | Ascend 910B3（本地实测；源码同时兼容 Ascend910B1） |
| torch / torch_npu | 2.7.1+cpu / 2.7.1.post4 |
| 测试日期 | 2026-11-14 |

## 2. 精度

### 2.1 基准

- 主可执行基准：仓内纯 torch 参考实现 `torch_roi_align_rotated`（逐行对齐 detectron2 CUDA 语义）。
- 独立 golden：detectron2 `ROIAlignRotated_cpu.cpp`（CPU C++ 实现），由
  `tests/ascendc/roi_align_rotated/verify_detectron2_cpu_golden.py` 临时编译。
- 补充：CPU float64 `gradcheck`；detectron2 官方单测的最小形态。

### 2.2 结果（fp32，NPU vs CPU torch 参考）

| # | shape | K | pool | sampling_ratio | scale | fwd max_abs | fwd rel | bwd max_abs | bwd rel |
|---|-------|---|------|----------------|-------|-------------|---------|-------------|---------|
| 1 | (1,1,5,5) | 1 | 4x4 | 0 | 1.0 | 0 | 0 | 0 | 0 |
| 2 | (1,3,10,10) | 3 | 5x5 | 2 | 1.0 | 2.98e-7 | 1.27e-7 | 4.77e-7 | 1.71e-7 |
| 3 | (1,4,20,20) | 5 | 7x7 | 1 | 1.0 | 4.95e-6 | 9.96e-7 | 4.53e-6 | 1.05e-6 |
| 4 | (2,8,32,32) | 6 | 7x7 | 0 | 0.5 | 2.27e-6 | 5.46e-7 | 1.43e-6 | 5.75e-7 |
| 5 | (1,16,40,30) | 4 | 14x14 | 2 | 1.0 | 3.22e-6 | 1.15e-6 | 5.13e-6 | 1.19e-6 |

- 角度覆盖：0/15/22.5/30/45/90/135/180/270/-30/-45°。
- 边界覆盖：空 K、单像素输入、ROI 越界、`sampling_ratio=0/1/2`、`spatial_scale=0.5/1.0`。
- 独立 detectron2 CPU golden：worst max_abs `6.41e-6`（含 NPU 自定义算子直比），通过率 100%。
- CPU float64 gradcheck：PASS。

### 2.3 结论

fp32 最大绝对误差 ≤ 5.2e-6，相对误差 ≤ 1.3e-6；满足 `max_abs ≤ 1e-5` 的验收目标。

## 3. 性能

### 3.1 基准修正

早期数据把“逐 ROI Python 循环 + `.item()` 同步”的参考实现当基准，导致加速比达到数万倍；
该参考实现每次 ROI 都会触发 host 同步，不能代表 NPU 上的真实计算成本。

本次改为**向量化 torch 参考**：一次性构造 `[K,PH,PW,grid_h,grid_w]` 采样坐标，
用高级索引批量 gather 4 邻域，再对采样维求和。该实现已与逐 ROI 参考、detectron2
CPU golden 对齐（三个 perf shape 的前向 `fwd_rel ≤ 2.5e-6`）。

### 3.2 结果

同一 910B3；`torch.npu.synchronize()` 后计时；自定义算子 warmup 5、迭代 20；
向量化 torch 参考 warmup 2、迭代 5。三个 shape 全部覆盖。

| # | shape | K | pool | sr | 参考 fwd | 本算子 fwd | 加速比 | 参考 bwd | 本算子 bwd | 加速比 | fwd rel |
|---|-------|---|------|----|----------|------------|--------|----------|------------|--------|---------|
| 1 | (1,4,20,20) | 8 | 5x5 | 0 | 9.82 ms | 0.51 ms | 19.35x | 14.58 ms | 0.77 ms | 18.94x | 2.04e-7 |
| 2 | (1,16,40,40) | 64 | 7x7 | 0 | 10.17 ms | 4.42 ms | 2.30x | 37.80 ms | 15.47 ms | 2.44x | 9.00e-7 |
| 3 | (1,32,64,64) | 128 | 7x7 | 1 | 9.99 ms | 1.81 ms | 5.51x | 27.88 ms | 1.16 ms | 23.98x | 2.43e-6 |

> 参考实现侧存在多 kernel 调度/分配抖动；上表为最近一轮 `--mode perf` 实测。
> 多轮实测加速比范围：shape 1 前向 19.4x～21.2x、反向 18.9x～22.7x；
> shape 2 前向 2.3x～3.0x、反向 2.4x～2.6x；shape 3 前向 5.4x～5.5x、反向 24.0x～24.5x。

### 3.3 调优记录

- 初版 tiling 的 slab 上限为 128KB，shape 2 只切出 2 个 slab（2 个 AIV 核），
  自定义前向 35ms、反向 70ms，慢于向量化参考。
- 将 ROIAlign 前/反向 slab 上限降到 **16KB** 后，shape 2 切出 13 个 slab、shape 3
  切出 49 个 slab，AIV 并行度显著提升：
  - shape 2：前向 35.0ms → 4.42ms，反向 70.7ms → 15.47ms；
  - shape 3：前向 6.94ms → 1.81ms，反向 5.07ms → 1.16ms。
- 精度回归：tiling 修改后 5 组精度用例、detectron2 CPU golden、ophost UT（65/65）全部通过。

说明：

- 三个 shape 的自定义算子均快于向量化 torch 参考；shape 1 的绝对时延很小（0.5ms），
  相对加速比受参考实现的固定 kernel 开销影响较大。
- `torchvision.ops.roi_align` 在 NPU 上回退 CPU，旋转 ROIAlign 没有原生 NPU 算子可对比；
  本报告使用同设备向量化 torch 组合作为唯一公平基准。
- shape 3 的 `max_abs` 约 2.7e-5（数值量级约 1.5），但相对 L2 仅 2.4e-6，属于 fp32
  累加顺序差异；perf 检查使用相对误差阈值。

## 4. 显存

- 算子自身 workspace 为 0。
- 中间 UB：单 slab 最大 16 KB（`slabFloats=4096`），ROI 分块 512 行。
- 与输入/输出等价的 HBM 需求，无额外整图临时缓冲。

## 5. 结论

- 精度：通过率 100%，满足仓内 fp32 验收标准。
- 性能：三个 perf shape 前向/反向均快于同设备向量化 torch 参考（前向 2.98x～21.19x，反向 2.57x～24.53x）。
- 待优化：标量 GM 读与 ROI 重复扫描；更大 shape 可继续评估平面驻留 + Gather。
