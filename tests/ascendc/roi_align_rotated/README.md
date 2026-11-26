# ROIAlignRotated / ROIAlignRotatedGrad NPU 对比测试

`ROIAlignRotated` 与反向的精度、autograd 与性能对比。基准是仓内纯 torch 参考实现
`torch_roi_align_rotated`（逐行对齐 detectron2 CUDA 语义）。

## 算子语义

```text
input [N,C,H,W] + rois [K,6] -> out [K,C,PH,PW]
rois 每行 = (batch_idx, x_ctr, y_ctr, width, height, angle_deg)
```

双线性插值边界、`sampling_ratio=0` 自适应网格、反向 `w*grad/count` 散射累加均严格对齐
`detectron2/layers/csrc/ROIAlignRotated/ROIAlignRotated_cuda.cu`。

## 目录

```text
tests/ascendc/roi_align_rotated/
├── test_roi_align_rotated_npu.py   # 精度 / autograd / 性能
├── run_test.sh                     # 安装 OPP 包 -> 编译扩展 -> 跑测试
└── README.md
```

被测 pybind 封装是算子交付件，见
[`ascendc/pybind/roi_align_rotated.py`](../../../ascendc/pybind/roi_align_rotated.py) 与
[`ascendc/pybind/csrc/roi_align_rotated/`](../../../ascendc/pybind/csrc/roi_align_rotated/)。

## 前置条件

1. 已构建 OPP 包：

   ```bash
   bash build.sh --pkg --soc=Ascend910B3 --ops=roi_align_rotated,roi_align_rotated_grad
   ```

2. 真实 NPU（本机 910B3）与 `torch` / `torch_npu`。

## 运行

```bash
cd tests/ascendc/roi_align_rotated
bash run_test.sh
bash run_test.sh --mode precision
bash run_test.sh --mode perf

# 指定 OPP 包
OIL_GAS_OPS_RUN_PKG=/path/to/oil_gas_ops_prospect_opp_*.run bash run_test.sh
```

## 输出

```text
[precision] (1, 3, 10, 10) k=3 pool=5x5 sr=2 scale=1.0 fwd(max=...,rel=...) bwd(max=...,rel=...)
[autograd] grad max=... rel=...
[module] out shape=(3, 3, 5, 5)
[fp16] dtype=torch.float16 shape=(2, 2, 4, 4)
[empty] shape=(0, 2, 4, 4)
[perf] ...
PASS
```

退出码：`0` 精度达标；`1` 超阈值 / 无 NPU / 扩展缺失。

## 独立 golden（detectron2 CPU C++）

`verify_detectron2_cpu_golden.py` 会临时编译 detectron2 的
`ROIAlignRotated_cpu.cpp`，作为独立于仓内 torch 参考实现的 golden；同时比较
torch 参考实现与（有 NPU 时）自定义 AscendC 算子的前向/反向。

```bash
DETECTRON2_ROOT=/path/to/detectron2 \
  python3 tests/ascendc/roi_align_rotated/verify_detectron2_cpu_golden.py
```

未设置 `DETECTRON2_ROOT` 时脚本 SKIP，不影响默认 `-u` 测试。

## 注意

- `ACLNN_CACHE_LIMIT=0` 必须在启动 Python 前设置（`run_test.sh` 已处理）。
- pybind 绑定用 `getCurrentNPUStream().stream(true)` 冲刷 torch_npu 延迟队列，避免
  `torch.ones_like` 等前序算子尚未下发时 kernel 读到未初始化内存。
- 性能段的 torch 参考在 CPU 上跑是临时口径；M5 会补同 NPU 的参考计时与 msprof kernel-only 数据。
