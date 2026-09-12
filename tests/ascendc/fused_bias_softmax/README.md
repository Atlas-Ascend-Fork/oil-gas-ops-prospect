# fused_bias_softmax / fused_softmax_grad NPU 对比测试

openfold 注意力前反向融合算子（一对，backward 依赖 forward 语义）：

- **FusedBiasSoftmax**：`y = softmax(x + bias1(广播/全尺寸) + bias2, -1)`，原地写回
- **FusedSoftmaxGrad**：`g = go @ V^T; probs ← probs⊙(g − rowsum(probs⊙g))`，fp32 域，原地覆写

## 目录

```
tests/ascendc/fused_bias_softmax/
├── test_fused_bias_softmax_npu.py  # 精度（forward / backward / e2e）与性能微基准
├── run_test.sh                     # 安装 OPP 包 → 编译扩展 → 跑测试
└── README.md
```

被测的 pybind 封装是算子交付件，不在本目录：

- Python API：[`ascendc/pybind/fused_bias_softmax.py`](../../../ascendc/pybind/fused_bias_softmax.py)
- C++ 绑定：[`ascendc/pybind/csrc/fused_bias_softmax/`](../../../ascendc/pybind/csrc/fused_bias_softmax/)

## 路径分发说明（backward）

| tiling key | 路径 | 条件 |
|---|---|---|
| 0/1/2 | 纯向量（AIV） | fp32 / fp16 / bf16 且 S 或 D 非 16 对齐 |
| 3 | MIX_AIC_1_2（AIC 矩乘 → workspace → AIV JVP） | bf16 且 S%16==0 且 D%16==0 |

fp32/fp16 走向量是精度设计（2201 fixpipe C 16bit 结构上限）。

## 前置条件

1. 已构建出 OPP 包，产物在仓库根目录 `output/<SOC>/oil_gas_ops_prospect_opp_*.run`。
2. **测试需在带 NPU 的机器上运行**，且环境内有 `torch` 与 `torch_npu`。

## 运行

```bash
cd tests/ascendc/fused_bias_softmax
bash run_test.sh

# 指定 OPP 包
OIL_GAS_OPS_RUN_PKG=/path/to/oil_gas_ops_prospect_opp_*.run bash run_test.sh
```

容差约定：fp32 `max_diff ≤ 1e-5`；bf16 契约式 `atol=2e-2, rtol=4e-3`
（bf16 输出量化半 ulp 随幅值增长，纯 abs 会误报）。
