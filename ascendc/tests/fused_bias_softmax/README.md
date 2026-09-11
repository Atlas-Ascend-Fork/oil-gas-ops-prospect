# FusedBiasSoftmaxOg / FusedSoftmaxGradOg 测试

openfold 注意力前反向融合算子（一对，backward 依赖 forward 语义）：

- **FusedBiasSoftmaxOg**：`y = softmax(x + bias1(广播/全尺寸) + bias2, -1)`，原地写回
- **FusedSoftmaxGradOg**：`g = go @ V^T; probs ← probs⊙(g − rowsum(probs⊙g))`，fp32 域，原地覆写

## 目录内容

| 文件 | 说明 |
|---|---|
| `cpp/bindings.cpp` | torch_npu pybind 绑定（fused_bias_softmax_npu），直连 autogen 的 aclnn 两段式入口 |
| `setup.py` | pybind 扩展编译配置（链接 cust_opapi） |
| `test_fused_bias_softmax_npu.py` | 精度（forward 4 case + backward 5 case + e2e 链 4 项）与性能微基准 |
| `run_test.sh` | 一键：安装 OPP 包 → 就地编译扩展 → 跑测试 |

## 路径分发说明（backward）

| tiling key | 路径 | 条件 |
|---|---|---|
| 0/1/2 | 纯向量（AIV） | fp32 / fp16 / bf16 且 S 或 D 非 16 对齐 |
| 3 | MIX_AIC_1_2（AIC 矩乘 → workspace → AIV JVP） | bf16 且 S%16==0 且 D%16==0 |

fp32/fp16 走向量是精度设计（2201 fixpipe C 16bit 结构上限），见 `docs/` 测试报告。

## 运行

```bash
cd ascendc && bash build.sh --pkg -v Ascend910B2   # 先构建 OPP 包
cd tests/fused_bias_softmax && bash run_test.sh        # 安装 + 编译扩展 + 测试（需 NPU）
```

容差约定：fp32 `max_diff ≤ 1e-5`；bf16 契约式 `atol=2e-2, rtol=4e-3`
（bf16 输出量化半 ulp 随幅值增长，纯 abs 会误报）。
