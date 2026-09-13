# 评测

精度门槛不改。性能必须报相对同设备 Torch 小算子的加速比。

## 无卡

```bash
bash ascendc/build.sh -u --ophost
```

覆盖 `tests/ascendc/ophost/test_fused_bias_softmax_tiling.cpp` 与 `test_fused_softmax_grad_tiling.cpp`。切分公式一变，先改这些手算期望。

## NPU 官方精度

```bash
bash tests/ascendc/fused_bias_softmax/run_test.sh
```

前向 / 反向 / e2e 全 PASS。bf16 契约 `atol=2e-2, rtol=4e-3`；fp32 `max_diff ≤ 1e-5`。

## NPU 性能（相对 Torch）

```bash
# 容器内需 customize vendor 的 libcust_opapi
python3 /root/work/tmp/oil-gas-qa/bench_ops.py
```

看第 2 节：`FusedBiasSoftmax` vs `softmax`，`FusedSoftmaxGrad` vs `matmul+JVP`。

最少对比这些行：

| 用例 | 当前 baseline |
|---|---|
| fwd `[2,4,64,128]` bf16 | 2.89x |
| fwd `[2,4,256,256]` bf16 | 2.67x |
| fwd `[2,4,64,64]` bf16 | 2.83x |
| fwd `[8,8,256,256]` bf16 | 2.68x |
| fwd `[2,4,64,64]` fp32 | 1.66x |
| bwd `[2,4,64,64]` D64 bf16 | 2.04x |
| bwd `[8,8,256,256]` D64 bf16 | 1.98x |

## 可选 msprof

cannbot `ops-profiling` 的 `msprof` 只在倍数对不上或要判 bound 时用。日常一轮用 `bench_ops.py` 即可。
