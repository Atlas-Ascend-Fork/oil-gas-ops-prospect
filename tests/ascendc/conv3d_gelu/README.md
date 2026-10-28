# Conv3DGelu 测试（conv3d + GELU 融合）

对 `Conv3DGelu`（cube 做 conv3d，vector 做 tanh 近似 GELU）做 NPU 对比。
用例形状沿用原 `oil_gas_ops` 的 conv3d_gelu 测试，并加上 FDPOT 里常见的 1×1 和 128×256×256 整网一步。

真值是 **CPU float32** 的 `F.gelu(F.conv3d(...), approximate="tanh")`，并且把
`torch_npu.npu.conv.allow_hf32` 关掉。NPU 上 `nn.GELU` / `F.gelu` 实际执行的就是这条 tanh 近似，
和 CPU `F.gelu(approximate="tanh")` 一致。CPU `nn.GELU()` 默认是 erf，不是本算子的对照基准。

- **精度**：相对 CPU float32 的相对误差 `< 1e-5`；输入梯度最大绝对误差 `< 1e-4`。
- **性能**：1×1 直写加速比须大于 1.05。3×3 在输出不超过 256MB 时一次算完，和拆开的 conv3d 同一量级；更大的体积沿 D 按大约 48MB 一块切。
- **显存**：前向返回后不再持有 conv 与 GELU 之间的预激活，持有量至少少掉该激活的 40%。`(2,32,160³)` 的前向峰值还要比拆开路径至少再低该激活的 40%，因为整幅私有格式缓冲不再分配。

## 算子说明

`y = gelu_tanh(conv3d(x, weight, bias))`，float32，groups=1。

1. **1×1、stride=1、padding=0，且输入通道不少于 8、输出通道不超过 64**：cube 结果在用户输出上原地做 GELU，不再做 NDC1HWC0 往返。输入通道为 1，或输出通道大于 64 的 1×1，仍走私有格式，否则会比拆开的 conv3d 更慢。
2. **其余卷积**（含 3×3）：输出不超过 256MB 时一次 launch。更大则沿 D 按大约 48MB 一块切（`OIL_GAS_OPS_CONV3D_GELU_DEPTH_CHUNK` 可改平面数，`<=0` 为整幅）。块内 cube 写入私有格式，`SyncAll` 后 AIV 原地 GELU，再转回 NCDHW。
3. **反向**：不保存预激活。小体积一次重算 conv3d、做 GELU 反向再做 conv3d 反向。大体积按和前向相同的深度块累加梯度。

GELU 公式与 `F.gelu(approximate="tanh")` 一致：

```text
gelu(x) = x / (1 + exp(-1.5957691216057308 * (x + 0.044715 * x^3)))
```

Python API：[`ascendc/pybind/conv3d_gelu.py`](../../../ascendc/pybind/conv3d_gelu.py)。
绑定：[`ascendc/pybind/csrc/conv3d_gelu/bindings.cpp`](../../../ascendc/pybind/csrc/conv3d_gelu/bindings.cpp)。
aclnn：`aclnnConv3dGeluCustom`（NCDHW 入口，内部 TransData）。

## 前置条件

1. OPP 包已构建，默认路径 `output/Ascend910B1/oil_gas_ops_prospect_opp_*.run`：

   ```bash
   bash ascendc/build.sh --pkg -v Ascend910B1 --ops conv3d_gelu
   ```

2. 测试需要真实 NPU，以及 `torch`、`torch_npu`。

## 运行

```bash
cd tests/ascendc/conv3d_gelu
bash run_test.sh
```

`run_test.sh`：安装 OPP（若产物存在）→ 必要时编译 `conv3d_gelu_npu` → 跑单算子测试和整网一步。

ophost 路径选择单测：`tests/ut/op_host/conv3d_gelu/test_conv3d_gelu_tiling.cpp`（`bash build.sh -u --ophost`）。
aclnn ST：`tests/st/aclnnConv3dGeluCustom/`（`bash build.sh -u --st --ops=conv3d_gelu`）。
aclnn 样例：`examples/aclnn/conv3d_gelu/`。

## 覆盖 shape

| 名称 | batch | in_c | D×H×W | out_c | kernel | pad | bias | 来源 |
|------|------|------|-------|-------|--------|-----|------|------|
| k3_16 | 1 | 16 | 16³ | 32 | 3 | 1 | 有 | 原仓 |
| k3_32 | 2 | 32 | 32³ | 32 | 3 | 1 | 有 | 原仓 |
| k3_24 | 1 | 8 | 24³ | 16 | 3 | 0 | 无 | 原仓 |
| k1_latent | 1 | 32 | 16×32×32 | 64 | 1 | 0 | 有 | FDPOT |
| k1_wide | 1 | 128 | 8×32×32 | 64 | 1 | 0 | 有 | FDPOT |
| k3_fullres_chunk | 1 | 32 | 80×128×128 | 32 | 3 | 1 | 有 | FDPOT 全分辨率块 |
| k3_96x192 | 1 | 32 | 96×192×192 | 32 | 3 | 1 | 有 | 切块 |
| k3_128x160 | 1 | 32 | 128×160×160 | 32 | 3 | 1 | 有 | 切块 |
| k3_64x256 | 1 | 16 | 64×256×256 | 32 | 3 | 1 | 有 | 切块 |
| k3_n2_112 | 2 | 16 | 112×160×160 | 32 | 3 | 1 | 有 | 切块 |
| k3_160 | 2 | 32 | 160³ | 32 | 3 | 1 | 有 | 原仓必测 |
| k_stride | 1 | 4 | 6×8×8 | 4 | 3 | `(1,1,0)`，stride `(2,1,1)` | 有 | 不等 stride / pad |
| k_dilate | 1 | 4 | 8×6×6 | 4 | `(3,1,1)` | `(2,0,0)`，dilation `(2,1,1)` | 无 | dilation |
| k_asym | 1 | 4 | 5×7×6 | 4 | `(1,3,1)` | `(0,1,0)`，stride `(1,1,2)` | 有 | 非对称核 |
| k_replicate | 1 | 4 | 4×6×6 | 4 | 3 | 1，`padding_mode=replicate` | 无 | Python 先 pad |

整网：`test_fdpot_train_128.py`，空间 `(128, 256, 256)`、pml=16，融合网络里 kernel 为 1 或 3 且 groups=1 的 Conv3d+GELU。要求 encoder 相对误差 `< 1e-5`，并且前向持有显存、前向峰值、整步峰值都低于未融合路径。

退出码跟 `--mode` 走。`--mode precision` 只看精度：达标 `0`，失败 `1`。`--mode perf` 只打印 `[perf]` 计时，退出码为 `0`。默认 `--mode all` 同时看精度、1×1 加速比和显存：都达标为 `0`，否则 `2`。NPU 不可用为 `1`。`ACLNN_CACHE_LIMIT=0` 要在启动 Python 之前设置。

按[精度验收报告模板](../../../docs/zh/develop/precision_acceptance_template.md)填写的 2026-10-28 结果：

- [精度与性能报告](report/precision_acceptance_report.md)
- [Precision and performance report](report/precision_acceptance_report_en.md)
