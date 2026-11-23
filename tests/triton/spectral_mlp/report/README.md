# spectral_mlp 验证报告

本目录保存 `spectral_mlp` 算子的测试环境、精度与性能记录。正式算子文档
（`docs/{zh,en}/op_list.md`、`api_list.md`、`triton-ascend/spectral_mlp/README.md`）
只描述功能、接口和约束，不重复这里的数值。

## 文件

| 文件 | 内容 |
| --- | --- |
| `README.md` | 测试环境、覆盖范围、复现命令与已知限制 |
| [`precision_perf.md`](precision_perf.md) | 精度与性能结果表、结论 |
| [`results.json`](results.json) | `verify_spectral_mlp.py --json` 的原始输出，含环境、容差、逐条精度/性能数据与最终 `status` |

## 测试环境

| 项目 | 内容 |
| --- | --- |
| 主机 | `bmswt010` |
| 设备 | Ascend 910B2，使用 NPU 0 |
| CANN | 9.0.0 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post4 |
| Triton | `triton-ascend` 3.2.0 |
| 测试日期 | 2026-11-20 |
| 测试入口 | `tests/triton/spectral_mlp/verify_spectral_mlp.py`、`tests/triton/spectral_mlp/test_spectral_mlp.py` |

**该卡与其他容器共享**：采集时 8 张卡的 AICore 占用率在 90%–100% 之间。因此
性能绝对值会随邻居负载波动；本目录的性能数字来自 `--repeats 5` 的
重复测量（每轮 20 次），取中位数。

性能基线固定为「同设备 fp32 的两层 `einsum` + 逐点门控 + 偏置 + 模型原实现的
`torch_npu.npu_gelu`」，即上游模型里被替换掉的代码；不使用手写逐元素 GELU
近似作为基线。

## 覆盖范围

精度（真值为 CPU `float64` 上的 `einsum` 参考实现）：

- 前向：`(2,3,(2,3),G=4,d=8,dm=8,rank=8)`、`(1,4,(4,4),2,16,16,8)`、
  `(1,3,(2,3),2,12,20,6)`（非 2 的幂维度，覆盖内核 padding）与业务形态
  `(1,64,(128,128),8,16,16,8)`。
- 反向：同一组 shape 下全部 14 个输入/参数的梯度，与参考 autograd 对比。
- 边界：`xr/xi` 形状不匹配、`up1` 与 `down1` 的 rank 不一致、门控形状错误
  均抛 `ValueError`；非连续输入；`float64` 回退路径。

判定口径：

- 前向按生态算子开源精度标准的 fp32 混合容差（`atol = rtol = 2^-13`），
  要求逐元素通过率 ≥ 0.99 且 `max_abs <= 1e-2`。
- 反向按相对 L2 误差 ≤ `1e-3` 且 `max_abs <= 1e-2`。跨 `S` 全量规约的
  权重/门控/偏置梯度在 Cube fp32 下有固有相对误差，因此不套用与元素个数
  无关的逐元素阈值；`results.json` 中仍保留 `matched_ratio` 供参考。
- 显存：给出可复算的中间量公式，并用 `bench_memory.py` 按「一条路径一个进程」
  测训练单步峰值（同一进程里连续测会被 allocator 复用污染），见
  [`precision_perf.md`](precision_perf.md) 第 2 节。

未覆盖：

- `float16` / `bfloat16`（算子只支持 `float32`，其它 dtype 回退 PyTorch）。
- `d / dm / rank > 256` 的超大分组维度（UB 预算会大幅折减 `block_m`，
  尚未在真实业务 shape 上标定）。

## 复现

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

# pytest：CPU 回退逐位一致 + NPU 前/反向精度
python3 -m pytest tests/triton/spectral_mlp/test_spectral_mlp.py -q

# 精度 + 性能，原始结果写入 results.json
python3 tests/triton/spectral_mlp/verify_spectral_mlp.py --mode all --img 128 \
  --iters 20 --repeats 5 --json tests/triton/spectral_mlp/report/results.json

# 只跑精度（npu_python 套件协议，tests/suites.json）
python3 tests/triton/spectral_mlp/verify_spectral_mlp.py --mode precision

# 训练单步峰值显存：两条路径各起一个进程
python3 tests/triton/spectral_mlp/bench_memory.py --mode ref
python3 tests/triton/spectral_mlp/bench_memory.py --mode op

# AOT 编译门禁：前向与反向
python3 triton-ascend/tools/aot_compile_check.py --filter spectral_mlp
```

## 已知限制

- **反向大规约的精度**：`bias1`、`up2`、`down2`、`gate2` 这类梯度是跨
  `B*L*S*G` 的全量规约。实现用「每核 8 槽位部分和 + host 归约」把顺序累加
  长度压到 1/8；实测相对 L2 在 2.5e-07 ~ 1.6e-04 之间波动，波动来自
  triton-ascend 为 fp32 `tl.dot` 选择的 Cube 路径（编译/缓存状态不同），
  不是每次都能落在高精度路径上。需要稳定更高精度时应改用 fp64 参考或多级
  归约，当前后端不支持 fp64。
- **前向误差**：各用例 `max_abs` 在 1.7e-08~4.2e-08，相对 L2 约 4e-08，
  属于 Cube fp32 累加顺序差异。
- **GELU 近似差**：算子用 sigmoid 近似，与精确 erf-GELU 相比 `N(0,1)` 输入下
  `max_abs ≈ 4.7e-04`；上游原实现用精确 `npu_gelu`，端到端会带上这一项差异。
- **反向是重算而非读取**：前向不保存 `x1/h1/o1/x2`，反向用 6 个小 GEMM 重算，
  因此前向只有一次输入读 + 一次输出写，训练显存只与输入/输出有关；代价是反向
  多几个小 GEMM。默认 `block_m=512` / `block_m_bwd=256` 是按 128³ 实测选的。
- **共享卡**：见上文的卡片占用说明，性能绝对值不宜作为独占卡结论。
