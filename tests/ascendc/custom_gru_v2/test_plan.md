# CustomGRUV2 详细测试计划

## 1. 测试范围

### 1.1 被测对象

| 组件 | 文件 | 送审路径 | 说明 |
|---|---|---|---|
| Kernel (前向) | `op_kernel/custom_gruv2.cpp` | `ascendc/operators/custom_gru_v2/op_kernel/` | Ascend C 逐时间步 GRU 计算, tanh = 2*sigmoid(2x)-1 |
| Host (Tiling+Infershape) | `op_host/custom_gruv2.cpp` | `ascendc/operators/custom_gru_v2/op_host/` | Tiling 参数提取 + 输出 shape 推导 |
| Tiling 数据结构 | `op_host/custom_gruv2_tiling.h` | `ascendc/operators/custom_gru_v2/op_host/` | 6 个 uint32_t 字段 || Python 封装 | `ascendc/pybind/custom_gru_v2.py` | `ascendc/pybind/` | autograd.Function + nn.Module |
| C++ Plugin | `ascendc/pybind/csrc/custom_gru_v2/bindings.cpp` | `ascendc/pybind/csrc/custom_gru_v2/` | ACLNN 调用 + pybind11 |

> **送审打包**: 执行 `bash testplan/pack_submission.sh` 生成完整 zip，包含上述全部文件。
> 首轮/次轮审核中被测模块不在包内的问题已通过打包脚本修复。

### 1.2 不测试范围

- CANN Runtime / ACL 底层 API (华为保障)
- torch_npu 框架层 (框架保障)
- PyTorch autograd 引擎 (PyTorch 保障)

## 2. 测试分层

### 2.1 单元测试 (Unit Tests)

#### 2.1.1 前向精度验证 (`test_forward_precision.py`)

| 用例 ID | 描述 | 输入 | 预期 | 方法 |
|---|---|---|---|---|
| UT-FWD-001 | 最小 shape | B=1,T=1,H=1,F=1 | max_diff ≤ 0.01 | vs GRUCell |
| UT-FWD-002 | 标准 shape | B=1,T=100,H=24,F=3 | max_diff ≤ 0.01 | vs GRUCell |
| UT-FWD-003 | 大序列 | B=1,T=1000,H=24,F=3 | max_diff ≤ 0.003 | vs GRUCell |
| UT-FWD-004 | batch=32 | B=32,T=100,H=24,F=3 | max_diff ≤ 0.01 | vs GRUCell |
| UT-FWD-005 | 隐藏层对齐 H=8 | B=1,T=50,H=8,F=3 | max_diff ≤ 0.01 | 8 对齐 8 |
| UT-FWD-006 | 隐藏层非对齐 H=7 | B=1,T=50,H=7,F=3 | max_diff ≤ 0.01 | 7 对齐 8 |
| UT-FWD-007 | 双向 GRU | B=1,T=50,H=8,F=3,bidir=True | max_diff ≤ 0.01 | vs nn.GRU bidir |
| UT-FWD-008 | 零输入 | B=1,T=10,H=8,F=3,x=0 | y 有界 (无 NaN) | sigmoid(0)=0.5 |
| UT-FWD-009 | 随机权重固定 seed | seed=42, B=4,T=100,H=16,F=8 | max_diff ≤ 0.01 | 可复现 |
| UT-FWD-010 | CPU fallback | CPU tensor, require=0 | 走 PureGRU 路径 | 不崩溃 |
| UT-FWD-011 | bias=False | B=1,T=10,H=8,F=3,bias=False | max_diff ≤ 1e-6 | vs GRUCell |
| UT-FWD-012 | 双向 batch | B=4,T=50,H=8,F=3,bidir=True | max_diff ≤ 0.01 | vs GRUCell |
| UT-FWD-013 | NPU 前向 (@npu_only) | B=1,T=100,H=24,F=3 | max_diff ≤ 0.01 | vs GRUCell (NPU) |

#### 2.1.2 反向梯度验证 (`test_backward_precision.py`)

| 用例 ID | 描述 | 输入 | 预期 | 方法 |
|---|---|---|---|---|
| UT-BWD-001 | 梯度精确匹配 | B=1,T=10,H=8,F=3 | gradcheck pass (double) | autograd.gradcheck |
| UT-BWD-002 | batch 梯度 | B=4,T=10,H=8,F=3 | gradcheck pass (double) | autograd.gradcheck |
| UT-BWD-003 | 大序列梯度 | B=1,T=10,H=8,F=3 | atol=1e-4, rtol=1e-3 | autograd.grad |
| UT-BWD-004 | 权重梯度 | B=1,T=10,H=8,F=3 | wi/wh grad 非零 | requires_grad |
| UT-BWD-005 | bias 梯度 | B=1,T=10,H=8,F=3 | bi/bh grad 非零 | requires_grad |
| UT-BWD-006 | CPU fallback backward | CPU tensor | grad 正确 | torch.autograd |
| UT-BWD-007 | NPU 梯度容差 (@npu_only) | B=1,T=10,H=8,F=3 | atol=1e-3, rtol=1e-2 | autograd.grad (NPU) |
| UT-BWD-008 | 手写反向公式 gradcheck (修正 dh_prev) | B=2,T=1,H=4,F=3 | gradcheck pass (double) | autograd.gradcheck |
| UT-BWD-009 | 错误公式 gradcheck 必须失败 | B=2,T=1,H=4,F=3 | gradcheck FAIL | autograd.gradcheck |
| UT-BWD-010 | 多步手写反向 gradcheck | B=2,T=3,H=4,F=3 | gradcheck pass (double) | autograd.gradcheck |

#### 2.1.3 Shape 覆盖测试 (`test_shape_coverage.py`)

| 用例 ID | B | T | H | F | 对齐 | 说明 |
|---|---|---|---|---|---|---|
| UT-SHAPE-001 | 1 | 1 | 1 | 1 | 1→8 | 最小 |
| UT-SHAPE-002 | 1 | 10 | 8 | 3 | 8→8 | 完美对齐 |
| UT-SHAPE-003 | 1 | 10 | 7 | 3 | 7→8 | 非对齐 padding |
| UT-SHAPE-004 | 1 | 10 | 16 | 3 | 16→16 | 2 倍对齐 |
| UT-SHAPE-005 | 1 | 10 | 24 | 3 | 24→24 | 业务 shape |
| UT-SHAPE-006 | 1 | 10 | 25 | 3 | 25→32 | 跨对齐组 |
| UT-SHAPE-007 | 32 | 100 | 24 | 3 | 24→24 | 训练 shape |
| UT-SHAPE-008 | 1 | 1000 | 24 | 3 | 24→24 | 长序列 |
| UT-SHAPE-009 | 1 | 10 | 1 | 1 | 1→8 | H=1 边界 |
| UT-SHAPE-010 | 1 | 2 | 8 | 1 | 8→8 | T=2 最短 |
| UT-SHAPE-011 | 2 | 50 | 8 | 3 | 8→8 | 双向 shape 验证 (P0-新-1 回归) |

#### 2.1.4 边界条件测试 (`test_edge_cases.py`)

| 用例 ID | 描述 | 条件 | 预期 |
|---|---|---|---|
| UT-EDGE-001 | 零输入 | x = zeros | y = sigmoid(bias) 传播 |
| UT-EDGE-002 | 大输入 | x = randn * 100 | 无 NaN/Inf |
| UT-EDGE-003 | 负大输入 | x = -randn * 100 | 无 NaN/Inf |
| UT-EDGE-004 | 零权重 | wi = wh = 0 | y 仅依赖 bias |
| UT-EDGE-005 | 零 bias | bi = bh = 0 | y 仅依赖权重×输入 |
| UT-EDGE-006 | 单步序列 | T=1 | y[:,0,:] == output_h |
| UT-EDGE-007 | 稳定性-重复 (NPU) | 同输入执行 10 次 | 结果完全一致 (@npu_only) |
| UT-EDGE-008 | 稳定性-连续 | 不 recreate 模块连续推理 | 结果一致 |
| UT-EDGE-009 | require_custom CPU 抛异常 | require=1, CPU tensor | RuntimeError |
| UT-EDGE-010 | bias=False | CustomGRU(bias=False) | bias_ih is None, 无空指针 |
| UT-EDGE-011 | UB 安全网大 H 拒绝 | H=7017 (Ha=7024), 7*Ha > 49152 | TilingFunc 返回 GRAPH_FAILED |

#### 2.1.5 数值稳定性测试 (`test_numerical_stability.py`)

| 用例 ID | 描述 | 条件 | 预期 |
|---|---|---|---|
| UT-NUM-001 | tanh 大正值 | gate_input = 20 | tanh ≈ 1.0, 无 Inf |
| UT-NUM-002 | tanh 大负值 | gate_input = -20 | tanh ≈ -1.0, 无 NaN |
| UT-NUM-003 | sigmoid 饱和 | gate = 50 | sigmoid ≈ 1.0 |
| UT-NUM-004 | sigmoid 负饱和 | gate = -50 | sigmoid ≈ 0.0, 无 Inf |
| UT-NUM-005 | 混合极端值 | 50% 大值 + 50% 正常 | 无 NaN/Inf, max_diff ≤ 0.1 |

#### 2.1.6 代码路径覆盖测试 (`test_coverage_paths.py`)

| 用例 ID | 描述 | 覆盖路径 | 环境 |
|---|---|---|---|
| UT-COV-001 | bias=None ih/hh/both | `_torch_gru_forward` bias=None 分支 | CPU |
| UT-COV-002 | _torch_gru_forward 3D weights unidir | 3D weight reshape 路径 | CPU |
| UT-COV-003 | _torch_gru_forward 2D weights bidir | 2D weight bidir 分支 | CPU |
| UT-COV-004 | CustomGRU bias=True unidir | `__init__` bias=True | CPU |
| UT-COV-005 | CustomGRU bias=False bidir | `__init__` bias=False | CPU |
| UT-COV-006 | CustomGRU _init_weights | `_init_weights` 方法 | CPU |
| UT-COV-007 | CustomGRU trainable | `trainable` 属性 | CPU |
| UT-COV-008 | _require_custom True/False | require_custom 函数 | CPU |
| UT-COV-009 | _is_npu_tensor True/False | _is_npu_tensor 函数 | CPU/NPU |
| UT-COV-010 | _CustomGRUV2Function.forward | NPU 前向 autograd | NPU |
| UT-COV-011 | _CustomGRUV2Function.backward | NPU 反向 autograd | NPU |
| UT-COV-012 | bidirectional NPU | 双向前向 | NPU |
| UT-COV-013 | bias=False backward NPU | bias=None 反向 | NPU |
| UT-COV-014 | extension load failure + require | RuntimeError 路径 | CPU |
| UT-COV-015 | extension load failure + fallback | CPU fallback 路径 | CPU |
| UT-COV-016 | CustomGRU module on NPU | nn.Module NPU 路径 | NPU |

### 2.2 集成测试 (Integration Tests)

#### 2.2.1 端到端训练验证 (`test_e2e_training.py`)

| 用例 ID | 描述 | 场景 | 验证 |
|---|---|---|---|
| IT-TRAIN-001 | 单步训练 | 1 epoch, B=32, T=100, H=24 | loss 下降 |
| IT-TRAIN-002 | 10 epoch 训练 | 10 epoch, B=32, T=100, H=24 | loss 收敛 |
| IT-TRAIN-003 | 梯度反传完整 | forward + backward + step | 权重更新 |
| IT-TRAIN-004 | 梯度值匹配参考 | forward+backward, 梯度值与参考一致 | 梯度非零, atol/rtol |
| IT-TRAIN-005 | optimizer 集成 | Adam + CustomGRU | loss 下降 |

#### 2.2.2 端到端推理验证 (`test_e2e_inference.py`)

| 用例 ID | 描述 | 场景 | 验证 |
|---|---|---|---|
| IT-INFER-001 | 单序列推理 | B=1, T=1000 | 输出 shape [1,1,1000,H] |
| IT-INFER-002 | batch 推理 | B=32, T=100 | 输出 shape [1,32,100,H] |
| IT-INFER-003 | 双向推理 | bidir=True | 输出 shape [2,B,T,H] |
| IT-INFER-004 | nn.Module 集成 | CustomGRU module forward | 输出正确 |
| IT-INFER-005 | 双向模块集成 | CustomGRU(bidirectional=True) | 输出 shape [2,B,T,H] |
| IT-INFER-006 | NPU graph capture | make_graphed_callables (@npu_only) | graph replay 正确 |

### 2.3 性能测试 (Performance Tests)

#### 2.3.1 前向性能基准 (`bench_forward.py`)

| 用例 ID | 场景 | 指标 | 基准 |
|---|---|---|---|
| PERF-FWD-001 | B=1,T=1000,H=24 | 延迟 (ms) | 可测量 |
| PERF-FWD-002 | B=32,T=100,H=24 | 延迟 (ms) | 可测量 |
| PERF-FWD-003 | B=1,T=1000,H=24 | 吞吐 (samp/s) | 可计算 |
| PERF-FWD-004 | vs PureGRU 加速比 | custom / pure | NPU: ratio < 1.0 (assert), CPU: skip |

#### 2.3.2 反向性能基准 (`bench_backward.py`)

| 用例 ID | 场景 | 指标 | 基准 |
|---|---|---|---|
| PERF-BWD-001 | B=1,T=100,H=24 | 延迟 (ms) | 可测量 |
| PERF-BWD-002 | B=32,T=100,H=24 | 延迟 (ms) | 可测量 |
| PERF-BWD-003 | vs autograd 基准 | custom / pure | NPU: ratio < 1.0 (assert), CPU: skip |

## 3. 代码覆盖率策略

### 3.1 Python 覆盖率 (目标: 100% 双环境合并 / 80% 仅 CPU)

使用 `pytest-cov` + `coverage.py`:

```bash
# NPU 环境: CPU+NPU 合并 → 100%
python coverage/run_coverage.py

# CPU 环境: 仅 CPU 路径 → 80%
python coverage/run_coverage.py
```

**必须覆盖的代码路径:**

| 路径 | 覆盖用例 |
|---|---|
| `custom_gru_v2()` NPU 路径 | UT-FWD-002 (需 NPU) |
| `custom_gru_v2()` CPU fallback 路径 | UT-FWD-010 |
| `custom_gru_v2()` require=True 崩溃路径 | UT-COV-014 |
| `_CustomGRUV2Function.forward` | UT-COV-010 (需 NPU) |
| `_CustomGRUV2Function.backward` | UT-COV-011 (需 NPU) |
| `_torch_gru_forward` 所有分支 | UT-COV-001~003 |
| `CustomGRU.__init__` bias=True/False | UT-COV-004~005 |
| `CustomGRU.forward` | UT-COV-004~005, UT-COV-016 |
| `_load_extension` 成功/失败 | UT-COV-014~015 |
| `_is_npu_tensor` True/False | UT-COV-009 |
| `_require_custom` True/False | UT-COV-008 |

### 3.2 C++ Kernel 覆盖率 (目标: 100%)

通过 msSanitizer + shape 覆盖确保:

| Kernel 代码块 | 覆盖用例 |
|---|---|
| `dir==0` 正向循环 | UT-FWD-002 |
| `dir==1` 反向循环 | UT-FWD-007 |
| `H < Ha` padding 分支 | UT-SHAPE-003, UT-SHAPE-006 |
| `H == Ha` 无 padding | UT-SHAPE-002 |
| `B > 1` batch 循环 | UT-FWD-004 |
| `T == 1` 单步 | UT-EDGE-006 |
| tanh 稳定计算路径 | UT-NUM-001 ~ 005 |
| 输出写回 | DataCopyPad 批量搬运 (替代逐元素 SetValue) | 所有前向用例 |
| 多核并行 | SetBlockDim 按 batch 维切分, 每核处理 B/blockDim 个样本 | UT-FWD-004 (B=32), UT-SHAPE-007 |

## 4. 测试环境

| 项 | 值 |
|---|---|
| NPU | Ascend 910B2 |
| CANN | 8.5.0+ |
| torch | 2.7.1+ |
| torch_npu | 2.7.1+ |
| Python | 3.10+ |
| pytest | 7.0+ |
| pytest-cov | 4.0+ |

## 5. 通过/失败标准

| 标准 | 通过条件 |
|---|---|
| 所有单元测试 | 100% pass |
| 所有集成测试 | 100% pass |
| 前向精度 | max_diff ≤ 0.01 (标准), ≤ 0.003 (长序列) |
| 反向精度 | gradcheck pass (double), atol=1e-4, rtol=1e-3 |
| 数值稳定性 | 0 个 NaN, 0 个 Inf |
| 代码覆盖率 | Python 100% (双环境合并) / 80% (仅CPU) |
| 性能 | NPU: ratio < 1.0 (custom 快于 Python loop), CPU: 跳过加速比断言 |
| 稳定性 | 10 次重复 torch.equal (NPU 环境) |

## 6. 用例合计

| 测试层 | 文件 | 用例数 |
|---|---|:---:|
| 单元-前向精度 | test_forward_precision.py | 13 |
| 单元-反向梯度 | test_backward_precision.py | 7 |
| 单元-反向 gradcheck | test_manual_backward_gradcheck.py | 3 |
| 单元-Shape 覆盖 | test_shape_coverage.py | 11 |
| 单元-边界条件 | test_edge_cases.py | 11 |
| 单元-数值稳定性 | test_numerical_stability.py | 5 |
| 单元-代码路径覆盖 | test_coverage_paths.py | 23 |
| 集成-训练 | test_e2e_training.py | 5 |
| 集成-推理 | test_e2e_inference.py | 6 |
| 性能-前向 | bench_forward.py | 4 |
| 性能-反向 | bench_backward.py | 3 |
| **合计** | | **91** |
