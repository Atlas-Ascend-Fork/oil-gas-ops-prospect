# CustomGRUV2 算子质量保障体系

## 一、设计依据

本测试方案基于以下 Ascend 算子质量标准设计：

| 标准/工具 | 作用 | 对应测试层 |
|---|---|---|
| **msOpST** (Operator System Test) | 算子功能验证，生成单算子 OM 并验证执行结果 | 单元测试 |
| **msKPP** (Kernel Performance Profiling) | 算子性能建模与预测 | 性能测试 |
| **msProf** (Profiling) | NPU 侧性能数据采集分析 | 性能测试 |
| **msSanitizer** | 内存越界、溢出检测 | 边界测试 |
| **Ascend C API 规范** | Vector API (Exp/Reciprocal/Muls/Add) 使用合规性 | 代码审查 |
| **PyTorch autograd.grad** | 反向梯度精度验证 | 精度测试 |

## 二、质量目标

| 维度 | 指标 | 阈值 |
|---|---|---|
| **功能正确性** | 前向输出 vs PyTorch GRUCell | max_diff ≤ 0.01 |
| **精度-前向** | 前向 max_abs_diff (H=24,T=1000) | ≤ 0.003 |
| **精度-反向** | 梯度 vs autograd (double) | gradcheck pass, atol=1e-4, rtol=1e-3 |
| **精度-tanh稳定性** | 大输入 (|x|>10) tanh 输出 | 无 NaN/Inf |
| **性能-前向** | 单序列推理延迟 (B=1,T=1000,H=24) | 可测量, NPU: ratio < 1.0 (vs Python loop) |
| **性能-吞吐** | batch 推理 (B=32,T=100,H=24) | 可测量 (steps/s) |
| **内存安全** | LocalTensor buffer 无溢出 | mssanitizer memcheck + racecheck (待 NPU 执行) |
| **代码覆盖率** | Python 单元测试行覆盖 | **100%** (双环境合并) / 80% (仅CPU) |
| **稳定性** | 连续 10 次执行结果一致 (NPU) | torch.equal 通过 |

## 三、测试架构

```
testplan/
├── README.md                    # 本文件
├── test_plan.md                 # 详细测试计划
├── test_evidence.md             # 测试执行证据 (git 回显 + collect-only + run_all_tests 输出)
├── pytest.ini                   # pytest 配置 (python_files = test_*.py bench_*.py)
├── conftest.py                  # pytest 共享 fixtures (npu_only, maybe_npu_sync, torch_gru_ref)
├── requirements.txt             # Python 依赖 (torch-npu 可选, 仅 NPU 用例需要)
├── unit/                        # 单元测试
│   ├── test_forward_precision.py    # 前向精度验证
│   ├── test_backward_precision.py   # 反向梯度验证
│   ├── test_shape_coverage.py       # Shape 覆盖测试
│   ├── test_edge_cases.py           # 边界条件测试
│   ├── test_numerical_stability.py  # 数值稳定性测试
│   └── test_coverage_paths.py       # 代码路径覆盖测试
├── integration/                 # 集成测试
│   ├── test_e2e_training.py         # 端到端训练验证
│   └── test_e2e_inference.py        # 端到端推理验证
├── performance/                 # 性能测试
│   ├── bench_forward.py             # 前向性能基准
│   └── bench_backward.py            # 反向性能基准
├── coverage/                    # 覆盖率
│   └── run_coverage.py              # 覆盖率收集脚本
├── msopst/                      # NPU 工具链测试
│   ├── custom_gruv2_test.json       # msOpST 用例定义 (5 cases)
│   ├── run_sanitizer.sh             # mssanitizer 内存安全检测脚本
│   ├── run_prof.sh                  # msprof 性能采集脚本
│   └── _prof_run.py                 # msprof 辅助运行脚本
├── pack_submission.sh            # 送审打包脚本
└── run_all_tests.sh             # 一键执行全部测试
```

## 四、执行方式

```bash
# 前置：已构建并安装 OPP .run 包 + whl 包
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH

# 执行全部测试
cd tests/ascendc/custom_gru_v2
bash run_all_tests.sh

# 仅执行单元测试
python -m pytest unit/ -v --tb=short

# 仅执行性能测试
python -m pytest performance/ -v -s

# 生成覆盖率报告 (要求 100%)
python coverage/run_coverage.py

# NPU 工具链: mssanitizer 内存安全检测
bash msopst/run_sanitizer.sh

# NPU 工具链: msprof 性能采集
bash msopst/run_prof.sh
```
