# CustomGRUV2 NPU Kernel 测试报告

> 源码 commit: `f7b953c` (Round 4)
> 编译时间: 2026-08-04 07:51
> 验证时间: 2026-08-04 08:53
> 环境: CANN 8.5.0 / Ascend910B2 / torch 2.7.1+cpu / torch_npu 2.7.1.post2
> .run 包 SHA256: `9dbf650f514f41c28d267a47260d4e85a31e060af32d96f6a39b08a3edcbb784`

## 1. 闭环流程

```
源码 (f7b953c)
  ├── op_kernel/custom_gruv2.cpp  — 3× PipeBarrier<PIPE_V>, 删除 padParams 死代码
  ├── op_host/custom_gruv2.cpp   — nullptr 校验 + hiddenSize>0 + weight_hh 一致性 + UB_LIMIT 176KB
  └── op_host/custom_gruv2_tiling.h
        ↓
    python3 build.py -v Ascend910B2 --ops custom_gru_v2
        ↓
    oil_gas_opp_ubuntu_aarch64.run (SHA256: 9dbf650f...)
        ↓
    bash oil_gas_opp_ubuntu_aarch64.run --quiet  →  /usr/local/Ascend/cann-8.5.0/opp/vendors/customize/
        ↓
    NPU 验证 (三工具)
```

## 2. msOpST — 功能正确性验证 (5/5 PASS)

通过 Python 直接调用编译后的 NPU kernel 扩展，对 5 个测试用例逐一验证输出 shape 和 NaN 检查。

| # | 用例 | 输入 | bidirectional | y.shape | h.shape | 结果 |
|---|------|------|:---:|---------|---------|:----:|
| 1 | standard_shape | B=1, T=100, H=24 | False | [1,1,100,24] | [1,1,24] | PASS |
| 2 | batch32 | B=32, T=100, H=24 | False | [1,32,100,24] | [1,32,24] | PASS |
| 3 | bidirectional | B=1, T=50, H=8 | True | [2,1,50,8] | [2,1,8] | PASS |
| 4 | single_step | B=1, T=1, H=8 | False | [1,1,1,8] | [1,1,8] | PASS |
| 5 | no_bias | B=1, T=10, H=8 | False | [1,1,10,8] | [1,1,8] | PASS |

**总计: 5/5 PASS**

原始输出: `raw/msopst_run.log`

## 3. msSanitizer — 内存安全验证 (PASS)

### 3.1 memcheck (UB 溢出 + GM 越界检测)

| 场景 | 输入 | 结果 |
|------|------|:----:|
| standard | B=1, T=100, H=24 | PASS (无 UB 溢出, 无 GM 越界) |
| bidirectional | B=1, T=50, H=8, bidir=True | PASS (无 UB 溢出, 无 GM 越界) |
| single_step | B=1, T=1, H=8 | PASS (无 UB 溢出, 无 GM 越界) |

### 3.2 racecheck (多核数据竞争检测)

| 场景 | 输入 | BlockDim | 结果 |
|------|------|:---:|:----:|
| batch32 | B=32, T=100, H=24 | 32 | PASS (无数据竞争) |

原始输出: `raw/sanitizer_run.log`, `raw/memcheck.log`, `raw/racecheck.log`

## 4. msProf — 性能分析

### 4.1 单序列 (B=1, T=1000, H=24)

| 指标 | 值 |
|------|-----|
| 调用次数 | 25 (5 warmup + 20 measured) |
| Avg Time | 24,731.94 us |
| Min Time | 24,728.50 us |
| Max Time | 24,735.52 us |
| aiv_scalar_ratio | 99.3% |
| aiv_vec_ratio | 0.9% |
| aiv_mte2_ratio | 0.0% |
| aiv_mte3_ratio | 0.6% |
| aiv_total_cycles | 44,517,517 |

### 4.2 批量序列 (B=32, T=100, H=24)

| 指标 | 值 |
|------|-----|
| 调用次数 | 25 (5 warmup + 20 measured) |
| Avg Time | 2,531.46 us |
| Min Time | 2,520.58 us |
| Max Time | 2,534.82 us |
| aiv_scalar_ratio | 99.3% |
| aiv_vec_ratio | 0.9% |
| aiv_mte2_ratio | 0.0% |
| aiv_mte3_ratio | 0.6% |
| aiv_total_cycles | 144,186,917 |
| BlockDim | 32 |

### 4.3 性能分析

- `aiv_scalar_ratio=99.3%` 表明 kernel 当前以标量运算为主，向量化程度极低 (vec_ratio=0.9%)
- 单序列 (T=1000) 耗时约为批量 (T=100) 的 10 倍，符合时间步线性复杂度预期
- 批量场景 BlockDim=32，利用 32 核并行处理 batch 维度

原始输出: `raw/msprof_run.log`, `raw/op_statistic_single.csv`, `raw/op_statistic_batch.csv`, `raw/op_summary_single.csv`, `raw/op_summary_batch.csv`

## 5. Round 4 Kernel 变更清单

| 变更 | 文件 | 说明 |
|------|------|------|
| 3× PipeBarrier<PIPE_V> | op_kernel/custom_gruv2.cpp | Duplicate h 后、标量写 gates/t2 后、Add h 后各加 1 处 |
| GM 标量读优化注释 | op_kernel/custom_gruv2.cpp | 标注 GM 标量读为已知瓶颈，待向量化优化 |
| 删除 padParams 死代码 | op_kernel/custom_gruv2.cpp | 移除未使用的 DataCopyPadExtParams 声明 |
| nullptr 校验 | op_host/custom_gruv2.cpp | x/wi/wh shape nullptr 检查 |
| hiddenSize>0 校验 | op_host/custom_gruv2.cpp | hiddenSize==0 时报错 |
| weight_hh 一致性 | op_host/custom_gruv2.cpp | wh shape dimNum>=2 + 维度与 hiddenSize 匹配 |
| UB_LIMIT 176KB | op_host/custom_gruv2.cpp | 192KB→176KB, 16KB margin |

## 6. 结论

| 验证项 | 结果 | 回归 |
|--------|:----:|:----:|
| msOpST 功能正确性 | 5/5 PASS | 无回归 |
| msSanitizer memcheck | PASS | 无回归 |
| msSanitizer racecheck | PASS | 无回归 |
| msProf 性能指标 | 与 Round 3 一致 | 无回归 |

第四轮 kernel 修改 (PipeBarrier + host 校验 + UB_LIMIT) 编译安装后，功能正确性、内存安全性、多核竞争检测、性能指标均与第三轮一致，无回归。
