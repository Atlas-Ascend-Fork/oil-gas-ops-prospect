# 测试用例

本仓库**所有**测试用例集中在这个目录下，算子目录（`ascendc/operators/`、
`triton-ascend/real_fft/`）只放实现，不放测试。

## 目录结构

```
tests/
├── conftest.py                          # pytest 收集配置
├── run_whl_tests.sh                     # 构建 whl → 安装 → 跑冒烟用例
├── test_oil_gas_ops_prospect_whl.py     # whl 打包与 Python 接口冒烟
├── ascendc/
│   ├── ophost/                          # tiling 算法单元测试（googletest）
│   │   ├── CMakeLists.txt
│   │   └── test_complex_mul_tiling.cpp
│   ├── complex_mul/                     # ComplexMul 精度 / 性能对比
│   │   ├── test_complex_mul_npu.py
│   │   ├── run_test.sh
│   │   └── README.md
│   └── fused_bias_softmax/              # 融合注意力精度 / 性能对比
│       ├── test_fused_bias_softmax_npu.py
│       ├── run_test.sh
│       └── README.md
└── triton/
    └── real_fft/                        # rFFT / iRFFT 精度
        ├── test_triton_fft_real.py
        └── verify_triton_fft_real_128_3d.py
```

## 用例分类

| 用例 | 是否需要 NPU | 入口 |
| --- | --- | --- |
| `ascendc/ophost/` | 否 | `bash build.sh -u --ophost`（或 `ctest --test-dir build`） |
| `triton/real_fft/test_triton_fft_real.py` | 否（无卡自动回退 CPU） | `python3 -m pytest tests/triton/real_fft -q` |
| `test_oil_gas_ops_prospect_whl.py` | 否 | `bash tests/run_whl_tests.sh` |
| `ascendc/complex_mul/` | **是** | `bash tests/ascendc/complex_mul/run_test.sh` |
| `ascendc/fused_bias_softmax/` | **是** | `bash tests/ascendc/fused_bias_softmax/run_test.sh` |
| `triton/real_fft/verify_triton_fft_real_128_3d.py` | **是** | 见 [`triton-ascend/real_fft/README.md`](../triton-ascend/real_fft/README.md) |

## 无卡环境一键跑

```bash
bash build.sh -u                 # 全部无卡 UT（--ut 是别名）
bash build.sh -u --ophost        # 只跑 ophost tiling
bash build.sh -u --triton --aot  # 只跑 Triton pytest + AOT
bash build.sh -u --noexec        # 只编译 ophost UT，不执行
bash build.sh -u -h              # Test Options
```

不加子集开关时依次执行 ophost tiling UT、Triton CPU 回退精度、Triton AOT 编译检查，
这也是 CI 门禁 `JOB_llt` 的内容。需要 NPU 的用例会自动 skip。

## 需要 NPU 的用例

```bash
# AscendC：安装 OPP 包 → 编译 pybind 扩展 → 跑精度/性能对比
bash tests/ascendc/complex_mul/run_test.sh
bash tests/ascendc/fused_bias_softmax/run_test.sh

# Triton：三方对比（torch.fft CPU / torch-npu / 融合 Triton kernel）
python3 tests/triton/real_fft/verify_triton_fft_real_128_3d.py --help
```

## 新增算子时

- AscendC：把算子的 `op_host` 目录加进 `tests/ascendc/ophost/CMakeLists.txt` 的
  `OPHOST_UT_INCLUDE_DIRS`，测试源加进 `ophost_ut` 目标；精度对比测试新建
  `tests/ascendc/${op_name}/`。
- Triton：在 `tests/triton/${op_family}/` 下新建 `test_triton_${op_family}.py`，
  用例需能在无卡环境下走 CPU 回退。

精度验收标准见 [`docs/zh/develop/precision_acceptance_template.md`](../docs/zh/develop/precision_acceptance_template.md)。
