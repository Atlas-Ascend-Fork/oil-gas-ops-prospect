# 测试分类与入口

`ascendc/operators/<op>/` 只放实现（`op_host` / `op_kernel`）。
测试在仓库根 `tests/`，上板小样例在 `examples/aclnn/`。
本仓没有 ATK / `OP_API_UT`。

统一入口：`bash build.sh -u ...`，跑完写出
`test_result/test_report_<时间戳>.md`、`test_result/test_report.md`，以及
`test_result/test_results.json`（开源报告的数据源）。

开源测试报告**不**随裸 `-u` 自动生成，需要额外参数。全量
`bash build.sh -u --opensource-report` 跑本仓全部测试套件（含文档、出包、安装、接口冒烟，以及 FFT 大形态与性能微基准），再按本次
JSON **生成** `docs/zh/oil-gas-ops-prospect仓开源测试报告.md`（并复制到
`test_result/`）。报告写本仓做了哪些验证、跑了多少用例；有 FAIL 时结论为不建议发布，按本次结果写用例数与通过率。

```bash
# 全量跑测并生成开源报告
bash build.sh -u --opensource-report

# 已有 test_result/test_results.json 时，只重新生成报告、不再跑测
bash build.sh --opensource-report
```

默认写出 `test_result/oil-gas-ops-prospect仓开源测试报告.md`，并同步覆盖 `docs/zh` 同名文件。

---

## 1. 分类（测什么）

按**目的**分成下面几类，不要按「有没有卡」来理解名字。

| 类别 | 测什么 | 路径 | 要 NPU？ |
| --- | --- | --- | --- |
| **UT-host** | 切分算法：UB 预算、对齐、退化输入、核间覆盖 | `tests/ut/op_host/<op>/test_*_tiling.cpp` | 否 |
| **UT-device** | 上板把 kernel 真正 launch 起来（内联小 tensor） | `examples/aclnn/<op>/` | **是** |
| **ST** | 多 shape 两段式 aclnn + numpy 真值 | `tests/st/aclnn<OpType>/` | **是** |
| **精度** | Python 对拍 PyTorch / autograd，看相对误差 | `tests/ascendc/<op>/test_*_npu.py --mode precision` | **是** |
| **性能** | Python 计时、加速比 | 同上 `--mode perf` | **是** |
| **Triton CPU** | rFFT CPU 回退精度（pytest） | `tests/triton/real_fft/test_triton_fft_real.py` | 否 |
| **Composite CPU** | Conv3d / conv3d 前反向精度与取值组合（pytest） | `tests/composite/conv3d_custom/` | 否 |
| **Triton AOT** | kernel 分支能编译 | `triton-ascend/tools/aot_compile_check.py` | 否 |
| **Triton 大形态 / 微基准** | `128^3` 与业务 5D 精度，以及开源报告 5.3 的 FFT 计时 | `tests/triton/real_fft/verify_triton_fft_real_128_3d.py`、`bench_opensource_perf.py` | **是** |
| **开源门禁** | 文档清单与章节关键词、`build.sh -h`、OPP vendor、公开接口 CPU 冒烟 | `tests/opensource_gates.py` | 安装痕迹要 CANN |

说明：

- **UT** = UT-host + UT-device。切分不依赖 NPU；「UT 要上板」指的是 UT-device 必须在 NPU 上把算子跑起来。
- **ST** 比 UT-device 更重：按 `cases.json` 循环多个 shape，并和 golden 比。
- **精度 / 性能** 走 Python API（pybind），不是 C++ aclnn 那条路径。脚本按算子登记在
  `tests/suites.json` 的 `npu_python`；composite 算子经同一 `--mode precision/perf` 协议接入
  （如 `tests/composite/conv3d_custom/bench_conv3d_custom_perf.py`）。
- Python 示例 `examples/*.py`、whl 冒烟见文末。Triton 大形态已纳入全量 `-u`。

---

## 2. 和 `build.sh` 的对应

`-u` 只表示「进入测试模式」。不跟子集时跑**全部**；跟上表某一类的开关就只跑那一类。子集可组合。

| 命令 | 跑哪些类 | 无 NPU 时 |
| --- | --- | --- |
| `bash build.sh -u` | 上表全部 | 上板类 **SKIP**，不失败 |
| `bash build.sh -u --all` | 同上 | 同上 |
| `bash build.sh -u --ut` | **UT**（UT-host + UT-device） | UT-device **FAIL** |
| `bash build.sh -u --ophost` | 只 UT-host | 可跑 |
| `bash build.sh -u --st` | **ST** | **FAIL** |
| `bash build.sh -u --precision` | **精度** | **FAIL** |
| `bash build.sh -u --perf` | **性能** | **FAIL** |
| `bash build.sh -u --examples` | 只 UT-device | **FAIL** |
| `bash build.sh -u --triton` | Triton CPU | 可跑 |
| `bash build.sh -u --composite` | Composite CPU | 可跑 |
| `bash build.sh -u --aot` | Triton AOT | 可跑 |

`--ut` 也可以单独写（等同进入测试模式并只跑 UT）：

```bash
bash build.sh --ut
```

其它子集仍建议写成 `-u --st` 这种形式；`--st` / `--precision` / `--perf` 等本身也会进入测试模式，因此 `bash build.sh --st` 同样只跑 ST。

```bash
# 四类一起（UT + ST + 精度 + 性能）
bash build.sh -u --ut --st --precision --perf

# 只上板 UT
bash build.sh -u --ut

# 只切分
bash build.sh -u --ophost

# 只 ST / 只精度 / 只性能
bash build.sh -u --st
bash build.sh -u --precision
bash build.sh -u --perf --ops=complex_mul

# 过滤算子（对 UT-device / ST / 精度 / 性能生效）
bash build.sh -u --st --precision --ops=complex_mul,fused_bias_softmax

# 报告路径
bash build.sh -u --report=test_result/my.md

# 开源测试报告（按本次 JSON 生成）
bash build.sh -u --opensource-report
```

辅助开关：`--gtest_filter=pattern`（只作用于 UT-host）、`--noexec`（只编译 UT-host）、`--cov`、`--report=<path>`、`--opensource-report[=path]`。

CI 门禁 `JOB_llt` 走 cmake/ctest + pytest + AOT，**不**调用 `build.sh -u`。

---

## 3. 目录

```
tests/
├── suites.json                          # ST / 精度 / UT-device 清单
├── ut/op_host/<op>/                     # UT-host
├── st/aclnn<OpType>/                    # ST
├── _common/aclnn/                       # ST 脚手架
├── ascendc/ophost/                      # 汇总编译 ophost_ut
├── ascendc/<op>/test_*_npu.py           # 精度 + 性能
├── triton/real_fft/                     # Triton CPU（及可选 NPU verify）
└── composite/conv3d_custom/             # Composite CPU（Conv3d / conv3d）

examples/aclnn/<op>/                     # UT-device（C++ 两段式小 launch）
```

新增算子：实现放 `operators/`；UT-host 放 `tests/ut/op_host/<op>/`；UT-device 放 `examples/aclnn/<op>/`；ST 放 `tests/st/aclnn<OpType>/`；精度脚本放 `tests/ascendc/<op>/`；后三项登记 `tests/suites.json`。Composite 算子的 NPU 精度/性能同样经 `tests/suites.json` 的 `npu_python` 登记（脚本需支持 `--mode precision/perf` 协议，参考 `bench_conv3d_custom_perf.py`）。

---

## 4. 不在 `-u` 里的

| 项 | 入口 | 说明 |
| --- | --- | --- |
| whl 冒烟 | `bash build.sh --install` 或 `bash tests/run_whl_tests.sh` | `--install` 路径的第 5 步 |
| Python 调用示例 | `python3 examples/complex_mul_example.py` 等 | 无 NPU 时走参考实现 |

精度验收模板见 [`docs/zh/develop/precision_acceptance_template.md`](../docs/zh/develop/precision_acceptance_template.md)。
