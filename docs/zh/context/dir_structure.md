# 目录结构

简体中文 | [English](../../en/context/dir_structure.md)

```text
oil-gas-ops-prospect/
├── LICENSE                                     # 许可证（Apache License 2.0）
├── README.md                                   # 项目入口文档
├── CONTRIBUTING.md                             # 贡献指南
├── SECURITY.md                                 # 安全声明（含公网地址声明表）
├── CHANGELOG.md                                # 变更记录
├── OAT.xml                                     # 开源合规扫描配置
├── Third_Party_Open_Source_Software_List.yaml  # 三方件清单
├── Third_Party_Open_Source_Software_Notice     # 三方件版权与许可证声明
├── .gitignore                                  # Git 忽略规则
├── .clang-format                               # C++ 代码格式
├── .pre-commit-config.yaml                     # 提交前检查钩子
├── build.sh                                    # 顶层构建入口（默认只编包；--install 安装；-u 跑测试）
├── CMakeLists.txt                              # 顶层 CMake（ophost tiling UT）
├── version.cmake                               # 版本与依赖定义
├── requirements.txt                            # Python 依赖声明
├── pyproject.toml                              # Python 打包元数据
├── setup.py                                    # Python 打包与 pybind 扩展构建
│
├── .gitcode/
│   └── workflows/                              # GitCode Action 门禁流水线
│       ├── PR-pipeline_oil-gas-ops-prospect.yml #   门禁总编排
│       ├── compile_action.yml                  #   编译（x86 / aarch64）
│       ├── llt_action.yml                      #   LLT（ophost UT + Triton 精度 + AOT 编译）
│       └── staticcheck_action.yml              #   静态检查
│
├── cmake/                                      # CMake 模块
│   ├── security_flags.cmake                    #   安全编译选项
│   └── third_party/
│       └── googletest.cmake                    #   googletest 依赖解析（gitcode 镜像）
│
├── ascendc/                                    # ★ AscendC 算子（交付物：CANN OPP .run 包）
│   ├── README.md                               #   AscendC 侧详细说明
│   ├── build.sh                                #   AscendC 侧入口（--pkg 打包 / -u --ophost 跑 UT）
│   ├── build.py                                #   打包实现（msopgen + merge + CustomOp）
│   ├── operators/
│   │   ├── manifest.tsv                        #   算子注册表（tab 分隔）
│   │   ├── _common/                            #   多算子共用的 host/kernel 头
│   │   ├── complex_mul/                        #   ComplexMul（仅 op_host / op_kernel）
│   │   │   ├── op_host/                        #     OpDef / InferShape / Tiling
│   │   │   └── op_kernel/                      #     kernel 实现
│   │   ├── complex_mul_grad/                   #   ComplexMulGrad
│   │   ├── fused_bias_softmax/                 #   FusedBiasSoftmax
│   │   └── fused_softmax_grad/                 #   FusedSoftmaxGrad
│   ├── pybind/                                 #   打包为 oil_gas_ops_prospect（按算子名暴露模块）
│   │   ├── __init__.py
│   │   ├── complex_mul.py                      #     ComplexMul Python API
│   │   ├── fused_bias_softmax.py               #     FusedBiasSoftmax / FusedSoftmaxGrad
│   │   └── csrc/                               #     <op_family>/bindings.cpp → <op_family>_npu
│   ├── pybind_extensions.py                    #   C++ 扩展统一定义
│   ├── pybind_setup.py                         #   就地编译入口
│   ├── scripts/                                #   构建流程内部模块（被 build.py 导入）
│   └── _work/                                  #   msopgen 临时工程（构建产生，不纳入版本管理）
│
├── triton-ascend/                              # ★ Triton-Ascend 算子（按算子名打包为 oil_gas_ops_prospect.<op>）
│   ├── real_fft/                               #   real_rfft / real_irfft
│   │   ├── triton_fft_real.py                  #     Triton kernel 与 autograd 封装
│   │   └── README.md
│   ├── complex_reciprocal/                     #   complex_reciprocal（z 或 xr, xi 两种形式）
│   │   ├── triton_complex_reciprocal.py        #     Triton kernel 与 autograd 封装
│   │   └── README.md                           #     算子说明
│   └── tools/
│       └── aot_compile_check.py                #   AOT 编译检查
│
├── composite-torch/                            # ★ Composite-Torch 算子（纯 PyTorch 组合，打包为 oil_gas_ops_prospect.<op>）
│   ├── README.md                               #   轨道说明与接入约定
│   └── conv3d_custom/                          #   Conv3d / conv3d（支持参数域内兼容 nn.Conv3d / F.conv3d）
│       ├── __init__.py                         #     对外导出 Conv3d
│       ├── conv3d.py                           #     Conv3d 模块（nn.Module 形态，NPU 回退）
│       ├── functional.py                       #     conv3d 函数式入口 + autograd 实现
│       └── README.md                           #     算子说明
│
├── tests/                                       # ★ 全部测试（operators/ 不再放 tests）
│   ├── README.md                               #   测试分类、套件目录与新算子必备用例
│   ├── suites.json                             #   build.sh -u 套件清单
│   ├── opensource_gates.py                     #   开源报告轻量门禁（文档 / 出包入口 / 安装痕迹 / 接口冒烟）
│   ├── _common/aclnn/                          #   ST 共用脚手架
│   ├── ut/op_host/<op>/                        #   tiling googletest
│   ├── st/aclnn<OpType>/                       #   多 case 两段式 aclnn
│   ├── conftest.py                             #   pytest 收集配置
│   ├── test_oil_gas_ops_prospect_whl.py        #   基于已安装 whl 的冒烟用例
│   ├── run_whl_tests.sh                        #   构建 whl → 安装 → 跑冒烟
│   ├── ascendc/
│   │   ├── ophost/                             #   汇总编译 ophost_ut
│   │   ├── complex_mul/                        #   NPU 精度/性能对比测试
│   │   └── fused_bias_softmax/                 #   融合注意力 NPU 精度/性能对比
│   ├── composite/conv3d_custom/                #   Conv3d / conv3d CPU 精度与性能基准
│   └── triton/
│       ├── real_fft/                           #   rFFT / iRFFT 精度（含 CPU 回退路径）
│       └── complex_reciprocal/                 #   复数倒数精度（CPU 回退 + NPU verify）
│
├── examples/                                   # Python 示例 + C++ 两段式样例
│   ├── _common/                                #   aclnn 样例脚手架
│   ├── aclnn/<op>/                             #   test_aclnn_*.cpp
│   └── *.py
├── SKILLS/                                     # 可复用的算子开发技能（优化 / 定位 / 修复），按后端拆开
├── scripts/                                    # 辅助脚本
│   ├── build_options.sh                        #   根目录 build.sh 的选项解析与分节帮助
│   ├── cann_env.sh                             #   CANN 安装目录解析与 aclnn 工程编译环境（由 aclnn 运行脚本 source）
│   ├── run_tests.py                            #   统一编排 UT/ST/精度/性能并写 Markdown 报告
│   ├── opensource_report.py                    #   按 -u JSON 生成开源测试报告
│   ├── check_license_header.sh                 #   许可证头检查
│   └── local_pkg.py                            #   没装 whl、直接跑仓内源码时解析 oil_gas_ops_prospect 导入名
├── docs/                                       # 文档
│   ├── README.md                               #   文档导航（中文）
│   ├── README_en.md                            #   文档导航（英文）
│   ├── QUICKSTART.md                           #   快速入门（中文）
│   ├── QUICKSTART_en.md                        #   快速入门（英文）
│   ├── zh/                                     #   中文文档
│   │   ├── op_list.md                          #     算子列表
│   │   ├── api_list.md                         #     接口列表
│   │   ├── context/                            #     背景与环境
│   │   ├── develop/                            #     开发指南
│   │   │   ├── operator_development_guide.md   #       新增算子步骤、约定与验收
│   │   │   └── precision_acceptance_template.md
│   │   ├── debug/                              #     调试与性能
│   │   └── invocation/                         #     调用示例
│   └── en/                                     #   英文文档（逐文件镜像 zh/）
└── output/                                     # 统一产物目录（构建产生，不纳入版本管理）
    ├── <SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run
    └── wheels/oil_gas_ops_prospect-*.whl
```

## 三类算子后端的分工

本仓按**交付物形态**分为三个业务目录，标记为 ★：

| 目录 | 后端 | 交付物 | 何时用 |
|------|------|--------|--------|
| `ascendc/` | AscendC | CANN OPP `.run` 算子包，暴露 `aclnn` 接口 | 需要精细控制切分与流水、追求极致性能的算子 |
| `triton-ascend/` | Triton-Ascend | Python whl 内的 Triton kernel | 用 Python 快速表达、依赖 autograd、形态多变的算子 |
| `composite-torch/` | Composite-Torch（纯 PyTorch） | Python whl 内的组合模块（`torch.autograd.Function`） | 组合现有 torch 算子即可表达、无需自定义 kernel 的算子 |

三者最终都通过统一的 Python 包 `oil_gas_ops_prospect` 按**算子名**对外暴露（源码树里**没有**这个目录，由 `setup.py` 的 `package_dir` 在打包时组装，不区分后端）：

- `oil_gas_ops_prospect.complex_mul` → `ascendc/pybind/complex_mul.py`
- `oil_gas_ops_prospect.fused_bias_softmax` → `ascendc/pybind/fused_bias_softmax.py`
- `oil_gas_ops_prospect.real_fft` → `triton-ascend/real_fft/`
- `oil_gas_ops_prospect.complex_reciprocal` → `triton-ascend/complex_reciprocal/`
- `oil_gas_ops_prospect.conv3d_custom` → `composite-torch/conv3d_custom/`

直接用仓内源码跑、没装 whl 时，`scripts/local_pkg.py`（examples / pytest `conftest.py` 会自动激活）把上述目录挂到同一导入名下，因此装包与不装包时调用方式一致；`composite-torch/` 下的算子由该脚本自动发现，无需手工登记。验证已安装 whl 时置 `OIL_GAS_OPS_WHL_TEST=1`（`tests/run_whl_tests.sh` 已处理），避免测到仓内源码。

## 产物与临时目录

以下目录由构建产生，已在 `.gitignore` 中排除，不纳入版本管理：

| 目录 | 内容 |
|------|------|
| `output/` | OPP `.run` 算子包与 Python whl |
| `ascendc/_work/` | msopgen 生成的临时算子工程、ophost UT 构建目录 |
| `build/`、`*.egg-info/` | Python 扩展编译中间件 |
| `test_result/` | 测试报告导出件 |

`OIL_GAS_OPS_OUTPUT_DIR` 可覆盖 `output/` 的位置，对 `build.sh`、`ascendc/build.sh --pkg`、`tests/run_whl_tests.sh` 均生效。
