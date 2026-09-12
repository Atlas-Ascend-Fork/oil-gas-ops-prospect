# oil-gas-ops-prospect

## 🚀 概述

oil-gas-ops-prospect 是面向**油气勘探**（oil & gas exploration）领域的昇腾自定义算子库。仓名即 **oil-gas-ops**（油气算子）+ **prospect**（勘探目标/圈闭），面向地震成像、全波形反演等勘探计算负载，不绑定任何单一上层模型工程。

本仓按**交付物形态**分为两类算子后端；Python 对外接口按算子名暴露，不区分实现后端：

| 后端 | 目录 | 交付物 | 适用场景 |
|------|------|--------|----------|
| AscendC | [`ascendc/`](ascendc/) | CANN OPP `.run` 算子包，暴露 `aclnn` 接口 | 需要精细控制核间切分与流水、追求极致性能的算子 |
| Triton-Ascend | [`triton-ascend/`](triton-ascend/) | Python whl 内的 Triton kernel | 用 Python 快速表达、依赖 autograd、形态多变的算子 |

## 📌 版本配套

当前仓库已验证通过的 CANN 版本如下：

| CANN 版本 | 发布时间 | 分支 |
|-----------|----------|------|
| [CANN 9.0.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0) | 2026/04/30 | master |

本仓库需要安装 **Toolkit 包**和对应 SoC 的 **Ops 包**，当前面向 **Ascend 910B**（`Ascend910B1`）。完整的安装步骤、SoC 支持矩阵与环境变量配置见[环境部署](docs/zh/context/quick_install.md)。

## ⬇️ 源码下载

```bash
git clone https://gitcode.com/cann/oil-gas-ops-prospect.git
cd oil-gas-ops-prospect
```

## ⚡️ 快速入门

### 编译与安装

在仓库根目录一键完成 OPP 编译 → 安装 → whl 构建 → whl 安装 → 冒烟测试：

```bash
bash build.sh
```

产物统一落到仓库根目录 `output/`：

```text
output/
├── <SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run   # OPP 算子包
└── wheels/oil_gas_ops_prospect-*.whl                # Python whl
```

安装完成后，后续 shell 中需要保留自定义 opapi 动态库路径：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
```

常用参数：

```bash
bash build.sh --help                                # 查看全部选项
bash build.sh -u -h                                 # Test Options
bash build.sh --clean                               # 先清理再构建
bash build.sh --ops=complex_mul,complex_mul_grad    # 只构建指定算子
bash build.sh --skip-tests                          # 跳过 whl 冒烟测试
bash build.sh --soc=Ascend910B1 -j16                # 指定 SOC 与并行度
```

完整的构建入口与参数说明见[编译参数说明](docs/zh/context/build.md)。

### 接口使用

```python
import torch
import torch_npu  # noqa: F401
from oil_gas_ops_prospect.real_fft import real_irfft, real_rfft

x = torch.randn(8, 256, device="npu")
xr, xi = real_rfft(x, dim=-1)          # 实数正向 FFT，频率轴长度 256 // 2 + 1 = 129
y = real_irfft(xr, xi, n=256, dim=-1)  # 从半谱重建实信号
```

可直接运行的示例见 [`examples/`](examples/)，更多调用方式见[算子调用](docs/zh/invocation/quick_op_invocation.md)。

### 测试

```bash
# 全量无卡单元测试：ophost tiling UT + Triton CPU 回退精度 + Triton AOT 编译检查
bash build.sh -u
# 只跑 ophost / 只编译不执行：
# bash build.sh -u --ophost
# bash build.sh -u --ophost --noexec

# 只验证已安装的 whl
bash tests/run_whl_tests.sh
```

所有测试用例集中在 [`tests/`](tests/) 下，算子目录只放实现；用例分类与 NPU 依赖见 [`tests/README.md`](tests/README.md)。

> **编译通过不等于功能正确。** 算子在 NPU 上的精度、性能、显存必须在真实 910B 上验证。无卡验证手段与问题定位见[调试与性能分析](docs/zh/debug/op_debug_prof.md)。

## 📖 接口支持清单

当前交付 4 个 AscendC 算子和 2 个 Triton-Ascend 算子：

| 名称 / OpType | 后端 / 交付物 | 功能 | dtype / format |
|---------------|---------------|------|----------------|
| [`ComplexMul`](ascendc/operators/complex_mul/) | AscendC / OPP `.run`（AIV-only） | 复数逐点乘：`(ur, ui), (vr, vi) -> (wr, wi)` | `float32` / `ND`，全部张量同 shape |
| [`ComplexMulGrad`](ascendc/operators/complex_mul_grad/) | AscendC / OPP `.run`（AIV-only） | `ComplexMul` 的反向 | `float32` / `ND`，全部张量同 shape |
| [`FusedBiasSoftmax`](ascendc/operators/fused_bias_softmax/) | AscendC / OPP `.run` | `softmax(x + bias1 + bias2, -1)`，原地写回 | `fp32` / `fp16` / `bf16`，`ND` |
| [`FusedSoftmaxGrad`](ascendc/operators/fused_softmax_grad/) | AscendC / OPP `.run`（可走 MIX） | 注意力 softmax JVP：`g = go @ V^T` 后原地覆写 probs | `fp32` / `fp16` / `bf16`，`ND` |
| [`real_rfft`](triton-ascend/real_fft/) | Triton-Ascend / Python whl | 实数正向 FFT：`x -> (xr, xi)`，频率轴长度 `n // 2 + 1` | NPU kernel 为 `float32`，非 NPU 环境回退到 PyTorch |
| [`real_irfft`](triton-ascend/real_fft/) | Triton-Ascend / Python whl | 从实部、虚部半谱重建实数：`(xr, xi), n -> y` | NPU kernel 为 `float32`，非 NPU 环境回退到 PyTorch |

数学定义、约束与支持范围见[算子列表](docs/zh/op_list.md)；完整接口签名、参数与环境变量见[接口列表](docs/zh/api_list.md)。

## 🔍 目录结构

```text
oil-gas-ops-prospect/
├── ascendc/                # AscendC 算子（交付物：CANN OPP .run 包）
│   ├── operators/          #   算子源码 + manifest.tsv 注册表
│   ├── pybind/             #   Python API（打包为 oil_gas_ops_prospect）+ C++ 绑定
│   ├── scripts/            #   msopgen 构建流程内部模块
│   └── build.sh            #   AscendC 侧入口（--pkg 打包 / -u --ophost 跑 UT）
├── triton-ascend/          # Triton-Ascend 算子（real_fft 打包为 oil_gas_ops_prospect.real_fft）
│   ├── real_fft/           #   real_rfft / real_irfft
│   └── tools/              #   无卡 AOT 编译检查
├── tests/                   # 全部测试用例（算子目录内不放用例）
│   ├── ascendc/ophost/     #   tiling UT（无卡，googletest）
│   ├── ascendc/complex_mul/#   NPU 精度/性能对比
│   ├── ascendc/fused_bias_softmax/
│   └── triton/real_fft/    #   rFFT / iRFFT 精度（无卡回退 CPU）
├── examples/               # 可运行示例
├── scripts/                # 辅助脚本
├── docs/                   # 文档
├── cmake/                  # CMake 模块（三方件解析、安全编译选项）
├── .gitcode/workflows/     # GitCode Action 门禁流水线
├── build.sh                # 顶层构建入口（-u 跑无卡 UT）
├── CMakeLists.txt          # 顶层 CMake（无卡 C++ 目标：ophost UT）
└── version.cmake           # 版本与依赖定义
```

逐目录职责说明见[目录结构](docs/zh/context/dir_structure.md)。

## 🛠️ 开发指南

新增 AscendC 或 Triton-Ascend 算子的完整流程、目录约定、命名约定与自检方式见[算子开发指南](docs/zh/develop/operator_development_guide.md)与[算子开发规格](docs/zh/develop/operator-dev-spec.md)。

提交前自检：

```bash
bash build.sh -u                        # 全量无卡单元测试
bash scripts/check_license_header.sh    # 许可证头检查
```

## 💬 相关信息

- [快速入门](docs/QUICKSTART.md)
- [文档导航](docs/README.md)
- [目录结构](docs/zh/context/dir_structure.md)
- [算子列表](docs/zh/op_list.md)
- [接口列表](docs/zh/api_list.md)
- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [变更记录](CHANGELOG.md)
- [许可证](LICENSE)（Apache License 2.0）
- [所属 SIG：oil-gas-engineering](https://gitcode.com/cann/community/tree/master/CANN/sigs/oil-gas-engineering)

---

PS：本项目功能和文档正在持续更新和完善中，欢迎您关注最新版本。

- **问题反馈**：通过 GitCode [Issues](https://gitcode.com/cann/oil-gas-ops-prospect/issues) 提交问题。
- **社区互动**：通过 GitCode [讨论](https://gitcode.com/cann/oil-gas-ops-prospect/discussions) 参与交流。
