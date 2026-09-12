# 变更记录

本文件记录 oil-gas-ops-prospect 的主要变更。版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- 合入 origin/master 的融合注意力算子：`FusedBiasSoftmax` / `FusedSoftmaxGrad`（含 ophost tiling UT、NPU 精度对比与 Python API `oil_gas_ops_prospect.fused_bias_softmax`）。
- AscendC 的 Python API 统一放在 `ascendc/pybind/<op_family>.py`，`ascendc/pybind/` 打包为 `oil_gas_ops_prospect` 包根。
- 合入算子开发规格（现位于 `docs/zh/develop/operator-dev-spec.md`）。
- 补齐开源合规交付件：`CONTRIBUTING.md`、`SECURITY.md`、`OAT.xml`、`Third_Party_Open_Source_Software_List.yaml`、`Third_Party_Open_Source_Software_Notice`。
- 新增顶层 CMake 构建体系：`CMakeLists.txt`、`version.cmake`、`cmake/third_party/googletest.cmake`、`cmake/security_flags.cmake`。顶层 `cmake -S . -B build` 可直接配置并运行 ophost 单元测试，C++ 目标统一应用安全编译选项。
- 新增 `.gitcode/workflows/` 门禁流水线配置：编译（x86 / aarch64）、无卡 LLT（ophost UT + Triton CPU 回退精度 + Triton AOT 编译检查）、静态检查。
- 新增 `docs/` 文档目录，含快速入门、目录结构、算子列表、接口列表、开发与调试指引。
- 新增 `examples/` 示例代码与 `scripts/` 辅助脚本（许可证头检查）。
- 新增 `requirements.txt`、`CHANGELOG.md`、`.clang-format`、`.pre-commit-config.yaml`。
- `build.sh` 新增 `--ut` 模式，一条命令跑完全部无卡单元测试。
- 新增 `tests/README.md`（用例分类与 NPU 依赖说明）与 `tests/conftest.py`。
- 新增 `ascendc/pybind/`，集中存放各 AscendC 算子的 Python API 与 C++ 绑定；`pybind_extensions.py` 按 `ascendc/pybind/csrc/<op_family>/*.cpp` 自动发现并生成扩展模块 `<op_family>_npu`。

### Changed

- 根目录 `build.sh` 按 [ops-math `scripts/build_options.sh`](https://gitcode.com/cann/ops-math/blob/master/scripts/build_options.sh) 拆出选项解析：新增 `scripts/build_options.sh`，无卡 UT 主入口改为 `-u`（`--ut` 仍可用），支持 `--ophost` / `--triton` / `--aot` 子集、`--noexec`、`--cov`、`--gtest_filter`。
- 仓库许可证由 CANN Open Software License Agreement 2.0 改为 Apache License 2.0（`LICENSE`、源文件头、`OAT.xml`、`pyproject.toml`）。
- 根目录测试目录由 `test/` 更名为 `tests/`，构建脚本、CI、CMake 与文档同步更新。

- 去掉 AscendC OpType 的 `Og` 后缀：`ComplexMul` / `ComplexMulGrad` / `FusedBiasSoftmax` / `FusedSoftmaxGrad`，aclnn 同步为 `aclnnComplexMul` 等。
- AscendC Python 公开接口为蛇形命名，不加 `npu_` 前缀：`complex_mul`、`fused_bias_softmax`、`fused_softmax_grad`、`attention_core`。
- 根目录构建入口 `build_install_ops.sh` 更名为 `build.sh`，与 CANN 社区仓惯例对齐。原 `ascendc/build.sh` 保持不变，仍是 AscendC 侧的分步入口。
- **测试用例统一收敛到根目录 `tests/`，算子目录不再包含测试用例**：`ascendc/tests/ophost/` → `tests/ascendc/ophost/`；`ascendc/tests/complex_mul/` → `tests/ascendc/complex_mul/`；`triton-ascend/real_fft/` 下的 `test_triton_fft_real.py` 与 `verify_triton_fft_real_128_3d.py` → `tests/triton/real_fft/`。
- `ascendc/build.py ut-ophost` 在检测到 CMake 缓存中的源码目录与当前不一致时自动清理重配，避免搬迁后复用旧缓存报错。
- pybind 封装从测试目录移出：C++ 绑定在 `ascendc/pybind/csrc/`，构建脚本为 `ascendc/pybind_extensions.py` / `ascendc/pybind_setup.py`。
- **删除源码树中的 `oil_gas_ops_prospect/` 目录**。Python 对外 API 按算子名暴露，不再带 `ascendc` / `triton` 二级包：`oil_gas_ops_prospect.complex_mul`、`oil_gas_ops_prospect.real_fft`。`setup.py` 用 `package_dir` 在打包时组装；裸 checkout 由 `scripts/local_pkg.py` 提供同一导入名。
- ophost 单元测试的 googletest 依赖从 `github.com` 直连改为 gitcode 三方件镜像，并抽取到 `cmake/third_party/googletest.cmake` 供顶层与独立配置两种方式复用。
- `README.md` 重写，补齐概述、版本配套、源码下载、目录结构、贡献指南 / 安全声明 / 许可证 / 所属 SIG 链接与问题反馈入口。
- 按现有实现补齐算子文档：`docs/zh/op_list.md` 为每个 AscendC / Triton 算子写明数学定义、输入输出与约束；`docs/zh/api_list.md` 补全 `fused_*` / `attention_core` / `make_basis` 参数表与 aclnn 两段式入口；`ascendc/README.md` 登记融合注意力算子及其切分/测试说明。
- 刷新算子开发规格并迁到 `docs/zh/develop/operator-dev-spec.md`：对齐当前 `tests/`、`ascendc/pybind/`、`build.sh` 与命名约定，去掉开源重整前的 `python/`、`testplan/`、`delivery/` 等过时要求。原 `docs/operator-dev-spec.md` 保留跳转。

## [0.1.0] - 2026-08-27

### Added

- 首次提交 oil-gas-ops-prospect 油气勘探领域昇腾自定义算子库。
- AscendC 算子 `ComplexMul`：复数逐点乘，float32 / ND，AIV-only，产出 CANN OPP `.run` 包。
- AscendC 算子 `ComplexMulGrad`：`ComplexMul` 的反向算子。
- Triton-Ascend 算子 `real_rfft` / `real_irfft`：实数正向 FFT 与半谱重建，支持 autograd，非 NPU 环境回退到 PyTorch。
- 基于 msopgen 的多算子构建流程（`ascendc/build.py` + `ascendc/build.sh`）。
- ophost 侧 tiling 单元测试（googletest），无卡可跑。
- 统一 Python 包 `oil_gas_ops_prospect` 与 whl 打包配置。
