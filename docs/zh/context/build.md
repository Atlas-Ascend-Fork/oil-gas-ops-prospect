# 编译参数说明

本仓构建入口职责如下：

| 入口 | 职责 |
|------|------|
| `build.sh`（仓库根目录） | 一键完成 OPP 编译 → 安装 → whl 构建 → whl 安装 → 冒烟测试；`-u` 模式跑无卡单元测试 |
| `scripts/build_options.sh` | 被根目录 `build.sh` source：选项解析、互斥校验与分节帮助（对齐 [ops-math](https://gitcode.com/cann/ops-math/blob/master/scripts/build_options.sh)） |
| `ascendc/build.sh` | AscendC 侧分步入口：只打包 OPP `.run`，或只跑 ophost UT |
| `CMakeLists.txt`（顶层 CMake） | 无卡 C++ 目标（ophost tiling UT）的标准 CMake 构建 |

## `build.sh`

```bash
bash build.sh [options]
bash build.sh -h            # 总览
bash build.sh -u -h         # Test Options
bash build.sh --clean -h    # Clean Options
```

选项解析在 [`scripts/build_options.sh`](../../../scripts/build_options.sh)。`--soc=SOC` / `--soc SOC`、`--ops=a,b` / `--ops a,b`、`-j16` / `-j 16` 均支持。

### 默认（打包安装）

不加 `-u` 时走全流程：OPP → 安装 → whl → 冒烟。

| 选项 | 默认 | 说明 |
|------|------|------|
| `--soc=<SOC>` | `Ascend910B1` | msopgen SOC 型号 |
| `--ops=op1,op2,...` | `all` | 逗号或空格分隔的 AscendC 算子名 |
| `-j[n]`, `--jobs <N>` | CPU 核数 | 编译并行度，透传给 `ascendc/build.py` |
| `--clean` | 关 | 构建前清理 `ascendc/_work` 与 `output/` |
| `--skip-tests` | 关 | 安装但跳过 whl 冒烟测试 |
| `-h, --help` | — | 打印帮助；与 `-u` / `--clean` 组合打印对应分节 |

`--pkg` 不能与 `-u` 同时使用（不加 `-u` 本来就是打包路径）。

```bash
bash build.sh                                       # 全流程
bash build.sh --clean                               # 先清理再全流程
bash build.sh --ops=complex_mul,complex_mul_grad    # 只构建指定算子
bash build.sh --skip-tests                          # 跳过冒烟测试
bash build.sh --soc=Ascend910B1 -j16                # 指定 SOC 与并行度
```

### Test Options（`-u`）

对齐 ops-math 的 `Test Options`：`-u` 跑全部无卡 UT；加上 `--ophost` / `--triton` / `--aot` 则只跑对应子集。`--ut` 是 `-u` 的别名。

| 选项 | 说明 |
|------|------|
| `-u`, `--ut` | 编译并运行全部无卡单元测试（不打包不安装） |
| `--noexec` | 只编译 ophost UT，不执行；Triton / AOT 同步跳过 |
| `--cov` | 打开覆盖率：ophost 用 `--coverage`；Triton 在已装 `pytest-cov` 时加 `--cov` |
| `--gtest_filter=pattern` | 只跑匹配的 ophost gtest（需配合 `-u` 或 `-u --ophost`） |
| `--ophost -u` | 只跑 ophost tiling UT（googletest） |
| `--triton -u` | 只跑 Triton CPU 回退 pytest |
| `--aot -u` | 只跑 Triton AOT 编译检查 |

```bash
bash build.sh -u                                 # 全部无卡 UT
bash build.sh -u --noexec                        # 只编译 ophost UT
bash build.sh -u --ophost                        # 只跑 ophost
bash build.sh -u --ophost --gtest_filter=ComplexMul*
bash build.sh -u --triton --aot
bash build.sh --ophost --triton --aot -u         # 与上一行等价（-u 可放后面）
bash build.sh --ut                               # 兼容旧写法，等同 -u
```

`--ophost` / `--triton` / `--aot` 必须搭配 `-u`。本仓没有 ops-math 的 `--opapi` / `--opgraph` / `--opkernel` UT。

## `ascendc/build.sh`

```bash
cd ascendc
bash build.sh --pkg [options]        # 编译并打包 OPP .run
bash build.sh -u --ophost [options]  # 编译并运行 ophost 单元测试
bash build.sh clean                  # 清理 ascendc/_work 与 output/
bash build.sh list-ops               # 列出 manifest 中的算子
```

打包相关选项（透传给 `build.py`）：

| 选项 | 默认 | 说明 |
|------|------|------|
| `-v, --soc-version <SOC>` | `Ascend910B1` | msopgen SOC 型号，支持逗号分隔的多型号 |
| `--ops <ops...>` | 全部 | 算子目录名，可多个或用逗号分隔 |
| `-j, --jobs <N>` | CPU 核数 | 编译并行度 |
| `-i, --install-path <path>` | 自动探测 | CANN 安装路径 |
| `--vendor-name <name>` | `customize` | OPP 安装目录 `opp/vendors/<name>/` |
| `--package-name <name>` | `oil_gas_ops_prospect` | 替换 `.run` 名中的 `custom` 前缀 |

示例：

```bash
bash build.sh --pkg -v Ascend910B1 --ops complex_mul complex_mul_grad -j 16
bash build.sh --pkg --vendor-name customize --package-name oil_gas_ops_prospect
```

等价的 Python 入口：

```bash
python3 build.py -v Ascend910B1 --ops complex_mul complex_mul_grad
python3 build.py clean
python3 build.py list-ops
```

## 顶层 CMake

顶层 CMake 只覆盖**无卡即可构建**的 C++ 目标，即 ophost 侧 tiling 单元测试：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

| CMake 选项 | 默认 | 说明 |
|------------|------|------|
| `BUILD_OPHOST_UT` | `ON` | 是否构建 ophost tiling 单元测试 |
| `CMAKE_BUILD_TYPE` | `Release` | 构建类型 |
| `OIL_GAS_GTEST_VERSION` | `1.14.0` | googletest 版本 |
| `OIL_GAS_GTEST_URL` | gitcode 镜像地址 | googletest 源码包地址，仅在本地找不到 gtest 时使用 |

**AscendC 算子本身不由顶层 CMake 编译**：CANN 自定义算子必须由 msopgen 依据当前 CANN 与 SOC 生成配套工程后，再交给 opbuild / bisheng 编译，入口是 `bash build.sh` 与 `bash ascendc/build.sh --pkg`。

版本与依赖定义集中在 [`version.cmake`](../../../version.cmake)：

| 变量 | 说明 |
|------|------|
| `OIL_GAS_OPS_PROSPECT_VERSION` | 仓库版本，需与 `pyproject.toml` / `setup.py` 一致 |
| `OIL_GAS_OPS_PROSPECT_CANN_VERSION` | 已验证的 CANN 版本 |
| `OIL_GAS_OPS_PROSPECT_SOC_VERSIONS` | 已验证的 SoC 型号 |
| `OIL_GAS_OPS_PROSPECT_BUILD_DEPENDENCIES` | 构建期依赖 |
| `OIL_GAS_OPS_PROSPECT_RUN_DEPENDENCIES` | 运行期依赖 |

## 产物位置

所有编译产物统一落到仓库根目录 `output/`：

```text
output/
├── <SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run   # OPP 算子包
└── wheels/oil_gas_ops_prospect-*.whl                # Python whl
```

设置 `OIL_GAS_OPS_OUTPUT_DIR` 可改变该位置，对 `build.sh`、`ascendc/build.sh --pkg`、`tests/run_whl_tests.sh` 均生效。

## 只构建并验证 whl

不需要重新编译 AscendC 算子时：

```bash
bash tests/run_whl_tests.sh
```

## 包命名规范

| 交付物 | 名称 | 规范 |
|--------|------|------|
| Python 分发名 | `oil-gas-ops-prospect` | PEP 508 |
| import 包名 | `oil_gas_ops_prospect` | PEP 8 |
| whl | `oil_gas_ops_prospect-<ver>-<py>-<abi>-<plat>.whl` | PEP 427（`-`/`.` 归一为 `_`） |
| OPP 算子包 | `oil_gas_ops_prospect_opp_<os>_<arch>.run` | CANN `<pkg>_opp_<os>_<arch>.run` |
| 环境变量前缀 | `OIL_GAS_OPS_` | — |

CANN vendor 目录仍为约定的 `opp/vendors/customize/`。
