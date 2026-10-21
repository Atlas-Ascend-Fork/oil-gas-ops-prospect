# 编译参数说明

简体中文 | [English](../../en/context/build.md)

## 简介

`build.sh` 位于仓库根目录，把源码编译成可安装的 OPP `.run` 与 Python whl，或按分类执行测试并写出报告。选项解析在 [`scripts/build_options.sh`](../../../scripts/build_options.sh)。

脚本按用途分成两条互斥路径，一次只走其中一条：

| 路径 | 怎么进入 | 做什么 |
|------|----------|--------|
| [编译构建](#编译构建) | 不加 `-u` | 编包到 `output/`；加 `--install` 才安装 |
| [测试报告](#测试报告) | `-u` | 不编包、不安装；只跑测并写报告 |

`--pkg`、`--install` 不能与 `-u` 同时使用，彼此也不能组合。本仓没有 ops-math 的 `--opapi` / `--opgraph` / `--opkernel`。

## 使用方法

1. 配置环境变量

   环境部署见[环境部署](quick_install.md)。默认路径安装、以 root 用户为例：

   ```bash
   source /usr/local/Ascend/cann-9.0.0/set_env.sh
   ```

   CANN 不在默认路径时：

   ```bash
   export ASCEND_HOME_PATH=/path/to/cann
   source ${ASCEND_HOME_PATH}/set_env.sh
   ```

2. 查看帮助

   ```bash
   bash build.sh --help             # 总览
   bash build.sh --pkg -h           # 编译构建分节
   bash build.sh --install -h       # 同上
   bash build.sh -u -h              # 测试报告分节
   bash build.sh --clean -h         # 清理分节
   ```

   `--soc=SOC` / `--soc SOC`、`--ops=a,b` / `--ops a,b`、`-j16` / `-j 16` 均支持。全量参数含义见下方两节，按实际场景选用。

---

## 编译构建

不加 `-u` 时走构建路径。根目录 `build.sh` 是主入口；`ascendc/build.sh` 与顶层 CMake 见节末[其它入口](#其它入口)。

### 构建命令格式

以编译并安装为例。`--soc`、`--ops`、`-j` 在该场景为可选项，合适的编译线程数可加快编译速度。

```bash
bash build.sh --install [--soc=${soc_version}] [--ops=${op_list}] [-j${n}]
```

### 三种入口

三种入口互斥，一次只选一种。

**单独编译 OPP `.run` 包**

```bash
bash build.sh --pkg
```

只调用 `ascendc/build.sh --pkg` 生成 OPP `.run`，写入 `output/<SOC>/`。不编 Python whl，不安装到 CANN。

**只编译不安装（默认）**

```bash
bash build.sh
```

编译 OPP `.run` 与 Python whl，产物写入 `output/`，不执行 `.run` 安装器、也不 `pip install`。顶层 `bash build.sh` 先执行 `build_opp_run`，再执行 `build_python_wheel`；`ascendc/pybind_extensions.py::_vendor_opapi` 优先使用 `ascendc/_work/CustomOp_*/build_out/autogen` 中的头文件和 `ascendc/_work/CustomOp_*/build_out/op_host/libcust_opapi.so`，所以默认顶层构建无需先安装 OPP。只有脱离顶层流程直接执行 `python3 setup.py bdist_wheel` 且工作区无有效 OPP 产物时，才回退到已安装的 CANN vendor 目录，并要求其中已有相应头文件和库。

**编译并安装**

```bash
bash build.sh --install
```

编译 OPP 与 whl，安装到 CANN / Python，并跑 whl 冒烟测试。安装完成后，后续 shell 需要保留自定义 opapi 动态库路径：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
```

### 参数说明

`bash build.sh --help` 可查看全部构建参数。

| 参数名 | 可选/必选 | 参数说明 |
|--------|-----------|----------|
| `--pkg` | 可选 | 只编译 OPP `.run`，不编 whl、不安装。不可与 `--install`、`-u` 同时使用。 |
| `--install` | 可选 | 编译 `.run` 与 whl，安装后跑冒烟。不可与 `--pkg`、`-u` 同时使用。 |
| （默认，无上述开关） | — | 编译 `.run` 与 whl 到 `output/`，不安装。 |
| `--soc=${soc}` | 可选 | 指定 NPU 型号，默认 `Ascend910B1`。每次编译只支持 1 个型号。 |
| `--ops=${op_list}` | 可选 | 指定待编译的 AscendC 算子，逗号或空格分隔，默认全部。 |
| `-j${n}`, `--jobs <N>` | 可选 | 编译线程数，默认 CPU 核数（如 `-j16`）。 |
| `--clean` | 可选 | 构建前清理 `ascendc/_work` 与 `output/`，然后继续执行所选入口。 |
| `--skip-tests` | 可选 | 仅配合 `--install`：安装但跳过 whl 冒烟测试。 |
| `-h`, `--help` | 可选 | 打印帮助；与 `--pkg` / `--install` / `--clean` 组合打印对应分节。 |

### 示例

```bash
bash build.sh --pkg                                 # 只出 OPP .run
bash build.sh                                       # 出 .run 与 whl，不安装
bash build.sh --install                             # 编译并安装
bash build.sh --clean                               # 先清理再编译（默认仍不安装）
bash build.sh --ops=complex_mul,complex_mul_grad    # 只构建指定算子
bash build.sh --install --skip-tests                # 安装后跳过冒烟测试
bash build.sh --soc=Ascend910B1 -j16 --install      # 指定 SOC 与并行度并安装
```

### 产物位置

所有编译产物统一落到仓库根目录 `output/`：

```text
output/
├── <SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run   # OPP 算子包
└── wheels/oil_gas_ops_prospect-*.whl                # Python whl
```

设置 `OIL_GAS_OPS_OUTPUT_DIR` 可改变该位置，对 `build.sh`、`ascendc/build.sh --pkg`、`tests/run_whl_tests.sh` 均生效。

| 交付物 | 名称 | 规范 |
|--------|------|------|
| Python 分发名 | `oil-gas-ops-prospect` | PEP 508 |
| import 包名 | `oil_gas_ops_prospect` | PEP 8 |
| whl | `oil_gas_ops_prospect-<ver>-<py>-<abi>-<plat>.whl` | PEP 427（`-`/`.` 归一为 `_`） |
| OPP 算子包 | `oil_gas_ops_prospect_opp_<os>_<arch>.run` | CANN `<pkg>_opp_<os>_<arch>.run` |
| 环境变量前缀 | `OIL_GAS_OPS_` | — |

CANN vendor 目录仍为约定的 `opp/vendors/customize/`。

不需要重新编译 AscendC 算子、只验证已安装的 whl 时：

```bash
bash tests/run_whl_tests.sh
```

### 其它入口

日常使用根目录 `build.sh` 即可。下面两个入口给需要分步调试或只编 ophost tiling 单元测试的场景。以下命令均从仓库根目录执行。

**`ascendc/build.sh`**：AscendC 侧分步入口，只打包 OPP `.run`，或只跑 ophost UT。

```bash
bash ascendc/build.sh --pkg [options]        # 编译并打包 OPP .run
bash ascendc/build.sh -u --ophost [options]  # 编译并运行 ophost 单元测试
bash ascendc/build.sh clean                  # 清理 ascendc/_work 与 output/
bash ascendc/build.sh list-ops               # 列出 manifest 中的算子
```

| 参数名 | 可选/必选 | 参数说明 |
|--------|-----------|----------|
| `-v`, `--soc-version <SOC>` | 可选 | msopgen SOC 型号，默认 `Ascend910B1`，支持逗号分隔的多型号 |
| `--ops <ops...>` | 可选 | 算子目录名，可多个或用逗号分隔，默认全部 |
| `-j`, `--jobs <N>` | 可选 | 编译并行度，默认 CPU 核数 |
| `-i`, `--install-path <path>` | 可选 | CANN 安装路径，默认自动探测 |
| `--vendor-name <name>` | 可选 | OPP 安装目录 `opp/vendors/<name>/`，默认 `customize` |
| `--package-name <name>` | 可选 | 替换 `.run` 名中的 `custom` 前缀，默认 `oil_gas_ops_prospect` |

```bash
bash ascendc/build.sh --pkg -v Ascend910B1 --ops complex_mul complex_mul_grad -j 16
python3 ascendc/build.py -v Ascend910B1 --ops complex_mul complex_mul_grad   # 等价 Python 入口
```

**顶层 CMake**：只覆盖 ophost 侧 tiling 单元测试。AscendC 算子本身不由顶层 CMake 编译，必须经 msopgen 生成工程后再由 `bash build.sh --pkg` 或 `bash ascendc/build.sh --pkg` 编译。

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

版本与依赖定义集中在 [`version.cmake`](../../../version.cmake)：`OIL_GAS_OPS_PROSPECT_VERSION`（需与 `pyproject.toml` / `setup.py` 一致）、`OIL_GAS_OPS_PROSPECT_CANN_VERSION`、`OIL_GAS_OPS_PROSPECT_SOC_VERSIONS`、以及构建期 / 运行期依赖变量。

---

## 测试报告

`-u` 进入测试模式：不编包、不安装。分类与用例路径以 [`tests/README.md`](../../../tests/README.md) 为准。本节能直接写出两类产物：

| 产物 | 何时生成 | 路径 |
|------|----------|------|
| 执行纪要 | 每次 `-u` | `test_result/test_report.md`（另存带时间戳的副本）与 `test_result/test_results.json` |
| 开源测试报告 | 加 `--opensource-report` | `test_result/oil-gas-ops-prospect仓开源测试报告.md`，并覆盖 `docs/zh` 同名文件 |

### 测试命令格式

不跟子集时跑全部（含 FFT 大形态与开源门禁）。`--ops`、`--report`、`--opensource-report` 为可选项。

```bash
bash build.sh -u [--ut|--st|--precision|--perf|...] [--ops=${op_list}] [--opensource-report]
```

无 NPU 时，全量 `-u` 的上板项 **SKIP**，不失败。显式指定 `--ut` / `--st` / `--precision` / `--perf` / `--examples` 且无 NPU 时 **FAIL**。

`--ut` / `--st` / `--precision` / `--perf` 等子集开关本身也会进入测试模式，因此 `bash build.sh --ut` 与 `bash build.sh -u --ut` 等价。CI `JOB_llt` 走 cmake/ctest，不调用 `build.sh -u`。

### 参数说明

`bash build.sh -u -h` 可查看测试分节帮助。

| 参数名 | 可选/必选 | 对应类别 | 参数说明 |
|--------|-----------|----------|----------|
| `-u` | 进入测试模式 | 全部 | 不跟子集时跑下表全部 |
| `--all` | 可选 | 全部 | 显式全量，与裸 `-u` 相同 |
| `--ut` | 可选 | **UT** | UT-host（切分）+ UT-device（上板小 launch）；需要 NPU |
| `--ophost` | 可选 | UT-host | 只跑 tiling googletest |
| `--st` | 可选 | **ST** | 多 shape C++ aclnn + golden；需要 NPU |
| `--precision` | 可选 | **精度** | NPU Python 对拍 |
| `--perf` | 可选 | **性能** | NPU Python 计时 |
| `--examples` | 可选 | UT-device | 只跑 C++ 小 launch；需要 NPU |
| `--triton` | 可选 | Triton CPU | CPU 回退 pytest |
| `--composite` | 可选 | Composite CPU | 纯 PyTorch 组合算子的 CPU pytest |
| `--aot` | 可选 | Triton AOT | kernel 可编译检查 |
| `--ops=${op_list}` | 可选 | 过滤 | 作用于 UT-device / ST / 精度 / 性能 |
| `--report=<path>` | 可选 | — | 执行纪要 Markdown 路径 |
| `--opensource-report[=path]` | 可选 | — | 按本次 `-u` 的 JSON 生成开源测试报告；单独使用时只读已有 JSON、不再跑测 |
| `--noexec` | 可选 | UT-host | 只编译，不执行 |
| `--cov` | 可选 | UT-host | 打开 `--coverage`；Triton 在已装 pytest-cov 时加 `--cov` |
| `--gtest_filter=pattern` | 可选 | UT-host | 过滤 gtest |
| `--clean` | 可选 | — | 跑测前清理 `ascendc/_work` 与 `output/` |

`--noexec` / `--cov` / `--gtest_filter` / `--report` 只能与测试模式一起用。`--opensource-report` 不能与 `--pkg` / `--install` 组合。

### 示例

```bash
bash build.sh -u                                 # 全部
bash build.sh -u --ut                            # UT（切分 + 上板）
bash build.sh -u --st --precision --perf
bash build.sh -u --ophost                        # 只切分
bash build.sh -u --perf --ops=complex_mul
bash build.sh -u --report=test_result/my.md      # 自定义执行纪要路径
bash build.sh -u --opensource-report             # 跑测并生成开源测试报告
bash build.sh --opensource-report                # 基于上次 -u 的 JSON 只出报告
```

### 执行纪要

每次 `bash build.sh -u` 跑完都会落盘：

```text
test_result/
├── test_report_<时间戳>.md    # 本次执行纪要
├── test_report.md             # 上一份纪要的稳定副本
└── test_results.json          # 开源测试报告的数据源
```

`--report=<path>` 只改 Markdown 纪要路径，不改变 JSON 位置。

### 开源测试报告

社区模板开源测试报告**不**随裸 `-u` 自动生成，需要显式加 `--opensource-report`。

| 命令 | 行为 |
|------|------|
| `bash build.sh -u --opensource-report` | 跑齐测试（含文档 / 出包入口 / 安装痕迹 / 接口冒烟，以及 FFT 大形态），再按本次 JSON **生成**报告 |
| `bash build.sh --opensource-report` | 只读已有 `test_result/test_results.json`，不再跑测 |

默认写出 `test_result/oil-gas-ops-prospect仓开源测试报告.md`，并同步覆盖 `docs/zh` 同名文件。有 FAIL 时结论为不建议发布，不会回填旧数字。

分类表、套件目录与新增算子应挂哪些测试，见 [`tests/README.md`](../../../tests/README.md)。
