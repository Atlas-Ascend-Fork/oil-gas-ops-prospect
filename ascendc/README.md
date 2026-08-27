# ascendc — 基于 msopgen 的多算子构建

`oil-gas-ops-prospect/ascendc`：**每次构建都从 msopgen 重新生成**与当前 CANN、SOC 匹配的工程，再合并多算子源码。

## 设计思路

```
install.sh 单算子流程:
  msopgen gen → cp 单算子实现 → build.sh

本目录多算子扩展:
  msopgen gen (内嵌 DummyOp IR)  →  与 CANN/SOC 匹配的脚手架
  merge_operators (Python)        →  拷贝 --ops 选中算子 + overlay 多 kernel CMake
  CustomOp/build.sh               →  msopgen 自带 binary + package（仍为 shell）
```

## 目录结构

```
ascendc/
├── build.py                 # 打包实现（msopgen + merge + CustomOp）
├── build.sh                 # 入口：`--pkg` 打包 / `-u --ophost` 跑 UT
├── operators/
│   ├── manifest.tsv
│   └── <name>/op_host|op_kernel/
├── scripts/
│   ├── config.py            # 常量、SOC 列表、CMake 模板、msopgen 占位 IR
│   ├── manifest.py          # 读取 manifest.tsv
│   ├── merge_operators.py   # 合并算子进 CustomOp
│   ├── patch_presets.py     # SOC / AddConfig 对齐
│   └── cann_env.py          # CANN 环境 + 子 shell 执行
└── _work/                   # msopgen 临时工程（含 ophost_ut/ 构建目录）
```

编译产物不落在本目录，统一放到**仓库根目录 `output/`**：

```
output/
├── <SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run   # OPP 算子包（build.sh --pkg）
└── wheels/oil_gas_ops_prospect-*.whl                # whl（build_install_ops.sh / tests/run_whl_tests.sh）
```

`OIL_GAS_OPS_OUTPUT_DIR` 可覆盖该目录。

## 编译与 UT

在本目录执行（容器内一般为 `/root/projects/oil-gas-ops-prospect/ascendc`）：

```bash
# 编译并打包 OPP .run（默认 SOC=Ascend910B1，默认全部算子）
# → <repo>/output/Ascend910B1/oil_gas_ops_prospect_opp_<os>_<arch>.run
bash build.sh --pkg

# 运行 ophost 侧单元测试（googletest，无需 NPU 设备）
bash build.sh -u --ophost
```

ophost UT 工程在 [`tests/ophost/`](tests/ophost/)，覆盖两部分：

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_complex_mul_tiling.cpp` | 两个算子的 `*_tiling_core.h`：UB 预算约束（前向 13 份 tile / 反向 21 份）、`tileFloats` 的 32B 对齐与上限、小 tensor 收缩、退化输入（负元素数、`aivNum=0`、空张量）、以及 kernel 侧核间切分的覆盖性不变量（每个元素恰好被一个 AIV 核处理一次、区间无空洞无重叠） |

> `<Op>TilingFunc` / `InferShape` 的入口签名依赖 `gert::TilingContext`，`Compute<Op>Tiling`
> 依赖 `platform_ascendc::PlatformAscendC`，CANN 二进制包未提供 gert faker（该基建在
> metadef 源码仓），两者都无法直接 UT。因此各算子把与平台无关的切分算法抽到
> `op_host/<op>_tiling_core.h`，UT 直接驱动核心函数，覆盖的是真实算法。产物在
> `_work/ophost_ut/`。
>
> 新增算子时：把它的 `op_host` 目录加进 `tests/ophost/CMakeLists.txt` 的
> `OPHOST_UT_INCLUDE_DIRS`，并把测试源加进 `ophost_ut` 目标。

常用打包参数：

```bash
bash build.sh --pkg -v Ascend910B1 --ops complex_mul complex_mul_grad -j 16
bash build.sh --pkg --vendor-name customize --package-name oil_gas_ops_prospect
bash build.sh clean
bash build.sh list-ops
```

`--ops` 支持空格分隔或逗号；当前 manifest 包含 `complex_mul`、`complex_mul_grad`。

等价的 Python 入口：

```bash
python3 build.py -v Ascend910B1 --ops complex_mul complex_mul_grad
```

## 一键构建并安装 OPP + whl

如果目标是直接使用自定义算子（含 whl），优先在仓库根目录执行：

```bash
cd /root/projects/oil-gas-ops-prospect
bash build_install_ops.sh
```

该脚本会自动完成：

1. 调用本目录的 `bash build.sh --pkg` 编译 AscendC OPP `.run` 包。
2. 安装生成的 `oil_gas_ops_prospect_opp_*.run` 到 CANN。
3. 构建统一的 `oil_gas_ops_prospect` whl 包。
4. 安装 whl 包。
5. 默认运行 `tests/test_oil_gas_ops_prospect_whl.py` 冒烟测试。

常用参数：

```bash
bash build_install_ops.sh --clean               # 先清理 ascendc/_work 与 output/
bash build_install_ops.sh --ops complex_mul     # 只构建指定 AscendC 算子
bash build_install_ops.sh --skip-tests          # 跳过 whl 冒烟测试
bash build_install_ops.sh --soc Ascend910B1 -j 16
```

如 CANN 安装路径不是默认的 `/usr/local/Ascend/cann-9.0.0`，可通过环境变量指定：

```bash
ASCEND_HOME_PATH=/path/to/cann bash build_install_ops.sh
```

### 包名与 vendor

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vendor-name` | `customize` | OPP 安装目录 `opp/vendors/<name>/` |
| `--package-name` | `oil_gas_ops_prospect` | 替换 `.run` 名中的 `custom` 前缀，如 `oil_gas_ops_prospect_opp_openEuler_aarch64.run` |

```bash
# 默认：vendor=customize，产物 oil_gas_ops_prospect_opp_<os>_<arch>.run
python3 build.py -v Ascend910B1

# 自定义前缀
python3 build.py -v Ascend910B1 --package-name my_prefix
# → my_prefix_opp_openEuler_aarch64.run
```

## 脚本模块说明

| 模块　　　　　　　　　　　　 | 作用　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------------------------| ---------------------------------------------------------------|
| `build.py`　　　　　　　　　 | CLI、多 SOC 循环、调用 msopgen / merge / CustomOp/build.sh　　|
| `scripts/manifest.py`　　　　| 解析 `operators/manifest.tsv`　　　　　　　　　　　　　　　　 |
| `scripts/merge_operators.py` | 删 dummy、拷 host/kernel、写多算子 `op_kernel/CMakeLists.txt` |
| `scripts/patch_presets.py`　 | 修正 `CMakePresets.json` 中 ASCEND_COMPUTE_UNIT　　　　　　　 |
| `scripts/patch_package.py`　 | 设置 vendor_name / package_name 与 .run 命名规则　　　　　　　|
| `scripts/cann_env.py`　　　　| 查找 CANN 路径，`source set_env.sh` 后执行命令　　　　　　　　|

单独调试：

```bash
python3 scripts/merge_operators.py _work/CustomOp_Ascend910B1 complex_mul complex_mul_grad
python3 scripts/patch_presets.py _work/CustomOp_Ascend910B1 Ascend910B
```

## 添加新算子

1. `operators/<name>/op_host/` + `op_kernel/`
2. `operators/manifest.tsv` 追加一行（tab 分隔）
3. `python3 build.py -v Ascend910B1 --ops <name>`

### 移植 CANN 框架算子的注意事项

部分从 CANN 原仓移植的算子（如 `conv3d`、`ffn`）的 host tiling/infershape 依赖
CANN 内部框架头（`op_host/tiling_base.h`、`op_host/tiling_templates_registry.h`、
`err/ops_err.h`、`log/log.h` 等），这些头在 msopgen 自定义算子工程的 opbuild 编译
环境下缺失或会拉入 slog 依赖。处理方式（保持源码改动最小）：

- 提供本地 compat shim（如 `ffn_log_compat.h` / `conv3d_log_compat.h`），用等价/降级
  实现替换 `OP_CHECK_IF`、`OP_CHECK_NULL_WITH_CONTEXT`、`OPS_REPORT_*`、`OP_LOG*`、
  `ASCENDC_EXTERN_C` 等宏；
- 仅把不可用的 include 替换为该 shim，不改动 tiling 算法本身；
- 若算子确需额外 CANN 头目录，可在 `scripts/patch_op_host.py` 的 `_EXTRA_INCLUDE_SUBDIRS`
  中追加（会注入到 opbuild 与 host 库的 include 路径）。

## 已集成算子

| 目录名 | OpType | aclnn 接口 | 说明 |
|--------|--------|-----------|------|
| `complex_mul` | `ComplexMulOg` | `aclnnComplexMulOg` | 复数逐点乘（AIV-only），见 `tests/complex_mul/` |
| `complex_mul_grad` | `ComplexMulGradOg` | `aclnnComplexMulGradOg` | 上者的反向（AIV-only） |

两个算子的语义：

```text
fwd:  wr = ur*vr - ui*vi          bwd:  gur =  gwr*vr + gwi*vi
      wi = ur*vi + ui*vr                gui = -gwr*vi + gwi*vr
                                        gvr =  gwr*ur + gwi*ui
                                        gvi = -gwr*ui + gwi*ur
```

全部输入/输出为 float32、`ND`、同 shape。切分策略是按元素总数在 AIV 核间均分（32B 对齐），
核内再按 UB 预算分拍；切分算法在各算子 `op_host/<op>_tiling_core.h`，由 ophost UT 覆盖。

> 算子 OpType 统一加后缀（`Og`(oil & gas) / `Custom`），避免与 CANN 内置算子注册名冲突。
> kernel 入口函数名通过算子 def 的 `opFile.value`/`opInterface.value` 固定为
> `complex_mul` / `complex_mul_grad`，OpType 改名不影响 kernel 源码；生成的 aclnn 接口
> 随 OpType 变化。

## 测试用例

各算子的 NPU 对比测试在 `tests/<op>/`，统一以等价的 **PyTorch/torch_npu** 表达式为基准对比
**精度 / 性能**。每个测试目录结构一致：

```
tests/<op>/
├── cpp/bindings.cpp   # pybind：通过自定义 aclnn 接口调用算子
├── setup.py           # 编译 torch_npu 扩展
├── test_<op>_npu.py   # 精度/性能对比脚本
├── run_test.sh        # 安装 OPP 包 → 编译扩展 → 运行测试
└── README.md
```

另有不依赖 NPU 的 ophost UT，见上文「编译与 UT」。

### 前置条件

1. 已在 `ascendc-dev` 容器中构建出 OPP 包（见上文「用法」），产物在 `<repo>/output/<SOC>/oil_gas_ops_prospect_opp_*.run`。
2. **测试需在带 NPU 设备的机器上运行**，且环境内有 `torch` 与 `torch_npu`。
   编译可在容器内完成，跑测试需真实 NPU。

### 运行

```bash
cd tests/complex_mul
bash run_test.sh

# 指定 shape
bash run_test.sh --shape 1 1 96 160 160

# 指定 OPP 包（默认取 <repo>/output/Ascend910B1/oil_gas_ops_prospect_opp_openEuler_aarch64.run）
OIL_GAS_OPS_RUN_PKG=/path/to/oil_gas_ops_prospect_opp_openEuler_aarch64.run bash run_test.sh
```

`run_test.sh` 两步：安装/更新 OPP 包 → 编译 pybind 扩展并跑
`test_complex_mul_npu.py`。也可手动分步：

```bash
bash ../../../output/Ascend910B1/oil_gas_ops_prospect_opp_openEuler_aarch64.run --quiet  # 安装算子包
python3 test_complex_mul_npu.py --build-ext                                             # 编译扩展并测试
python3 test_complex_mul_npu.py                                                         # 复用已编译扩展
```

### 输出

脚本打印前向/反向的相对误差、前向/反向的时延与加速比，以及 autograd 封装的梯度误差：

```text
[fwd] shape=... rel_err wr=... wi=...
[bwd] rel_err gur=... gui=... gvr=... gvi=...
[perf] fwd torch=...ms ascendc=...ms speedup=...x
[perf] bwd torch=...ms ascendc=...ms speedup=...x
[autograd] gur=... gui=... gvr=... gvi=...
PASS
```

退出码：`0` 精度达标；`1` 超阈值 / 无 NPU / 扩展缺失。

详见 [`tests/complex_mul/README.md`](tests/complex_mul/README.md)。
