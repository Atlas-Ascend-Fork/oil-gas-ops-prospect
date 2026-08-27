# oil-gas-ops-prospect

油气勘探领域（oil & gas exploration）的昇腾自定义算子库。仓名即 **oil-gas-ops**（油气算子）+ **prospect**（勘探目标/圈闭），面向地震成像、全波形反演等勘探计算负载，不绑定任何单一上层模型工程。

目录结构：

- `ascendc/`：AscendC/CANN OPP 算子工程与 `.run` 包构建入口。
- `triton-ascend/`：Triton-Ascend Python 算子实现。
- `triton-ascend/tools/`：无卡环境的编译验证工具。
- `python/oil_gas_ops_prospect/`：统一 Python 包入口。
- `tests/`：基于已安装 `oil_gas_ops_prospect` whl 的验证用例。
- `.kiro/steering/`：容器编译验证流程与仓库约定的知识库。

## 包命名

| 交付物 | 名称 | 规范 |
|--------|------|------|
| Python 分发名 | `oil-gas-ops-prospect` | PEP 508 |
| import 包名 | `oil_gas_ops_prospect` | PEP 8 |
| whl | `oil_gas_ops_prospect-<ver>-<py>-<abi>-<plat>.whl` | PEP 427（`-`/`.` 归一为 `_`） |
| OPP 算子包 | `oil_gas_ops_prospect_opp_<os>_<arch>.run` | CANN `<pkg>_opp_<os>_<arch>.run` |
| 环境变量前缀 | `OIL_GAS_OPS_` | — |

CANN vendor 目录仍为约定的 `opp/vendors/customize/`。

## AscendC 编译与 UT

推荐在 `ascendc-dev` 容器内、仓库挂载路径下执行（`/root/projects/oil-gas-ops-prospect`）：

```bash
cd /root/projects/oil-gas-ops-prospect/ascendc

# 编译并打包 OPP .run
bash build.sh --pkg

# 运行 ophost 侧单元测试（googletest，无需 NPU 设备）
bash build.sh -u --ophost
```

常用打包参数：

```bash
bash build.sh --pkg -v Ascend910B1 --ops complex_mul complex_mul_grad -j 16
```

所有编译产物统一落到**仓库根目录 `output/`**：

```
output/
├── <SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run   # OPP 算子包
└── wheels/oil_gas_ops_prospect-*.whl                # Python whl
```

需要改产物位置时设 `OIL_GAS_OPS_OUTPUT_DIR`（对 `build.sh --pkg`、`build_install_ops.sh`、
`tests/run_whl_tests.sh` 都生效）。更多说明见 [`ascendc/README.md`](ascendc/README.md)。

## 一键构建并安装（OPP + whl）

如需同时完成 OPP 安装、`oil_gas_ops_prospect` whl 构建与冒烟测试，可在仓库根目录执行：

```bash
cd /root/projects/oil-gas-ops-prospect
bash build_install_ops.sh
```

脚本会按顺序完成：

1. 调用 `ascendc/build.sh --pkg` 编译 AscendC OPP `.run` 包（默认 `Ascend910B1`，默认全部算子）。
2. 安装生成的 `oil_gas_ops_prospect_opp_*.run` 到当前 CANN 环境。
3. 构建统一的 `oil_gas_ops_prospect` whl 包。
4. 通过 `pip` 安装 whl 包。
5. 运行 `tests/test_oil_gas_ops_prospect_whl.py` 冒烟测试。

常用参数：

```bash
bash build_install_ops.sh --clean
bash build_install_ops.sh --ops complex_mul complex_mul_grad
bash build_install_ops.sh --skip-tests
bash build_install_ops.sh --soc Ascend910B1 -j 16
```

默认 CANN 路径为 `/usr/local/Ascend/cann-9.0.0`。如容器环境不同，可显式指定：

```bash
ASCEND_HOME_PATH=/path/to/cann bash build_install_ops.sh
```

安装完成后，后续 shell 中需要保留自定义 opapi 动态库路径：

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/customize/op_api/lib:/usr/local/Ascend/cann-9.0.0/lib64:$LD_LIBRARY_PATH
```

## 分步构建

AscendC 分步调试见 [`ascendc/README.md`](ascendc/README.md)。

如果只需要构建并验证 whl 包，可运行：

```bash
cd /root/projects/oil-gas-ops-prospect
bash tests/run_whl_tests.sh
```

## 无卡环境验证

开发容器 `ascendc-dev` 没有 NPU 设备。下列验证**不需要卡**：

```bash
# AscendC：编译打包 + ophost 单元测试
cd ascendc && bash build.sh --pkg && bash build.sh -u --ophost

# Triton：全部 kernel 分支的 AOT 编译（TTIR → Linalg → bishengir → npubin）
python3 triton-ascend/tools/aot_compile_check.py

# CPU 回退精度
python3 -m pytest triton-ascend/real_fft/test_triton_fft_real.py -q
```

算子在 NPU 上的**精度、性能、显存**必须上真实 910B 验证，编译通过不等于功能正确。
完整的容器操作流程、环境事实与踩坑记录见 [`.kiro/steering/`](.kiro/steering/)。

## Triton-Ascend 本地源码 / whl 切换

`triton-ascend/` 下的验证脚本默认直接使用当前仓库里的 Triton-Ascend Python 文件，修改算子后不需要重新构建 `oil_gas_ops_prospect` whl，也不需要配置环境变量：

```bash
cd /root/projects/oil-gas-ops-prospect
python3 triton-ascend/real_fft/verify_triton_fft_real_128_3d.py --print-shape-only
```

如果需要验证已安装的 whl 包内容，再显式配置环境变量切到 whl 模式：

```bash
cd /root/projects/oil-gas-ops-prospect
OIL_GAS_OPS_TRITON_SOURCE=whl \
python3 triton-ascend/real_fft/verify_triton_fft_real_128_3d.py --print-shape-only
```

环境变量取值：

- 未设置或为空：默认优先导入 `triton-ascend/` 本地文件。
- `local` / `source` / `file`：优先导入 `triton-ascend/` 本地文件。
- `whl` / `wheel` / `installed`：不注入本地源码路径，使用当前 Python 环境里安装的 `oil_gas_ops_prospect` 包。
