---
inclusion: always
---

# 仓库定位与命名约定

**oil-gas-ops-prospect** = 油气算子（oil & gas ops）+ 勘探目标/圈闭（prospect）。
面向地震成像、全波形反演等**油气勘探领域**计算负载的昇腾自定义算子库。

这是一个**领域级算子库，不绑定任何单一上层模型工程**。不要在仓内引入对某个下游
模型仓目录布局的硬编码依赖；需要与下游模型交叉校验时，走环境变量注入（见下）。

## 目录

| 目录 | 内容 |
|------|------|
| `ascendc/` | AscendC/CANN OPP 算子工程，`.run` 包构建入口 |
| `ascendc/operators/<name>/` | 算子源码（`op_host/` + `op_kernel/`），在 `operators/manifest.tsv` 注册 |
| `ascendc/tests/ophost/` | ophost 侧 googletest UT（无需 NPU） |
| `triton-ascend/` | Triton-Ascend Python 算子（`real_fft/`） |
| `triton-ascend/tools/` | 无卡编译验证工具 |
| `python/oil_gas_ops_prospect/` | 统一 Python 包入口 |
| `tests/` | 基于已安装 whl 的验证 |

## 命名规范（改名时必须整套保持一致）

| 交付物 | 名称 | 依据 |
|--------|------|------|
| Python 分发名 | `oil-gas-ops-prospect` | PEP 508 |
| import 包名 | `oil_gas_ops_prospect` | PEP 8 |
| whl | `oil_gas_ops_prospect-<ver>-<py>-<abi>-<plat>.whl` | PEP 427，`-`/`.` 归一为 `_` |
| OPP 算子包 | `oil_gas_ops_prospect_opp_<os>_<arch>.run` | CANN `<pkg>_opp_<os>_<arch>.run` |
| 环境变量前缀 | `OIL_GAS_OPS_` | — |
| CANN vendor 目录 | `opp/vendors/customize/` | **保持不动** |

`vendor_name` 是 CANN 约定的"自定义 vendor"语义，不属于包名。改它会让所有下游的
`LD_LIBRARY_PATH` 和 `setup.py` include 路径失效。

算子 OpType 统一加后缀（`Og` = oil & gas，或 `Custom`）避免与 CANN 内置算子注册名
冲突，例如 `ComplexMulOg`。

## 环境变量

| 变量 | 作用 |
|------|------|
| `OIL_GAS_OPS_TRITON_SOURCE` | `local`(默认)/`whl`，Triton 算子用仓内源码还是已安装包 |
| `OIL_GAS_OPS_SKIP_EXT` | `1` 跳过 pybind 扩展编译（无 CANN 环境下构建纯 Python whl） |
| `OIL_GAS_OPS_REQUIRE_ASCENDC_CONV3D_GELU` | `1` 禁止回退到 torch 实现，强制用自定义算子 |
| `OIL_GAS_OPS_WHL_REQUIRE_NPU_EXT` | `1` 要求 whl 内编译好的 NPU 扩展可导入 |
| `OIL_GAS_OPS_FFT_BLOCK_M` | 覆盖 FFT DFT-matmul 的行 tile |
| `OIL_GAS_OPS_RUN_128_FFT_TEST` / `..._RUN_LATENT3D_FFT_TEST` | `1` 打开重型 NPU 用例 |
| `OIL_GAS_OPS_MODEL_REPO` | 下游模型仓根目录，opt-in 交叉校验用例 |
| `OIL_GAS_OPS_MODEL_CONFIG` | 下游模型 YAML，供 `--from-model-config` 推导形状 |
| `OIL_GAS_OPS_SPECTRAL_CONV` / `..._FFT_BACKBONE` | `<module>:<class>`，下游被对比模块的导入路径（real_fft 的 opt-in 交叉校验用） |

## 下游耦合的正确做法

算子的**溯源注释**（"本 kernel 等价于下游某文件某几行"）是有价值的文档，保留。
但**可执行路径**不得硬编码下游布局：用 `OIL_GAS_OPS_MODEL_REPO` +
`<module>:<class>` 形式的环境变量注入，未配置时 `pytest.skip` 或 `SystemExit`
并给出可执行提示。
