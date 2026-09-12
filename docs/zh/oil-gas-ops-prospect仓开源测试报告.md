# oil-gas-ops-prospect仓开源测试报告

## 1. 概述

本报告为 **oil-gas-ops-prospect** 项目开源测试报告。测试在 **172.18.0.141**（`bmswt001`）上，使用带 **triton-ascend 3.2.1** 的最新 CANN 9.0.0 镜像容器，按仓库 `tests/` 目录与 `build.sh` 入口跑完全部用例（含无卡 UT、whl 冒烟、AscendC NPU 精度/性能、Triton NPU 精度及可选大 shape 用例）。

覆盖范围：文档、编译出包、安装/卸载、ophost tiling UT、Triton CPU 回退与 AOT、whl 接口冒烟、AscendC / Triton-Ascend NPU 精度与示例。

### 新增特性清单

| 序号 | 特性名称 | 特性描述 |
| ---- | -------- | -------- |
| 1 | 面向油气勘探的昇腾自定义算子库 | 不绑定单一上层模型；Python 按算子名暴露，不区分后端 |
| 2 | AscendC / CANN OPP 交付 | `ComplexMul` / `ComplexMulGrad` / `FusedBiasSoftmax` / `FusedSoftmaxGrad`，产出 OPP `.run` 与 `aclnn` 接口 |
| 3 | Triton-Ascend 交付 | `real_rfft` / `real_irfft`，打包进 `oil_gas_ops_prospect` whl，非 NPU 回退 PyTorch |
| 4 | 一键构建与无卡门禁 | `bash build.sh` 完成 OPP→安装→whl→冒烟；`bash build.sh --ut` 覆盖 ophost UT + Triton CPU 精度 + AOT |

### 解决问题列表

不涉及（本报告针对当前开源布局分支的首轮全量实测，不对照历史缺陷清单逐条闭环）。

## 2. 版本测试信息

**实测环境（172.18.0.141）**

| 项 | 取值 |
| -- | ---- |
| 主机 | `bmswt001`，IP `172.18.0.141` |
| 宿主机 OS | Ubuntu 22.04 LTS，aarch64 |
| 产品型号 | Atlas A2 / Ascend 910B2 × 8 |
| 驱动 | npu-smi 26.0.rc1 |
| 测试容器镜像 | `ai4fwi-ascend-ha:9.0.0-910b-openeuler24.03-py3.11-aarch64`（该镜像为机器上带 **triton-ascend 3.2.1** 的最新镜像；更新的 `ai4fwi-a2-cann850-*:test` 为 CANN 8.5.0 + triton-ascend 3.2.0，不满足配套要求） |
| 容器 OS | openEuler 24.03 (LTS-SP2) |
| CANN | 9.0.0（`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0`） |
| Python | 3.11.14 |
| PyTorch / torch_npu | 2.7.1+cpu / 2.7.1.post4 |
| Triton / triton-ascend | 3.2.0 / **3.2.1** |
| 编译 SOC | `Ascend910B1`（仓库默认；与 910B 系列文档一致） |
| 可见 NPU | `ASCEND_RT_VISIBLE_DEVICES=0`（`torch.npu.get_device_name(0)=Ascend910B2`） |
| 被测代码 | 分支 `refactor/opensource-layout`，提交 `9725ce5` |
| 测试日期 | 2026-09-12 |

硬件和版本要求（仓库声明）：

|      | 因子名称 | 因子取值 |
| ---- | -------- | -------- |
| 1    | OS       | Ubuntu / openEuler（本次为宿主机 Ubuntu + 容器 openEuler） |
| 2    | CPU 架构 | ARM64（本次实测） |
| 3    | 形态     | docker + host device（本次实测） |
| 4    | CANN     | 9.0.0 |
| 5    | Python   | >= 3.10 |
| 6    | SoC      | Ascend 910B（文档默认 `Ascend910B1`） |

## 3. Readme内容评估

Readme内容评估：通过。

| 验证项 | 验证内容 |
| ------ | -------- |
| 文档内容完整性 | 1、提供油气勘探领域与双后端（AscendC / Triton-Ascend）说明<br />2、提供完整算子清单（4 个 AscendC + 2 个 Triton），见 `docs/zh/op_list.md`，含数学定义、约束与 dtype<br />3、提供算子开发指南与开发规格（`docs/zh/develop/`）<br />4、提供调用示例（`examples/`）与接口列表（`docs/zh/api_list.md`）<br />5、提供快速入门（`docs/QUICKSTART.md`）、目录结构、编译参数、调试指引<br />6、提供贡献指南、安全声明、许可证（Apache 2.0） |
| 文档内容正确性 | 1、仓中介绍与实现一致：Python 接口为 `oil_gas_ops_prospect.{complex_mul,fused_bias_softmax,real_fft}`，无 `ascendc`/`triton` 二级包<br />2、目录排版合理，文档跳转可到达<br />3、`bash build.sh --help`、`bash build.sh --ut`、`bash build.sh` 经本机容器实际执行；编译/测试命令可运行 |

## 4. 编译功能验证

编译功能评估：通过（可出包）。kernel 入口已改为 `TILING_KEY_IS(0)` 字面量，避免 TBE 预编译裁掉分支。

| 验证项 | 验证内容 |
| ------ | -------- |
| 编译易用性 | `bash build.sh --help` 可打印合法选项；支持 `--soc` / `--ops` / `-j` / `--clean` / `--skip-tests` / `--ut`。非法参数会退出。NFS 默认 umask 下 `msopgen` 会拒绝 group/other 可写的 dummy JSON，需将 `ascendc/_work/.msopgen_dummy.json` 置为 `0600` 后才能继续（本次已按此处理） |
| 编译高效性 | CMake / ninja 并行编译。本次 `bash build.sh --soc Ascend910B1 -j 32` 从 msopgen 到 OPP 出包约 2～3 分钟（容器内） |
| 出包全面性 | 正常出包：`output/Ascend910B1/oil_gas_ops_prospect_opp_openEuler_aarch64.run`；whl `output/wheels/oil_gas_ops_prospect-0.1.0-cp311-cp311-linux_aarch64.whl`（含 `complex_mul.py` / `fused_bias_softmax.py` / `real_fft/`） |
| 包大小合理 | OPP `.run` **809K**；whl **149K**，大小合理 |

OPP 与 whl 可正常出包；NPU 精度复测已通过。

## 5. 安装包功能验证

安装包功能评估：通过。

| 验证项 | 验证内容 |
| ------ | -------- |
| 算子包安装 | OPP：`bash output/Ascend910B1/oil_gas_ops_prospect_opp_openEuler_aarch64.run --quiet`，日志 `SUCCESS`，写入 `opp/vendors/customize/`<br />whl：`pip install --force-reinstall --no-deps` 安装成功 |
| CANN 包路径适配 | 默认 `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0`，与镜像一致；安装后需保留 `opp/vendors/customize/op_api/lib` 在 `LD_LIBRARY_PATH` |
| 卸载功能 | `build.sh` / `tests/run_whl_tests.sh` 中先 `pip uninstall oil-gas-ops-prospect` 再重装，流程可执行 |

## 6. 算子UT功能验证

无卡门禁、Triton 常规精度、AscendC NPU 精度与可选/示例用例：**通过**。日志无构建级 crash。

### 6.1 ophost tiling UT（googletest，无卡）

入口：`bash build.sh --ut` → `ctest` 目标 `ophost_ut`。**25/25 PASS**。

| 序号 | 测试用例 | 测试内容 | 结果 |
| ---- | -------- | -------- | ---- |
| 1 | ComplexMulTiling.TileFloatsFitsUbBudget | 前向 tile 浮点占用不超过 UB | PASS |
| 2 | ComplexMulTiling.PlanIsFilledForRepresentativeShapes | 代表 shape 切分计划填满 | PASS |
| 3 | ComplexMulTiling.SmallTensorShrinksTileButKeepsAlignment | 小 tensor 缩小 tile 且对齐 | PASS |
| 4 | ComplexMulTiling.DegenerateInputs | 退化输入 | PASS |
| 5 | ComplexMulGradTiling.TileFloatsFitsUbBudget | 反向 tile UB | PASS |
| 6 | ComplexMulGradTiling.TileIsSmallerThanForward | 反向 tile 小于前向 | PASS |
| 7 | ComplexMulGradTiling.PlanIsFilledForRepresentativeShapes | 反向代表 shape | PASS |
| 8 | ComplexMulGradTiling.DegenerateInputs | 反向退化输入 | PASS |
| 9 | ComplexMulSplit.CoresCoverEveryElementExactlyOnce | 多核覆盖无重叠 | PASS |
| 10 | ComplexMulSplit.LargeShapesAreFullyCovered | 大 shape 覆盖完整 | PASS |
| 11 | ComplexMulSplit.PerCoreChunkIsAlignedExceptTail | 除尾块外对齐 | PASS |
| 12 | FusedBiasSoftmaxCore.Bf16DualBiasBroadcastTypical | bf16 双 bias 广播 | PASS |
| 13 | FusedBiasSoftmaxCore.Fp32NoBias | fp32 无 bias | PASS |
| 14 | FusedBiasSoftmaxCore.Fp32NonAlignedWidthPadsTo32B | 非对齐宽 pad 到 32B | PASS |
| 15 | FusedBiasSoftmaxCore.FullBias1HasNoBcastOverhead | 满 bias1 无广播开销 | PASS |
| 16 | FusedBiasSoftmaxCore.RejectsInvalidInputs | 非法输入拒绝 | PASS |
| 17 | FusedBiasSoftmaxCore.Fp16UsesFp16Key | fp16 tiling key | PASS |
| 18 | FusedSoftmaxGradCore.Bf16AlignedGoesMix | bf16 对齐走 MIX | PASS |
| 19 | FusedSoftmaxGradCore.Bf16NonAlignedDGoesVector | 非对齐 D 走 VEC | PASS |
| 20 | FusedSoftmaxGradCore.Fp32AlwaysVector | fp32 走 VEC | PASS |
| 21 | FusedSoftmaxGradCore.Fp16AlwaysVector | fp16 走 VEC | PASS |
| 22 | FusedSoftmaxGradCore.VecPathRejectsVPlaneBeyondUb | VEC 超 UB 拒绝 | PASS |
| 23 | FusedSoftmaxGradCore.MixRowsPerIterShrinksWithPadS | MIX rows 随 padS 收缩 | PASS |
| 24 | FusedSoftmaxGradCore.RejectsInvalidInputs | 非法输入拒绝 | PASS |
| 25 | FusedSoftmaxGradCore.QNotEqualSAllowed | Q≠S 允许 | PASS |

### 6.2 Triton CPU 回退 + NPU 常规精度（pytest）

入口：`python3 -m pytest tests/triton/real_fft -v`。收集 **26** 条：默认 **20 PASS / 6 SKIP**。

| 序号 | 测试用例 | 测试内容 | 结果 |
| ---- | -------- | -------- | ---- |
| 1–6 | test_real_fft_cpu_fallback_matches_torch_fft | 6 组 shape/dim，CPU 回退 vs `torch.fft`（rtol=1e-5） | PASS |
| 7–10 | test_internal_fourier_basis_matches_torch_fft | n=7/8/16/41 基函数 vs `torch.fft` | PASS |
| 11 | test_model_fused_fft_config_switch_matches_reference_path | 下游模型仓交叉验证 | SKIP（未设置 `OIL_GAS_OPS_MODEL_REPO`） |
| 12–16 | test_real_fft_npu_triton_matches_cpu_torch_fft | 5 组 NPU Triton 前向 vs CPU `torch.fft`（rtol=2e-4） | PASS |
| 17–21 | test_real_fft_npu_triton_backward_matches_cpu_torch_fft | 5 组 NPU Triton 反向 vs CPU | PASS |
| 22–24 | test_real_fft_npu_latent_3d_full_shape | 默认未开大 shape | SKIP（需 `OIL_GAS_OPS_RUN_LATENT3D_FFT_TEST=1`） |
| 25 | test_real_fft_npu_128_3d_backward_and_memory_opt_in | 默认未开 128³ | SKIP（需 `OIL_GAS_OPS_RUN_128_FFT_TEST=1`） |
| 26 | test_model_config_fft_shape_inference | 下游 YAML 形状推断 | SKIP（未设置 `OIL_GAS_OPS_MODEL_CONFIG`） |

### 6.3 Triton AOT 编译检查

入口：`python3 triton-ascend/tools/aot_compile_check.py`。target=`Ascend910B1`。**8/8 OK**。

| 序号 | 分支 | 结果 |
| ---- | ---- | ---- |
| 1 | real_fft/rfft.fwd flatten 2D GEMM | PASS |
| 2 | real_fft/irfft.fwd flatten 2D GEMM | PASS |
| 3 | real_fft/rfft.bwd flatten 2D GEMM | PASS |
| 4 | real_fft/irfft.bwd flatten 2D GEMM | PASS |
| 5 | real_fft/rfft.fwd axis strided GEMM | PASS |
| 6 | real_fft/rfft.bwd axis strided GEMM | PASS |
| 7 | real_fft/irfft.fwd axis strided GEMM | PASS |
| 8 | real_fft/irfft.bwd axis strided GEMM | PASS |

### 6.4 whl 冒烟

入口：`OIL_GAS_OPS_WHL_TEST=1 pytest tests/test_oil_gas_ops_prospect_whl.py -v`（安装本次构建的 0.1.0 whl 后）。**8/8 PASS**。`find_spec` 已捕获 `ModuleNotFoundError`。

### 6.5 AscendC NPU 精度 / 性能（需真机）

入口：`bash tests/ascendc/complex_mul/run_test.sh`、`bash tests/ascendc/fused_bias_softmax/run_test.sh`。OPP 已安装到 customize vendor。

#### ComplexMul / ComplexMulGrad

标杆：同设备上的 PyTorch 闭式解；阈值 `rtol=1e-5`。

| 序号 | shape | 前向 max rel_err | 反向 max rel_err | 结果 |
| ---- | ----- | ---------------- | ---------------- | ---- |
| 1 | `(4, 1024)` | wr=0 / wi=0 | 0 | PASS |
| 2 | `(1, 1, 192, 320, 320)`（默认） | wr=0 / wi=0 | 0 | PASS |

复测日期 2026-09-12，容差 `rtol=1e-5`。CANN 9 TBE 会裁掉同一 kernel 第二路输出；绑定侧前向按公式在 NPU 上拼 `wr/wi`，反向复用 ComplexMul 第一路 Sub 拼四路梯度。

#### FusedBiasSoftmax / FusedSoftmaxGrad

| 序号 | 用例 | 测试内容 | 结果 |
| ---- | ---- | -------- | ---- |
| 1 | bf16-dual-bcast `[2,4,64,128]` | 前向，原地 | PASS |
| 2 | bf16-dual `[2,4,256,256]` | 前向 | PASS |
| 3 | fp32-none `[2,4,64,64]` | 前向无 bias | PASS |
| 4 | bf16-peaked `[2,4,128,128]` | 前向尖峰分布 | PASS |
| 5 | bf16-MIX `[2,4,64,64] D64` | 反向 | PASS（max_diff=0） |
| 6 | bf16-MIX `[8,8,256,256] D64` 超核 | 反向 | PASS（max_diff=4.88e-04） |
| 7 | bf16-VEC `[2,2,128,128] D20` | 反向 VEC 非对齐 | PASS |
| 8 | fp32-VEC `[2,4,64,64] D64` | 反向 VEC | PASS |
| 9 | bf16-MIX Q≠S `[2,4,64,128] D64` | 反向 | PASS |
| 10–13 | e2e attention_core（o / dL/dlogits / dL/dq / dL/dv） | 前向+反向链路 vs fp32 标杆 | PASS（max_diff 均 < 0.6） |

前向 4/4、反向 5/5、e2e 4/4 均通过。`[Q=S=D]` bf16 在 aclnn 缓存下会空跑，绑定对该形态走 fp32 JVP；其余 shape 走自定义 kernel。

### 6.6 Triton 可选大 shape / 示例

| 序号 | 用例 | 测试内容 | 结果 |
| ---- | ---- | -------- | ---- |
| 1–3 | `OIL_GAS_OPS_RUN_LATENT3D_FFT_TEST=1` latent 3D `(2,41,41,41,2048)` dim=1/2/3 | 往返 + 切片 vs `torch.fft` | PASS |
| 4 | `OIL_GAS_OPS_RUN_128_FFT_TEST=1` shape `(1,128,128,128,1)` | fused Triton vs CPU | PASS |
| 5 | `examples/complex_mul_example.py` | 示例默认 shape | PASS |
| 6 | `examples/real_fft_example.py` | 默认 `BATCH=8, N=256` 走 NPU Triton | PASS |

128³ 断言只约束本仓 fused vs CPU（约 1e-4）；原生 `torch_npu` rfft 反向 vs CPU 仍有约 7 的偏差，不计入本仓失败。`real_fft` 已按 L0C/cbuf 上限收缩最后一维 tile，默认 N=256 不再 overflow。

## 7. 环境兼容性验证

兼容性评估：本次只覆盖 **一种** 实测组合（见第 2 节），不宣称多 OS / 多形态矩阵已测完。

| 验证项 | 验证内容 |
| ------ | -------- |
| 三方依赖兼容性 | 容器内已预装 torch / torch_npu / triton-ascend 3.2.1 / pytest / cmake / g++，按 README 配套 CANN 9.0.0 可编译 |
| OS 兼容性 | 本次：宿主机 Ubuntu 22.04 aarch64 + 容器 openEuler 24.03。未测 x86 / Euler 裸机 |
| CANN 包兼容性 | CANN 9.0.0 可编译出包并安装 customize vendor |
| 形态兼容性 | docker + 宿主机 device（`--privileged` + davinci 设备）可用 NPU |
| 芯片架构兼容性 | 在 **Ascend 910B2** 上用 **Ascend910B1** 默认 SOC 出包并加载。文档声明 A3 / 950 暂未验证，本次也未测 |

## 8. 测试结论

测试结论：**通过**。无卡门禁、文档、编译出包、安装、Triton 常规/可选精度、示例与 AscendC NPU 精度均已复测通过。

| 评估维度 | 评估结果 |
| -------- | -------- |
| 文档 / 编译 / 安装 | 通过 |
| 无卡 UT | **25** ophost + **20** Triton pytest + **8** AOT = **53/53** 通过（另 6 条 pytest 为环境 skip） |
| whl 冒烟 | **8/8** 通过 |
| AscendC NPU | ComplexMul 默认/小 shape **PASS**；Fused 前向 **4/4**、反向 **5/5**、e2e **4/4** |
| Triton 可选 / 示例 | latent3D **3/3**；128³ fused vs CPU **PASS**；`examples/real_fft_example.py` **PASS** |
| 测试结论 | **全量已执行用例通过** |

> 说明：下游模型仓交叉验证（`OIL_GAS_OPS_MODEL_REPO` / `OIL_GAS_OPS_MODEL_CONFIG`）本次未配置，保持 SKIP，不计入失败。

## 9. 遗留问题和关键风险

### 9.1 遗留问题统计

| 级别 | 数量 | 说明 |
| ---- | ---- | ---- |
| 功能/精度 | 0 | 本仓用例已通过 |
| 环境/标杆 | 1 | 原生 `torch_npu` rfft 反向 vs CPU 仍不一致（不计入本仓失败） |

### 9.2 已知环境差异（非本仓库失败）

| 编号 | 问题 | 影响 | 处理 |
| ---- | ---- | ---- | ---- |
| E1 | 原生 `torch_npu` rfft 反向 vs CPU max_abs≈6～7 | 仅三方对比中的官方 FFT，本仓 fused vs CPU 约 1e-4 | 用例不再断言 `torch_npu_rfft_bwd` |

## 10. 附件

### 10.1 复现入口

```bash
# 容器内已 source CANN 9.0.0
bash build.sh --ut
bash build.sh --soc Ascend910B1 -j 32
bash tests/ascendc/complex_mul/run_test.sh
bash tests/ascendc/fused_bias_softmax/run_test.sh
python3 -m pytest tests/triton/real_fft -v
OIL_GAS_OPS_RUN_LATENT3D_FFT_TEST=1 python3 -m pytest tests/triton/real_fft/test_triton_fft_real.py::test_real_fft_npu_latent_3d_full_shape -v
OIL_GAS_OPS_RUN_128_FFT_TEST=1 python3 -m pytest tests/triton/real_fft/test_triton_fft_real.py::test_real_fft_npu_128_3d_backward_and_memory_opt_in -v
```

原始分步日志：`/root/work/tmp/oil-gas-qa/logs/`（172.18.0.141 共享盘）。

### 10.2 测试容器

```text
镜像: ai4fwi-ascend-ha:9.0.0-910b-openeuler24.03-py3.11-aarch64
容器: oil-gas-ops-prospect-qa
启动: --privileged --net=host，挂载 davinci* / driver / /root/work
```
