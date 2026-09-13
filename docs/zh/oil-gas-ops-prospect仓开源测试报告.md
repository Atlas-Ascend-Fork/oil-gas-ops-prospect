# oil-gas-ops-prospect仓开源测试报告

## 1. 概述

本报告为 oil-gas-ops-prospect 项目开源测试报告。测试基于 CANN 9.0.0 开源包，覆盖仓内已交付算子的文档、编译出包、安装卸载、功能、精度与性能，以及配套环境兼容性。测试过程中对已纳入范围的特性执行覆盖率为 100%。

本仓面向油气勘探计算负载，按算子名对外提供 Python / aclnn 接口，不绑定单一上层模型。

### 新增特性清单

| 序号 | 特性名称 | 特性描述 |
| --- | --- | --- |
| 1 | ComplexMul | 复数逐点乘：`(ur, ui), (vr, vi) → (wr, wi)`，`float32` / `ND`，全部张量同 shape |
| 2 | ComplexMulGrad | `ComplexMul` 的反向：由 `gwr, gwi` 与前向输入得到 `gur, gui, gvr, gvi`，`float32` / `ND` |
| 3 | FusedBiasSoftmax | `y = softmax(x + bias1 + bias2, -1)`，`bias1`/`bias2` 可选，NPU 路径原地写回 `x`，支持 `fp32` / `fp16` / `bf16` |
| 4 | FusedSoftmaxGrad | 注意力 softmax JVP：`g = go @ V^T` 后原地覆写 `probs`，支持 `Q≠S`，`fp32` / `fp16` / `bf16` |
| 5 | real_rfft | 实数正向 FFT：`x → (xr, xi)`，`norm="ortho"`，频率轴长度为 `n//2+1` |
| 6 | real_irfft | 由半谱实部、虚部重建实数：`(xr, xi), n → y`，`norm="ortho"` |

`attention_core(q, k, v, bias1, bias2)` 为 FusedBiasSoftmax 与 FusedSoftmaxGrad 按 openfold 注意力语义的组合封装，随上述融合算子一并验证。

### 解决问题列表

不涉及（首个开源测试版本，无历史问题清单）。

### 测试活动

本次测试活动包括：Readme 与算子文档评估、一键编译出包、算子包与 Python 包安装卸载、按算子的功能/精度/性能验证、可靠性与兼容性验证。

## 2. 版本测试信息

本节描述被测对象的版本信息和测试时间，包括依赖环境与测试源码仓。

**硬件和版本要求**

产品型号：Atlas A2 / Ascend 910B（文档默认 `SOC_VERSION=Ascend910B1`）

操作系统：覆盖以下测试因子组合

| | 因子名称 | 因子取值 |
| --- | --- | --- |
| 1 | OS | Ubuntu / openEuler |
| 2 | CPU 架构 | ARM64 |
| 3 | 形态 | docker / host+device |

CANN 版本：CANN 9.0.0

驱动版本：Ascend HDK 26.0.RC1

Python 版本：Python >= 3.10（本次实测 3.11）

PyTorch 版本：2.7.1（配套 torch_npu 2.7.1）

依赖三方库版本：与 CANN 9.0.0 配套的 torch_npu、Triton-Ascend 3.2.1；构建依赖见仓库 `requirements.txt`

测试源码仓：https://gitcode.com/cann/oil-gas-ops-prospect

测试日期：2026-09-12

## 3. 测试结论

给出明确的版本发布结论。

oil-gas-ops-prospect 基于 CANN 9.0.0 完成开源测试，共计执行 18 组特性级测试用例，发现 0 个问题，其中有效问题 0 个。整体质量良好，满足出口质量标准。**建议发布。**

| 评估维度 | 评估结果 |
| --- | --- |
| 测试用例总数 | 18 组（按已交付算子与关键测试活动归并，不含参数化子项展开） |
| 测试执行通过率 | 100%（18/18） |
| 测试结论 | 通过 |

## 4. 特性质量评估

新特性的验证结果清单如下。功能、精度在 CPU 参考路径与 NPU 路径上一并验证（非 NPU 环境走 PyTorch 回退，NPU 环境与同设备标杆对拍）。

| 序号 | 特性 | 测试结论 | 功能 | 精度 | 性能 | 可靠性 | 兼容性 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ComplexMul | 通过 | Pass | Pass | Pass | Pass | Pass |
| 2 | ComplexMulGrad | 通过 | Pass | Pass | Pass | Pass | Pass |
| 3 | FusedBiasSoftmax | 通过 | Pass | Pass | Pass | Pass | Pass |
| 4 | FusedSoftmaxGrad | 通过 | Pass | Pass | Pass | Pass | Pass |
| 5 | real_rfft | 通过 | Pass | Pass | Pass | Pass | Pass |
| 6 | real_irfft | 通过 | Pass | Pass | Pass | Pass | Pass |

特性级用例（按特性介绍组织，同一用例同时覆盖非 NPU 回退与 NPU 对拍，不按实现后端拆分）：

| 序号 | 测试用例 | 测试内容 | 结果 |
| --- | --- | --- | --- |
| 1 | ComplexMul 切分与约束 | 代表 shape / 小 tensor / 退化输入的切分计划，核间覆盖与对齐，UB 预算 | PASS |
| 2 | ComplexMul 前反向精度 | 闭式解对拍（含 `whl` 接口与 NPU 执行）：`(4,1024)`、`(1,1,192,320,320)` 等，相对误差阈值 `1e-5` | PASS |
| 3 | ComplexMulGrad 切分与梯度 | 反向 tile 小于前向、代表 shape 填满、退化输入；梯度与 autograd / 闭式解一致 | PASS |
| 4 | FusedBiasSoftmax 切分与约束 | 双 bias 广播、无 bias、非对齐宽、满尺寸 bias1、非法输入拒绝、dtype 对应 tiling key | PASS |
| 5 | FusedBiasSoftmax 前向精度 | `softmax(x+bias1+bias2,-1)`：bf16 广播/全尺寸/尖峰分布、fp32 无 bias；原地语义；同 shape 不同缓冲互不污染；与 fp32 标杆对拍 | PASS |
| 6 | FusedSoftmaxGrad 切分与路径 | 对齐 MIX、非对齐 VEC、fp32/fp16 走向量、超 UB 拒绝、`Q≠S` 合法、非法输入拒绝 | PASS |
| 7 | FusedSoftmaxGrad 反向精度 | `g=go@V^T` 后 JVP 原地覆写：典型 / 超核 / 非对齐 D / fp32 / `Q≠S`；与 fp32 标杆对拍 | PASS |
| 8 | 融合注意力链路 | `attention_core` 前向输出与 `dL/dlogits`、`dL/dq`、`dL/dv` 对拍，容差 `max_diff<0.6` | PASS |
| 9 | real_rfft 基函数与前向 | Fourier 基与 `torch.fft` 一致；多 shape/dim 前向半谱对拍（含回退路径与 NPU） | PASS |
| 10 | real_irfft 重建与往返 | 半谱重建实信号，往返误差与 `torch.fft.irfft(..., norm="ortho")` 对齐 | PASS |
| 11 | real_rfft / real_irfft 反向 | 多 shape/dim 反向与 CPU `torch.fft` 对拍 | PASS |
| 12 | real_rfft / real_irfft 大形态 | 三维潜变量形态与 `128^3` 往返/反向；默认示例可运行 | PASS |
| 13 | 算子文档 | 算子清单、数学定义、约束、接口列表与实现一致；快速入门与开发指南可跟随 | PASS |
| 14 | 编译出包 | `bash build.sh --help` 合法选项可打印；`--soc` / `--ops` / `-j` 可组合；产出 OPP `.run` 与 Python whl | PASS |
| 15 | 安装与卸载 | OPP 写入 customize vendor；`pip install` / `pip uninstall oil-gas-ops-prospect` 可执行 | PASS |
| 16 | 公开接口冒烟 | `complex_mul` / `fused_bias_softmax` / `fused_softmax_grad` / `real_rfft` / `real_irfft` 可导入；CPU 回退前反向可跑通 | PASS |
| 17 | 可靠性 | 非法输入被切分侧拒绝；融合算子保持原地写回；非 NPU 自动回退且接口语义不变 | PASS |
| 18 | 配套兼容 | CANN 9.0.0 + 配套 PyTorch / torch_npu 可完成编译、安装与算子验证 | PASS |

## 5. DFX专项质量评估

### 5.1 安全测试

版本整体安全测试结论：满足开源发布对安全声明与许可证的要求。

仓内提供 `SECURITY.md`、Apache 2.0 `LICENSE`；构建入口对非法参数拦截并提示。本次未开展独立漏洞扫描、fuzz 与交付件病毒扫描。

| 序号 | 测试项 | 测试结论 | 遗留风险 |
| --- | --- | --- | --- |
| 1 | 安全声明与许可证 | 通过 | 无 |
| 2 | 构建参数校验 | 通过 | 无 |

### 5.2 可靠性测试

可靠性指标的验证达成情况如下。

| 序号 | 可靠性特性 | 测试结论 | 遗留风险 |
| --- | --- | --- | --- |
| 1 | 非法或退化输入被拒绝或收缩为合法切分 | 通过 | 无 |
| 2 | 融合算子原地写回（输出与输入共享 device 指针） | 通过 | 无 |
| 3 | 非 NPU 或扩展不可用时回退 PyTorch，接口不变 | 通过 | 无 |
| 4 | 同 shape 重复调用精度稳定（含不同缓冲、融合注意力链路） | 通过 | 无 |

### 5.3 性能测试

该版本的性能按算子微基准测量：预热 5 次、计时 20 次取均值，加速比 = 同设备 Torch 小算子时延 / 本算子时延。测试环境为 Atlas A2 / CANN 9.0.0，不绑定具体主机。四类算子均于 2026-09-13 对同设备 Torch 标杆复测。

| 场景 | 模型 | 特性 | Torch 小算子标杆 | 性能指标 | 测试环境 | 测试结果 | 遗留风险 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 算子微基准 | 不涉及 | ComplexMul / ComplexMulGrad | 同设备逐元素 `wr=ur*vr-ui*vi` 等 | 前向约 2.7x～2.8x，反向约 3.5x～5.5x | Atlas A2 + CANN 9.0.0 | Pass | 无 |
| 算子微基准 | 不涉及 | FusedBiasSoftmax | 同设备 `softmax(x+bias1+bias2,-1)`（fp32 累加） | bf16 约 2.7x～2.9x，小 fp32 `[2,4,64,64]` 约 1.66x | Atlas A2 + CANN 9.0.0 | Pass | 无 |
| 算子微基准 | 不涉及 | FusedSoftmaxGrad | 同设备 `g=go@V^T` + softmax JVP | bf16 约 1.98x～2.04x，fp32 约 1.14x | Atlas A2 + CANN 9.0.0 | Pass | 无 |
| 算子微基准 | 不涉及 | real_rfft / real_irfft | 同设备 `torch.fft.rfft` / `irfft`，`norm="ortho"` | 常规末轴约 1.1x～2.0x；业务 5D W/H/D 前向约 3.8x / 16.6x / 14.6x，irfft 约 4.1x / 18.2x / 14.5x；fwd+bwd 约 9.4x / 32x / 29x | Atlas A2 + CANN 9.0.0 | Pass | 无 |

代表形态实测如下（精度全部 PASS）。

**ComplexMul / ComplexMulGrad（相对同设备逐元素公式）**

| 用例 | 标杆 ms | 本算子 ms | 加速比 |
| --- | ---: | ---: | ---: |
| fwd `[4,1024]` | 0.270 | 0.102 | 2.65x |
| bwd `[4,1024]` | 0.652 | 0.119 | 5.49x |
| fwd `[1,1,32,32,32]` | 0.282 | 0.103 | 2.75x |
| bwd `[1,1,32,32,32]` | 0.644 | 0.126 | 5.13x |
| fwd `[1,1,96,160,160]` | 0.288 | 0.107 | 2.69x |
| bwd `[1,1,96,160,160]` | 0.652 | 0.126 | 5.17x |
| fwd `[1,1,192,320,320]` | 1.162 | 0.412 | 2.82x |
| bwd `[1,1,192,320,320]` | 2.478 | 0.709 | 3.50x |

**FusedBiasSoftmax（相对同设备 fp32 `softmax`）**

| 用例 | 标杆 ms | 本算子 ms | 加速比 |
| --- | ---: | ---: | ---: |
| fwd `[2,4,64,128]` bf16 | 0.529 | 0.183 | 2.89x |
| fwd `[2,4,256,256]` bf16 | 0.529 | 0.198 | 2.67x |
| fwd `[2,4,64,64]` bf16 | 0.240 | 0.085 | 2.83x |
| fwd `[2,4,64,64]` fp32 | 0.145 | 0.087 | 1.66x |
| fwd `[8,8,256,256]` bf16 | 0.244 | 0.091 | 2.68x |

**FusedSoftmaxGrad（相对同设备 `matmul` + softmax JVP）**

| 用例 | 标杆 ms | 本算子 ms | 加速比 |
| --- | ---: | ---: | ---: |
| bwd `[2,4,64,64]` D64 bf16 | 0.632 | 0.310 | 2.04x |
| bwd `[8,8,256,256]` D64 bf16 | 0.650 | 0.328 | 1.98x |
| bwd `[2,2,128,128]` D20 bf16 | 0.616 | 0.312 | 1.98x |
| bwd `[2,4,64,128]` D64 bf16 | 0.613 | 0.306 | 2.00x |
| bwd `[2,4,64,64]` D64 fp32 | 0.388 | 0.342 | 1.14x |

**real_rfft / real_irfft 常规形态（相对同设备 `torch.fft`）**

相对误差约 `2e-7`～`5e-7`。含业务 5D 的全表几何平均加速比约 **2.71x**。业务 5D 相对标杆少约 1.1～1.7 GB 峰值显存。

| 用例 | 标杆 ms | 本算子 ms | 加速比 |
| --- | ---: | ---: | ---: |
| rfft `[8,256]` 末轴 | 0.401 | 0.275 | 1.46x |
| irfft `[8,256]` 末轴 | 0.554 | 0.280 | 1.98x |
| rfft `[32,512]` 末轴 | 0.406 | 0.260 | 1.56x |
| irfft `[32,512]` 末轴 | 0.546 | 0.272 | 2.01x |
| rfft `128³` 末轴 | 0.300 | 0.262 | 1.14x |
| irfft `128³` 末轴 | 0.488 | 0.283 | 1.73x |
| rfft `128³` 非末轴 | 0.270 | 0.271 | 1.00x |
| irfft `128³` 非末轴 | 0.478 | 0.284 | 1.68x |

**real_rfft / real_irfft 业务 5D `[1,35,128,128,128]`（相对同设备 `torch.fft`）**

| 轴 | rfft | irfft | fwd+bwd |
| --- | ---: | ---: | ---: |
| W 末轴 | 3.81x（7.964 / 2.089 ms） | 4.11x（10.573 / 2.570 ms） | 9.39x（45.875 / 4.886 ms） |
| H | 16.59x（8.501 / 0.512 ms） | 18.20x（10.517 / 0.578 ms） | 32.23x（49.057 / 1.522 ms） |
| D | 14.59x（7.908 / 0.542 ms） | 14.53x（10.646 / 0.733 ms） | 29.15x（48.925 / 1.678 ms） |

### 5.4 兼容性测试

兼容性评估：通过。

| 序号 | 兼容性场景 | 验证结果 | 遗留风险 |
| --- | --- | --- | --- |
| 1 | CANN 9.0.0 配套（Toolkit + 910B Ops） | pass | 无 |
| 2 | Python 3.11 + PyTorch 2.7.1 + 配套 torch_npu | pass | 无 |
| 3 | `requirements.txt` 声明的构建与测试依赖 | pass | 无 |
| 4 | docker 与 host+device 形态下编译、安装、跑通算子 | pass | 无 |
| 5 | 文档声明的 Atlas A2 / Ascend 910B | pass | Atlas A3 / 950 系列文档标明暂未验证 |

## 6. 测试执行评估

### 6.1 测试覆盖

特性以及关键测试活动的测试覆盖情况如下。

| 测试活动 | 测试结论 | 用例数 | 用例覆盖率 | 用例通过率 |
| --- | --- | --- | --- | --- |
| 特性测试 | pass | 16 | 100% | 100% |
| 性能测试 | pass | 4 | 100% | 100% |
| 可靠性测试 | pass | 4 | 100% | 100% |
| 继承特性测试 | 不涉及 | 0 | — | — |
| 兼容性测试 | pass | 5 | 100% | 100% |
| 安全测试 | pass | 2 | 100% | 100% |

说明：特性测试第 1～16 组覆盖文档、编译安装与全部已交付算子；性能测试、可靠性、兼容性、安全与第 4、5 节表格对应。首发版本无继承特性。下游模型仓交叉验证未配置，不纳入本次范围，不计入失败。

## 7. 遗留问题和关键风险

不涉及本仓功能缺陷。

### 7.1 遗留问题统计

| | 问题总数 | 严重 | 主要 | 次要 | 不重要 | 已取消 |
| --- | --- | --- | --- | --- | --- | --- |
| 数目 | 0 | 0 | 0 | 0 | 0 | 0 |
| 百分比 | 100% | 0% | 0% | 0% | 0% | 0% |

### 7.2 遗留问题列表

| 问题单(issue链接) | 问题描述 | 问题级别 | 问题影响和规避措施 | 当前状态 |
| --- | --- | --- | --- | --- |
| 不涉及 | 不涉及 | — | — | — |

已知环境差异（非本仓库失败，不计入上表）：原生 `torch.fft` 在部分 NPU 官方实现上的反向与 CPU 不完全一致；本仓 `real_rfft` / `real_irfft` 以 CPU `torch.fft` 为标杆验收。官方用例门槛 `1e-5`；相位折叠造基后实测相对误差约 `2e-7`～`5e-7`。

## 8. 附件
不涉及
