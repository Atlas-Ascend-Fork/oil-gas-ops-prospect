# 贡献指南

本项目欢迎广大开发者体验并参与贡献。在参与社区贡献之前，请先阅读 [cann-community](https://gitcode.com/cann/community) 了解行为准则、完成 CLA 协议签署，并了解源码仓的贡献流程。

本仓归属 [oil-gas-engineering SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/oil-gas-engineering)，聚焦油气勘探（地震成像、全波形反演等）计算负载的昇腾原生算子。

开发者准备本地代码与提交 PR 时需重点关注：

1. 提交 PR 时，请按 PR 模板填写本次 PR 的业务背景、目的、方案。
2. 若您的修改不是简单的 bug 修复，而涉及新增算子、新增接口、修改调用约定或改动构建流程，请务必先通过 Issue 讨论方案，以免代码被拒绝合入。若不确定是否属于"简单的 bug 修复"，同样建议先提 Issue。

---

## 一、贡献新算子

本仓有两类算子后端，交付件不同，请按实际后端选择。

### 1. 创建 Issue 需求

新建 `Requirement|需求建议` 类 Issue，阐明新增算子的设计方案，一般包含：

- **背景信息**：来自哪个勘探业务场景（叠前偏移、RTM、FWI 等），当前用什么实现、瓶颈在哪。
- **价值/作用**：预期加速比、显存收益或精度收益。
- **设计方案**：算子数学定义、输入输出 shape/dtype/format、切分策略、目标 SOC。

请在 Issue 中评论 `/assign @yourself` 认领该任务。

### 2. 需求评审

SIG 组将指派 Committer 对 Issue 进行评审并反馈修改意见。请在完成修改后，于 Issue 中 @ 对应 Committer。

### 3. 代码开发与交付件

#### AscendC 算子（OPP `.run` 包）

提交至 `ascendc/operators/<op_name>/`，交付件如下：

```text
ascendc/operators/${op_name}/
├── op_host/
│   ├── ${op_name}.cpp                  # 算子原型定义（OpDef）、InferShape、TilingFunc
│   ├── ${op_name}_tiling_data_def.h    # TilingData 结构体定义
│   ├── ${op_name}_tiling_core.h        # 平台无关的切分算法（header-only，供 UT 直接驱动）
│   └── ${op_name}_tiling_impl.h        # 对接 gert::TilingContext 的实现
└── op_kernel/
    ├── ${op_name}.cpp                  # kernel 入口
    ├── ${op_name}_impl.h               # kernel 实现
    └── ${op_name}_tiling.h             # kernel 侧 TilingData 视图

ascendc/operators/manifest.tsv          # 追加一行（tab 分隔）注册算子
ascendc/pybind/${op_name}.py            # Python API（autograd + 回退）
ascendc/pybind/csrc/${op_name}/bindings.cpp  # C++ 绑定（打进 whl，非测试资产）
tests/ut/op_host/${op_name}/                   # tiling UT（自动汇入 ophost_ut）
tests/st/aclnn<OpType>/                        # 多 case 两段式 aclnn + 真值
examples/aclnn/${op_name}/                     # 两段式 aclnn 小样例
tests/ascendc/${op_name}/                      # NPU Python 精度/性能对比
```

C++ 测试与样例放在仓库根 `tests/`、`examples/`；`operators/` 只放实现。

约定：

- **切分算法必须抽到 `op_host/<op>_tiling_core.h`**。`<Op>TilingFunc` / `InferShape` 的签名依赖 `gert::TilingContext`，CANN 二进制包未提供 gert faker，无法直接 UT；把与平台无关的算法抽出来后，ophost UT 才能覆盖真实算法。
- **C++ / CANN 接口**：OpType 用 PascalCase（如 `ComplexMul`），aclnn 为 `aclnn` + OpType（如 `aclnnComplexMul`）。
- **Python 接口**：蛇形命名（如 `complex_mul`），不要用驼峰，也不要加 `npu_` 前缀。
- 新增算子后，在 `tests/ut/op_host/<op>/` 放下 `test_*.cpp`；`tests/ascendc/ophost/CMakeLists.txt` 会自动收集。
- 有卡 aclnn：在 `examples/aclnn/<op>/` 放 `test_aclnn_<op>.cpp`（内联小 tensor），在 `tests/st/aclnn<OpType>/` 放 `cases.json` + `gen_data.py` + 两段式 `main.cpp`，并登记 `tests/suites.json`。本仓没有 ATK / `OP_API_UT`，不要空放 `ut/op_api`。
- pybind 封装：Python API 放在 `ascendc/pybind/<op_family>.py`，C++ 源码放在 `ascendc/pybind/csrc/<op_family>/`，会自动编译成扩展模块 `<op_family>_npu`，不需要改构建脚本。

#### Triton-Ascend 算子（Python whl）

提交至 `triton-ascend/<op_family>/`，交付件如下：

```text
triton-ascend/${op_family}/
├── __init__.py                         # 对外导出
├── triton_${op_family}.py              # Triton kernel 与 autograd 封装
└── README.md                           # 算子说明

tests/triton/${op_family}/
└── test_triton_${op_family}.py         # CPU 回退路径精度测试
```

同时需在 `setup.py` 的 `packages` / `package_dir` 中按算子名注册，使其进入 whl
（`oil_gas_ops_prospect.${op_family}`，不要加 `triton` / `ascendc` 二级包）。

约定：

- kernel 需提供 **PyTorch 回退实现**，接口语义与 NPU 路径一致。
- 新增 kernel 分支需能被 `triton-ascend/tools/aot_compile_check.py` 覆盖到。

### 4. 精度要求

新贡献算子需满足精度标准，具体参见[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)。目录、命名、切分可测与 PR 交付件见[算子开发指南](docs/zh/develop/operator_development_guide.md)。请在 PR 描述中附上精度测试报告，报告格式参考 [`docs/zh/develop/precision_acceptance_template.md`](docs/zh/develop/precision_acceptance_template.md)。

> **算子在 NPU 上的精度、性能、显存必须在真实 910B 上验证通过。**

### 5. 提交前本地自检

```bash
bash build.sh --install
bash build.sh -u --ut --st --precision --perf
bash scripts/check_license_header.sh
```

### 6. PR 提交

- **合规检查**
  - C++ 代码符合《[C++ 编程规范](https://gitcode.com/cann/community/blob/master/contributor/coding-standards/C++%20Coding%20standards.md)》，可用仓内 `.clang-format` 格式化。
  - 所有新增源文件带 Apache License 2.0 许可证头（参考现有文件）。
  - Markdown 文档语法符合规范。
  - 本地自检全部通过（含 910B 上板）。
- **PR 提交**：通过 `git` 命令向目标分支提 PR，检查 PR 标题是否清晰、PR 描述是否规范（说明更改内容和原因、关联对应 Issue）、是否已签署 CLA。

### 7. CI 门禁

通过评论 `compile` 指令触发开源仓门禁，并依据 CI 结果修改。当前门禁包含：

- 代码编译
- 静态检查（如涉及 codecheck 误报，请提交给 SIG 成员屏蔽）
- UT 测试

门禁通过后，请在关联的 Issue 中 @ 指派的 Committer。

### 8. 代码检视与合入

合入前需至少两名 Committer 完成代码检视，并分别在 PR 中评论 `/lgtm` 与 `/approve`。检视意见请据此修改，完成后再次 @ 检视人。

---

## 二、算子 Bug 修复

如果您在本项目中发现了算子 Bug，希望对其进行修复，欢迎新建 Issue 反馈和跟踪。

请按[提交 Issue/处理 Issue 任务](https://gitcode.com/cann/community#%E6%8F%90%E4%BA%A4Issue%E5%A4%84%E7%90%86Issue%E4%BB%BB%E5%8A%A1)指引新建 `Bug-Report|缺陷反馈` 类 Issue 描述问题，然后在评论框中输入 `/assign` 或 `/assign @yourself` 认领处理。

Bug 类 Issue 建议附带：

- 复现用的 shape、dtype 与调用方式
- CANN 版本、SOC 型号、`torch` / `torch_npu` 版本
- 期望结果与实际结果（含误差量级）

---

## 三、算子优化

如果您对本项目中某些算子实现有泛化性增强或性能优化思路，欢迎贡献优化。

请新建 `Requirement|需求建议` 类 Issue 说明优化点并提供设计方案，然后在评论框中输入 `/assign` 或 `/assign @yourself` 认领跟踪。优化类 PR 需在描述中给出优化前后的性能对比数据。

---

## 四、帮助解决他人 Issue

如果社区中他人遇到的问题您有合适的解决方法，欢迎在 Issue 中发表评论交流，共同优化易用性。

如果对应 Issue 需要代码修改，可在 Issue 评论框中输入 `/assign` 或 `/assign @yourself` 认领，跟踪协助解决。

---

## 相关信息

- [快速入门](docs/QUICKSTART.md)
- [目录结构](docs/zh/context/dir_structure.md)
- [算子列表](docs/zh/op_list.md)
- [接口列表](docs/zh/api_list.md)
- [算子开发指南](docs/zh/develop/operator_development_guide.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)（Apache License 2.0）
- [变更记录](CHANGELOG.md)
