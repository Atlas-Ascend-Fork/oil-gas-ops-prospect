# 文档导航

oil-gas-ops-prospect 的完整文档索引。

## 入门

| 文档 | 说明 |
|------|------|
| [快速入门](QUICKSTART.md) | 从环境准备到跑通第一个算子的最短路径 |
| [环境部署](zh/context/quick_install.md) | CANN Toolkit / Ops 包、Triton-Ascend 安装与环境变量配置 |
| [目录结构](zh/context/dir_structure.md) | 仓库各目录职责说明 |

## 使用

| 文档 | 说明 |
|------|------|
| [算子列表](zh/op_list.md) | 按算子给出数学定义、输入输出、dtype/format 与约束（含 Triton） |
| [接口列表](zh/api_list.md) | 全部 Python / aclnn 接口签名、参数与环境变量 |
| [算子调用](zh/invocation/quick_op_invocation.md) | 端到端调用示例 |

## 开发

| 文档 | 说明 |
|------|------|
| [编译参数说明](zh/context/build.md) | `build.sh` 的编译构建与测试报告 |
| [算子开发指南](zh/develop/operator_development_guide.md) | 新增 AscendC / Triton 算子的步骤、约定与验收 |
| [精度验收报告模板](zh/develop/precision_acceptance_template.md) | PR 中附带的精度报告格式 |
| [调试与性能分析](zh/debug/op_debug_prof.md) | Host / Kernel 调试与 msProf 性能采集 |

## 社区

- [贡献指南](../CONTRIBUTING.md)
- [安全声明](../SECURITY.md)
- [变更记录](../CHANGELOG.md)
- [许可证](../LICENSE)（Apache License 2.0）
- [所属 SIG：oil-gas-engineering](https://gitcode.com/cann/community/tree/master/CANN/sigs/oil-gas-engineering)
