# Composite-Torch 算子（纯 PyTorch 组合）

Composite-Torch 轨道放**只用现有 PyTorch 算子组合**实现的算子：前向 / 反向通过
`torch.autograd.Function` 组织，不含 AscendC kernel、不含 Triton kernel，随 Python whl 直接交付。

## 何时选这条轨道

| 场景 | 选择 |
|------|------|
| 组合现有 torch 算子即可表达，无需自定义 kernel | Composite-Torch → `composite-torch/` |
| 需要精细控制核间切分、UB 分配与流水，追求极致性能 | AscendC → `ascendc/` |
| 用 Triton kernel 表达、依赖 autograd、形态多变 | Triton-Ascend → `triton-ascend/` |

## 目录结构

```text
composite-torch/${op_family}/
├── __init__.py                         # 对外导出
├── ${op_family}.py                     # torch.autograd.Function 封装 + 公开 API
└── README.md                           # 算子说明

tests/composite/${op_family}/
└── test_${op_family}.py                # pytest 用例
```

`composite-torch/` 与 `triton-ascend/` 同级，本身不是包目录（测试目录同 `tests/triton/` 取短名 `tests/composite/`）；实现简单时可退化为顶层单文件 `composite-torch/${op_family}.py`，也可按 torch 习惯在包内拆分模块类与函数式入口（如 `conv3d_custom/` 的 `conv3d.py` + `functional.py` 分别对应 `nn.Conv3d` / `F.conv3d`）。

## 对外暴露

- whl：在根 `setup.py` 的 `packages` / `package_dir` 中按算子名登记，打包为
  `oil_gas_ops_prospect.${op_family}`（同 Triton 方式，不建二级后端包）。
- 算子登记：`scripts/local_pkg.py` **自动发现** `composite-torch/` 下的算子
  （子目录含 `__init__.py`，或顶层单文件 `.py`，`_` 前缀跳过），无需手工登记。
- 算子名跨后端唯一：`oil_gas_ops_prospect.<op_family>` 不区分实现后端。

## 封装约定

| 规则 | 说明 |
|------|------|
| 函数式 API 走 `autograd.Function.apply` | 自定义 backward 时保证梯度图完整 |
| 公开名蛇形、无 `npu_` 前缀 | 与其他轨道一致 |
| 不依赖自定义扩展 | 纯 torch 算子组合；需要 kernel 时应走 AscendC / Triton 轨道 |

约束与数学定义写入 [`docs/zh/op_list.md`](../docs/zh/op_list.md)，签名写入 [`docs/zh/api_list.md`](../docs/zh/api_list.md)。完整开发流程见[算子开发指南](../docs/zh/develop/operator_development_guide.md)。
