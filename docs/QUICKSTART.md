# 快速入门

简体中文 | [English](./QUICKSTART_en.md)

本文给出从环境准备到跑通算子的最短路径。更完整的安装说明见[环境部署](zh/context/quick_install.md)。

## 1. 前置条件

| 项 | 要求 |
|----|------|
| CANN | 9.0.0 官方版本（Toolkit 包 + 对应 SoC 的 Ops 包） |
| SoC | Ascend 910B（`Ascend910B1`） |
| Python | >= 3.10 |
| Python 依赖 | `torch`、`torch-npu`，以及 **Triton-Ascend 3.2.1**（不要用 PyPI 社区 `triton` 代替） |
| NPU 设备 | 真实 910B；精度、性能、显存必须在真机上验证通过 |

配置 CANN 环境变量：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

Triton-Ascend 安装步骤见[环境部署 · 安装 Triton-Ascend](zh/context/quick_install.md#6-安装-triton-ascend)。

## 2. 获取源码

```bash
git clone https://gitcode.com/cann/oil-gas-ops-prospect.git
cd oil-gas-ops-prospect
```

## 3. 编译与安装

仓库根目录执行。默认只编译、不安装：

```bash
bash build.sh --pkg      # 只出 OPP .run
bash build.sh            # 出 .run 与 Python whl 到 output/，不安装
bash build.sh --install  # 编译并安装，然后跑 whl 冒烟
```

顶层流程先构建 OPP，再使用工作区产物构建 whl，无需预装本次 OPP。

`--install` 依次完成：

1. 调用 `ascendc/build.sh --pkg` 编译 AscendC OPP `.run` 包（默认 `Ascend910B1`，默认全部算子）。
2. 安装生成的 `oil_gas_ops_prospect_opp_*.run` 到当前 CANN 环境。
3. 构建统一的 `oil_gas_ops_prospect` whl 包。
4. 通过 `pip` 安装 whl。
5. 运行 `tests/test_oil_gas_ops_prospect_whl.py` 冒烟测试。

产物统一落在仓库根目录 `output/`：

```text
output/
├── <SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run   # OPP 算子包
└── wheels/oil_gas_ops_prospect-*.whl                # Python whl
```

安装完成后，后续 shell 中需要保留自定义 opapi 动态库路径：

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/customize/op_api/lib:/usr/local/Ascend/cann-9.0.0/lib64:$LD_LIBRARY_PATH
```

常用参数见[编译参数说明](zh/context/build.md)。

## 4. 调用算子

### AscendC 复数逐点乘

```python
import os
os.environ["OIL_GAS_OPS_COMPLEX_MUL_IMPL"] = "ascendc"   # 默认为 torch 参考实现

import torch
import torch_npu  # noqa: F401
from oil_gas_ops_prospect.complex_mul import complex_mul

ur, ui, vr, vi = (torch.randn(4, 1024, device="npu") for _ in range(4))
wr, wi = complex_mul(ur, ui, vr, vi)
```

### Triton-Ascend 实数 FFT

```python
import torch
import torch_npu  # noqa: F401
from oil_gas_ops_prospect.real_fft import real_irfft, real_rfft

x = torch.randn(8, 256, device="npu")
xr, xi = real_rfft(x, dim=-1)          # 频率轴长度 256 // 2 + 1 = 129
y = real_irfft(xr, xi, n=256, dim=-1)  # 重建实信号
```

更多示例见 [`examples/`](../examples/) 与[算子调用](zh/invocation/quick_op_invocation.md)。

## 5. 测试

分类见 [`tests/README.md`](../tests/README.md)。在已安装 OPP / whl 的 910B 上执行：

```bash
bash build.sh -u
```

算子在 NPU 上的精度、性能、显存必须在真实 910B 上验证通过。

## 下一步

- [算子列表](zh/op_list.md)：当前支持哪些算子、约束是什么
- [接口列表](zh/api_list.md)：完整接口签名与环境变量
- [算子开发指南](zh/develop/operator_development_guide.md)：新增算子步骤
- [贡献指南](../CONTRIBUTING.md)：提交 Issue / PR
