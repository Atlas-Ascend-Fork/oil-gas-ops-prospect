# 快速入门

本文给出从环境准备到跑通算子的最短路径。更完整的安装说明见[环境部署](zh/context/quick_install.md)。

## 1. 前置条件

| 项 | 要求 |
|----|------|
| CANN | 9.0.0（Toolkit 包 + 对应 SoC 的 Ops 包） |
| SoC | Ascend 910B（`Ascend910B1`） |
| Python | >= 3.10 |
| Python 依赖 | `torch`、`torch-npu`、`triton`（Triton-Ascend），版本需与 CANN 配套 |
| NPU 设备 | **精度与性能验证必须有真实 NPU**；编译与部分单元测试无卡即可 |

配置 CANN 环境变量：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

## 2. 获取源码

```bash
git clone https://gitcode.com/cann/oil-gas-ops-prospect.git
cd oil-gas-ops-prospect
```

## 3. 一键构建并安装

在仓库根目录执行：

```bash
bash build.sh
```

该脚本依次完成：

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

## 5. 无卡验证

开发容器（如 `ascendc-dev`）没有 NPU 设备时，下列验证**不需要卡**：

```bash
# 全量无卡单元测试（--ut 是 -u 的别名）
bash build.sh -u
# 只跑 ophost / 只编译不执行：
# bash build.sh -u --ophost
# bash build.sh -u --noexec
```

等价的分步命令：

```bash
# 1. ophost 侧 tiling 单元测试（googletest）
cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
#    或：bash build.sh -u --ophost
#    或走 AscendC 侧入口：bash ascendc/build.sh -u --ophost

# 2. Triton 算子 CPU 回退精度
python3 -m pytest tests/triton/real_fft -q

# 3. Triton kernel 全分支 AOT 编译（TTIR → Linalg → bishengir → npubin）
python3 triton-ascend/tools/aot_compile_check.py
```

> **编译通过不等于功能正确。** 算子在 NPU 上的精度、性能、显存必须在真实 910B 上验证。

## 下一步

- [算子列表](zh/op_list.md)：当前支持哪些算子、约束是什么
- [接口列表](zh/api_list.md)：完整接口签名与环境变量
- [算子开发指南](zh/develop/operator_development_guide.md)：新增算子步骤
- [算子开发规格](zh/develop/operator-dev-spec.md)：约束、交付件与验收
- [贡献指南](../CONTRIBUTING.md)：提交 Issue / PR
