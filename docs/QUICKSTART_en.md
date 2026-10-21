# Quick Start

English | [简体中文](./QUICKSTART.md)

This page gives the shortest path from environment setup to a running operator. For the complete installation instructions see [Environment Setup](en/context/quick_install.md).

## 1. Prerequisites

| Item | Requirement |
|------|-------------|
| CANN | 9.0.0 official release (Toolkit package + the Ops package for your SoC) |
| SoC | Ascend 910B (`Ascend910B1`) |
| Python | >= 3.10 |
| Python dependencies | `torch`, `torch-npu`, and **Triton-Ascend 3.2.1** (do not substitute the community `triton` from PyPI) |
| NPU device | a real 910B; precision, performance and device memory must be verified on real hardware |

Configure the CANN environment variables:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

For the Triton-Ascend installation steps see [Environment Setup · Install Triton-Ascend](en/context/quick_install.md#6-install-triton-ascend).

## 2. Get the Source

```bash
git clone https://gitcode.com/cann/oil-gas-ops-prospect.git
cd oil-gas-ops-prospect
```

## 3. Build and Install

Run from the repository root. The default builds without installing:

```bash
bash build.sh --pkg      # OPP .run only
bash build.sh            # .run and Python whl into output/, no installation
bash build.sh --install  # build and install, then run the whl smoke test
```

The top-level flow builds the OPP first and then builds the whl from the workspace artifacts, so the current OPP does not need to be preinstalled.

`--install` performs, in order:

1. Calls `ascendc/build.sh --pkg` to build the AscendC OPP `.run` package (default `Ascend910B1`, all operators by default).
2. Installs the generated `oil_gas_ops_prospect_opp_*.run` into the current CANN environment.
3. Builds the unified `oil_gas_ops_prospect` whl package.
4. Installs the whl through `pip`.
5. Runs the `tests/test_oil_gas_ops_prospect_whl.py` smoke test.

All artifacts land under `output/` in the repository root:

```text
output/
├── <SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run   # OPP operator package
└── wheels/oil_gas_ops_prospect-*.whl                # Python whl
```

After installation, later shells still need the custom opapi shared library path:

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/customize/op_api/lib:/usr/local/Ascend/cann-9.0.0/lib64:$LD_LIBRARY_PATH
```

For the commonly used options see [Build Options](en/context/build.md).

## 4. Invoke Operators

### AscendC Complex Element-Wise Multiply

```python
import os
os.environ["OIL_GAS_OPS_COMPLEX_MUL_IMPL"] = "ascendc"   # the torch reference implementation is the default

import torch
import torch_npu  # noqa: F401
from oil_gas_ops_prospect.complex_mul import complex_mul

ur, ui, vr, vi = (torch.randn(4, 1024, device="npu") for _ in range(4))
wr, wi = complex_mul(ur, ui, vr, vi)
```

### Triton-Ascend Real FFT

```python
import torch
import torch_npu  # noqa: F401
from oil_gas_ops_prospect.real_fft import real_irfft, real_rfft

x = torch.randn(8, 256, device="npu")
xr, xi = real_rfft(x, dim=-1)          # frequency axis length 256 // 2 + 1 = 129
y = real_irfft(xr, xi, n=256, dim=-1)  # reconstruct the real signal
```

For more examples see [`examples/`](../examples/) and [Operator Invocation](en/invocation/quick_op_invocation.md).

## 5. Testing

For the categories see [`tests/README.md`](../tests/README.md) (Chinese). On a 910B with the OPP / whl already installed:

```bash
bash build.sh -u
```

Operator precision, performance and device memory on the NPU must be verified on a real 910B.

## Next Steps

- [Operator List](en/op_list.md): which operators are supported and what the constraints are
- [API List](en/api_list.md): full signatures and environment variables
- [Operator Development Guide](en/develop/operator_development_guide.md): steps for adding an operator
- [Contribution Guide](../CONTRIBUTING.md) (Chinese): filing an Issue / PR
