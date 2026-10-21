# Environment Setup

English | [简体中文](../../zh/context/quick_install.md)

## Version Compatibility

The CANN versions verified against this repository:

| CANN Version | Release Date |
|--------------|--------------|
| [9.0.0 official release](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0) | 2026/04/30 |

This repository requires the **Toolkit package**, the **Ops package** for your SoC, and the **[Triton-Ascend](#6-install-triton-ascend)** build that matches your CANN version. Pick the `.run` installers that match your CPU architecture and product model from the directory linked above.

Installer file name patterns:

- Toolkit package
  - `Ascend-cann-toolkit_${cann_version}_linux-aarch64.run`
  - `Ascend-cann-toolkit_${cann_version}_linux-x86_64.run`
- Ops package
  - `Ascend-cann-${soc_name}-ops_${cann_version}_linux-aarch64.run`
  - `Ascend-cann-${soc_name}-ops_${cann_version}_linux-x86_64.run`

## 1. Install the CANN Toolkit Package

```bash
# make sure the installer is executable
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
# install
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
```

- `${cann_version}`: installer version, `9.0.0` here.
- `${arch}`: CPU architecture, such as `aarch64` or `x86_64`.
- `${install_path}`: installation path. Defaults to `/usr/local/Ascend`.

## 2. Install the CANN Ops Package

The Ops package must be installed into the same directory as the Toolkit package. Choose the one matching your product model.

```bash
chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
```

- `${soc_name}`: NPU product model name. This repository currently targets Ascend 910B, so the value is `910b`.

## 3. Configure Environment Variables

`${install_path}` is the **installer prefix** passed to the `.run` installers (for example `/usr/local/Ascend`), whereas `ASCEND_HOME_PATH` must be the **actual CANN root** that contains `set_env.sh`. After installing into the default layout, run:

```bash
export ASCEND_HOME_PATH=${install_path}/cann
source ${ASCEND_HOME_PATH}/set_env.sh
```

Common detectable layouts include `<prefix>/cann`, `<prefix>/cann-9.0.0`, and `<prefix>/ascend-toolkit/latest`; use the directory that actually contains `set_env.sh`.

## 4. SoC Support

### Verified Matrix

Open the [CANN download center](https://www.hiascend.com/cann/download), copy the hardware query command shown on the page, run it in your environment to get the chip ID, then feed that ID back into the website to obtain the product name.

| SoC Model | `SOC_VERSION` | Verification Status |
|-----------|---------------|---------------------|
| Atlas A2 training/inference series | `Ascend910B1` | ✅ Verified |
| Atlas A3 training/inference series | `Ascend910_93` | ❌ Not verified |
| 950 series | `Ascend950` | ❌ Not verified |

### Accepted Build-Parameter Values

`SUPPORTED_SOC_VERSIONS` in `ascendc/scripts/config.py` is only the allowlist for msopgen / build arguments; it is not the verified support matrix. Passing a model outside the allowlist fails the build immediately.

## 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> `torch-npu` must match your local CANN version exactly. It is normally installed from the release channel that ships with CANN; pulling an arbitrary version from PyPI is not recommended.

## 6. Install Triton-Ascend

The Triton-Ascend operators in this repository (`real_rfft` / `real_irfft` / `complex_reciprocal`) depend on the `triton.backends.ascend` backend provided by **Triton-Ascend**. The community `triton` package on PyPI has no Ascend backend and cannot replace the installation below. This repository has verified the **CANN 9.0.0 official release** together with **`triton-ascend==3.2.1`**.

For upstream details see [Triton-Ascend](https://gitcode.com/Ascend/triton-ascend) ([installation guide](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html)). Installation commands:

```bash
pip uninstall -y triton triton-ascend
pip install triton-ascend==3.2.1 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

After installing, confirm that the package is `triton-ascend` at version `3.2.1`:

```bash
python3 -c "import importlib.metadata as m; print(m.version('triton-ascend'))"
```

Neither `pip install .` nor installing the project wheel **installs Triton-Ascend automatically**. Preinstall it from the Ascend package index as shown in this section so that pip does not resolve the community `triton` package from public PyPI as the required backend. The project dependency metadata does not declare community `triton`, and this repository neither downloads nor redistributes Triton-Ascend.

## 7. Install This Project

See [Quick Start · Build and Install](../../QUICKSTART_en.md#3-build-and-install), or simply run:

```bash
bash build.sh --install
```

After installation, later shells still need the custom opapi shared library path:

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
```

If CANN is not installed at the default `/usr/local/Ascend/cann-9.0.0`, point to it with an environment variable:

```bash
ASCEND_HOME_PATH=/path/to/cann bash build.sh --install
```
