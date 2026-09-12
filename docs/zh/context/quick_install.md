# 环境部署

## 版本配套

当前仓库已验证通过的 CANN 版本如下：

| CANN 版本 | 发布时间 | 分支 |
|-----------|----------|------|
| [CANN 9.0.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0) | 2026/04/30 | master |

本仓库需要安装 **Toolkit 包**和对应 SoC 的 **Ops 包**。请根据实际 CPU 架构和产品型号，从上述链接目录中选择对应的 `.run` 安装包。

安装包文件名格式：

- Toolkit 包
  - `Ascend-cann-toolkit_${cann_version}_linux-aarch64.run`
  - `Ascend-cann-toolkit_${cann_version}_linux-x86_64.run`
- Ops 包
  - `Ascend-cann-${soc_name}-ops_${cann_version}_linux-aarch64.run`
  - `Ascend-cann-${soc_name}-ops_${cann_version}_linux-x86_64.run`

## 1. 安装 CANN Toolkit 包

```bash
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
# 安装命令
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
```

- `${cann_version}`：安装包版本号，此处为 `9.0.0`。
- `${arch}`：CPU 架构，如 `aarch64`、`x86_64`。
- `${install_path}`：指定安装路径，默认安装在 `/usr/local/Ascend` 目录。

## 2. 安装 CANN Ops 包

Ops 包需与 Toolkit 包安装到同一目录下，请根据实际产品型号选择。

```bash
chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
```

- `${soc_name}`：NPU 产品型号名称。本仓当前面向 Ascend 910B，取值为 `910b`。

## 3. 配置环境变量

```bash
source ${install_path}/cann/set_env.sh
```

请将 `${install_path}` 替换为 Toolkit 包的实际安装目录，例如 `/usr/local/Ascend` 或 `${HOME}/Ascend`。

## 4. SoC 支持矩阵

访问 [CANN 下载中心](https://www.hiascend.com/cann/download)，根据页面提示复制硬件查询命令，在当前环境中执行返回芯片 ID 信息，再回填到官网获取产品名。

| SoC 型号 | `SOC_VERSION` | 支持状态 |
|----------|---------------|----------|
| Atlas A2 训练/推理系列产品 | `Ascend910B1` | ✅ 默认支持 |
| Atlas A3 训练/推理系列产品 | `Ascend910_93` | ❌ 暂未验证 |
| 950 系列产品 | `Ascend950` | ❌ 暂未验证 |

本仓当前支持的 SoC 列表定义在 `ascendc/scripts/config.py` 的 `SUPPORTED_SOC_VERSIONS`，构建时传入不在列表中的型号会直接报错。

## 5. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> `torch-npu` 与 Triton-Ascend 需与本地 CANN 版本严格配套，通常随 CANN 配套发布渠道安装，不建议直接从 PyPI 拉取任意版本。

## 6. 安装本项目

见[快速入门](../../QUICKSTART.md)第 3 节，或直接：

```bash
bash build.sh
```

安装完成后，后续 shell 中需要保留自定义 opapi 动态库路径：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
```

## 7. 容器环境说明

在开发容器（如 `ascendc-dev`）中，仓库通常挂载在 `/root/projects/oil-gas-ops-prospect`。容器内一般没有 NPU 设备，此时只能做编译与无卡验证，见[快速入门](../../QUICKSTART.md)第 5 节。

如果 CANN 安装路径不是默认的 `/usr/local/Ascend/cann-9.0.0`，通过环境变量指定：

```bash
ASCEND_HOME_PATH=/path/to/cann bash build.sh
```
