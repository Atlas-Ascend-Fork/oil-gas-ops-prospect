---
inclusion: always
---

# 在 ascendc-dev 容器内编译与验证

宿主机（macOS/arm64）**没有 CANN、没有 NPU**，所有编译和验证一律在 `ascendc-dev`
容器内进行。不要试图在宿主机上跑 `build.sh` / pytest。

## 进容器

容器常处于 `Exited` 状态，先启动：

```bash
docker start ascendc-dev
```

之后一律用 `docker exec` 单行执行，**不要用 `cd` 切宿主目录**：

```bash
docker exec ascendc-dev bash -lc 'cd /root/projects/oil-gas-ops-prospect && <命令>'
```

## 容器环境事实

| 项 | 值 |
|----|----|
| 仓库挂载点 | `/root/projects/oil-gas-ops-prospect`（与宿主同一份，改动实时可见） |
| CANN | `/usr/local/Ascend/cann-9.0.0`（`ASCEND_HOME_PATH` 默认值） |
| msopgen | `$ASCEND_HOME_PATH/bin/msopgen` |
| Python | 3.11 @ `/opt/conda` |
| torch / torch_npu | `2.7.1+cpu` / `2.7.1.post4` |
| triton | `3.2.0`，backend 只有 `ascend` |
| bishengir | `$ASCEND_HOME_PATH/tools/bishengir/bin/bishengir-compile` |
| googletest | `/opt/conda/{include,lib}`（1.11，`find_package(GTest)` 可直接找到） |
| **NPU 设备** | **无**。`npu-smi` 不存在 |

### 两个必须知道的噪音/陷阱

1. 容器内每个 login shell 都会往 stderr 打一行 `can not use command: npu-smi info`。
   它**不是错误**。读输出时过滤掉：

   ```bash
   docker exec ascendc-dev bash -lc '...' 2>&1 | grep -v "npu-smi info"
   ```

2. AscendC 打包（msopgen + CMake + kernel 二进制生成）耗时数分钟。给足 timeout，
   或重定向到日志后台跑，不要用交互式/watch 类命令。

## 无卡环境能验证什么

| 能力 | 无卡可行 | 入口 |
|------|:---:|------|
| AscendC 算子编译 + OPP `.run` 打包 | 是 | `bash build.sh --pkg` |
| OPP 包安装进 CANN、aclnn 符号导出 | 是 | `bash <pkg>.run --quiet` + `nm -D` |
| ophost 侧单元测试（tiling 算法） | 是 | `bash build.sh -u --ophost` |
| aclnn e2e 程序**编译链接** | 是 | `cmake` + `make`（跑需要卡） |
| whl 构建 / 安装 / CPU 回退冒烟 | 是 | `bash tests/run_whl_tests.sh` |
| Triton 算子 AOT 编译（IR + 二进制） | 是 | `triton-ascend/tools/aot_compile_check.py` |
| Triton/算子 CPU 回退精度 | 是 | `verify_*.py --mode accuracy/backward` |
| **算子在 NPU 上的精度/性能/显存** | **否** | 需真实 910B |
| **kernel 实际执行、torch_npu 对比** | **否** | 需真实 910B |

汇报结论时要明确区分"已验证"和"需上卡验证"，不要把编译通过说成功能正确。

## 常用命令

```bash
# 一键：OPP 打包 → 安装 → whl 构建 → 安装 → 冒烟测试
bash build_install_ops.sh --clean --soc Ascend910B1 -j 16

# 只打 OPP 包
cd ascendc && bash build.sh --pkg -v Ascend910B1 -j 16

# ophost UT（googletest，无需卡）
cd ascendc && bash build.sh -u --ophost

# Triton 无卡 AOT 编译验证（全部 kernel 分支）
python3 triton-ascend/tools/aot_compile_check.py

# CPU 回退精度
python3 -m pytest triton-ascend/real_fft/test_triton_fft_real.py -q
```

产物统一落在**仓库根目录 `output/`**（`OIL_GAS_OPS_OUTPUT_DIR` 可覆盖）：
- OPP：`output/<SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run`
- whl：`output/wheels/oil_gas_ops_prospect-<ver>-<py>-<abi>-<plat>.whl`

`ascendc/_work/` 只放 msopgen 临时工程和 ophost UT 构建目录，不再有 `ascendc/dist/`。

## 改包名/改构建后的复验清单

```bash
grep -ri "<旧名>" . --exclude-dir=.git --exclude-dir=build \
  --exclude-dir=_work --exclude-dir=dist --exclude-dir=__pycache__   # 应为空
```

先清干净再验，否则会被旧产物/旧安装骗过：

```bash
rm -rf build *.egg-info; find . -name __pycache__ -type d -exec rm -rf {} +
rm -f output/*/[旧名]*.run output/wheels/[旧名]*.whl
python3 -m pip uninstall -y <旧 dist 名> <新 dist 名>
# 并删掉 site-packages 下残留的包目录 / dist-info / *.so
```

然后跑：`build_install_ops.sh --clean` → whl 冒烟 → real_fft pytest →
ophost UT → AOT 编译验证。

> 验证脚本输出"零残留"时保持怀疑：用**独立的 grep 命令**复核，不要复用同一个
> shell 里可能未展开的变量。曾经因为 `for f in $FILES` 未展开而得到假阳性。
