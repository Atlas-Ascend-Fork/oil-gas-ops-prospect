# 调试与性能分析

## 常见问题定位

### 自定义算子没有被调用

现象：设置了 `OIL_GAS_OPS_COMPLEX_MUL_IMPL=ascendc`，但性能与 PyTorch 实现一致。

原因：`complex_mul` 在 AscendC 调用失败时**默认静默回退**到 PyTorch 实现。

定位：

```bash
export OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL=1
```

置 1 后调用失败会直接抛异常，暴露真实原因。常见根因：

| 根因 | 排查 |
|------|------|
| pybind 扩展 `complex_mul_npu` 未编译或未安装 | `python3 -c "import complex_mul_npu"` |
| OPP 包未安装 | `ls ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib` |
| 动态库路径缺失 | `echo $LD_LIBRARY_PATH`，确认含 `opp/vendors/customize/op_api/lib` |
| 张量不在 NPU 上 | `x.device.type == "npu"` |
| dtype 不是 float32 或几个输入 shape 不一致 | 检查入参 |

补齐动态库路径：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
```

### 找不到 aclnn 接口

自定义算子的 aclnn 接口名随 OpType 变化。本仓 `ComplexMul` 对应 `aclnnComplexMul`。改了 OpType 就要同步改 pybind 绑定。

### `.run` 包安装后不生效

OPP 包安装到 `opp/vendors/<vendor_name>/`，默认 `customize`。若 CANN 侧配置了多个 vendor，需确认 `opp/vendors/config.ini` 中的优先级包含 `customize`。

重装：

```bash
bash output/Ascend910B1/oil_gas_ops_prospect_opp_*.run --quiet
```

### 编译报错：缺少 CANN 内部框架头

从 CANN 原仓移植的算子常依赖 `op_host/tiling_base.h`、`err/ops_err.h`、`log/log.h` 等内部头，msopgen 自定义算子工程的 opbuild 环境下不可用。处理方式见[算子开发指南](../develop/operator_development_guide.md)第 8 节。

### Triton 算子在 CPU 上跑但没报错

`real_rfft` / `real_irfft` 在非 NPU 环境（或 Triton 不可用）会**自动回退到 `torch.fft`**。这是设计行为，用于无卡精度自检。确认是否走了 NPU kernel，检查张量 device 与 Triton 是否可用。

## 无卡验证手段

容器（如 `ascendc-dev`）没有 NPU 时，下列手段不需要卡：

```bash
# 全量无卡单元测试
bash build.sh -u
```

分项说明：

| 手段 | 命令 | 覆盖什么 |
|------|------|----------|
| ophost tiling UT | `bash ascendc/build.sh -u --ophost` | 切分算法：UB 预算约束、对齐与上限、退化输入、核间切分覆盖性不变量 |
| 顶层 CMake + ctest | `cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure` | 同上，走标准 CMake 流程 |
| Triton CPU 回退精度 | `python3 -m pytest tests/triton/real_fft -q` | Triton 算子语义（对 `torch.fft` 对拍） |
| Triton AOT 编译 | `python3 triton-ascend/tools/aot_compile_check.py` | 全部 kernel 分支可编译（TTIR → Linalg → bishengir → npubin） |
| OPP 打包 | `bash ascendc/build.sh --pkg` | 算子能被 opbuild / bisheng 编译并打包 |
| 许可证头 | `bash scripts/check_license_header.sh` | 合规 |

> **编译通过不等于功能正确。** 精度、性能、显存必须在真实 910B 上验证。

## NPU 上的精度与性能验证

```bash
cd tests/ascendc/complex_mul
bash run_test.sh

# 指定 shape
bash run_test.sh --shape 1 1 96 160 160

# 指定 OPP 包
OIL_GAS_OPS_RUN_PKG=/path/to/oil_gas_ops_prospect_opp_openEuler_aarch64.run bash run_test.sh
```

`run_test.sh` 分两步：安装/更新 OPP 包 → 编译 pybind 扩展并跑对比脚本。也可手动分步：

```bash
bash ../../../output/Ascend910B1/oil_gas_ops_prospect_opp_openEuler_aarch64.run --quiet
python3 test_complex_mul_npu.py --build-ext    # 编译扩展并测试
python3 test_complex_mul_npu.py                # 复用已编译扩展
```

输出形如：

```text
[fwd] shape=... rel_err wr=... wi=...
[bwd] rel_err gur=... gui=... gvr=... gvi=...
[perf] fwd torch=...ms ascendc=...ms speedup=...x
[perf] bwd torch=...ms ascendc=...ms speedup=...x
[autograd] gur=... gui=... gvr=... gvi=...
PASS
```

退出码：`0` 精度达标；`1` 超阈值 / 无 NPU / 扩展缺失。

## 性能调优

### Triton FFT 分块

`OIL_GAS_OPS_FFT_BLOCK_M` 控制 DFT-matmul 的行分块大小（默认 512，受 L0C 中 fp32 累加器 ≤128KB 约束）。扫参：

```bash
for m in 128 256 512; do
  OIL_GAS_OPS_FFT_BLOCK_M=$m python3 tests/triton/real_fft/verify_triton_fft_real_128_3d.py
done
```

`verify_triton_fft_real_128_3d.py` 也支持 `--block-m` 参数覆盖该环境变量。

### AscendC 切分

切分算法在各算子的 `op_host/<op>_tiling_core.h`，改动后先用 ophost UT 验证不变量，再上板测性能。

### msprof 采集

CANN 提供的 `msprof` 可采集算子级性能数据，用于定位瓶颈（AIV 利用率、搬运/计算比、UB 命中）。采集需要真实 NPU 设备。

## 日志

CANN 运行日志（plog）默认在 `~/ascend/log/`。提高日志级别：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1   # 0-DEBUG 1-INFO 2-WARNING 3-ERROR
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

算子执行异常时建议同时开启同步执行，让报错栈指向真实出错位置：

```python
import torch_npu
torch_npu.npu.set_sync_debug_mode(1)
```
