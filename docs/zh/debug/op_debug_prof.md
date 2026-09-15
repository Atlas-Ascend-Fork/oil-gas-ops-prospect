# 调试与性能分析

本仓 AscendC 算子跑在 **AI Core**（`ComplexMul` / `ComplexMulGrad` 为 AIV，`FusedSoftmaxGrad` 的 bf16 路径为 MIX）。文档说明本仓能用来定位问题、核对精度和采集性能的入口。

精度、性能与显存必须在真实 **910B（NPU）** 上验证通过：

```bash
bash build.sh -u
```

---

## 调试定位（AI Core 算子）

算子执行失败或精度异常时，可分别看 Host 日志和 Kernel 中间结果。

### 1、Host 侧日志

程序结束后，Host 调试日志默认在：

```text
$HOME/ascend/log/debug/plog/plog-pid_*.log
```

路径与分类见 CANN 9.0 [查看日志](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/logreference/logreference_0002.html)。

开启 [`ASCEND_SLOG_PRINT_TO_STDOUT`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0121.html) 可将 Host 应用类日志打到标准输出（`1` 开启，`0` 关闭）。开启后日志不再写入 log 文件：

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

需要更详细级别时设置 [`ASCEND_GLOBAL_LOG_LEVEL`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)（仅对调试日志生效：`0` DEBUG、`1` INFO、`2` WARNING、`3` ERROR）：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```

完整环境变量列表见 [环境变量参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)。

Python 调用路径建议打开同步执行，让报错栈指向真实出错位置：

```python
import torch_npu
torch_npu.npu.set_sync_debug_mode(1)
```

aclnn 调用失败时，用 [`aclGetRecentErrMsg`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_0028.html) 取最近一次 ACL 错误描述（取完即清空，应在失败后立刻调用）。本仓样例 `examples/aclnn/complex_mul/test_aclnn_complex_mul.cpp` 的 `LaunchOp` 目前只打印返回码，排查时可改成：

```cpp
auto ret = aclnnComplexMulGetWorkspaceSize(
    t.inT[0], t.inT[1], t.inT[2], t.inT[3], t.wrT, t.wiT, &workspaceSize, &executor);
if (ret != ACL_SUCCESS) {
    const char* msg = aclGetRecentErrMsg();
    printf("aclnnComplexMulGetWorkspaceSize failed. ERROR: %d\n%s\n",
           ret, msg == nullptr ? "" : msg);
    return ret;
}
```

头文件已由 `examples/_common/aclnn_example.h` 引入 `acl/acl.h`。

### 2、Kernel 调试

`AscendC::printf` / `DumpTensor` 会拖慢 Kernel，只在调测阶段打开。接口说明见 [printf](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0193.html) 与 [DumpTensor](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0192.html)。

本仓样例里 `aclInit(nullptr)`（见 `examples/_common/aclnn_example.h` 的 `Init`），默认不导出 Kernel dump。调测时在可执行文件目录放 `acl.json`，并把 `aclInit(nullptr)` 改成 `aclInit("acl.json")`：

```json
{
  "dump": {
    "dump_kernel_data": "printf,tensor",
    "dump_path": "./kernel_dump"
  }
}
```

落盘路径优先级：`ASCEND_DUMP_PATH` > `ASCEND_WORK_PATH` > 配置里的 `dump_path`。导出的 bin 不能直接当文本看，用 [show_kernel_debug_data](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0102.html) 解析；`printf` 内容在 `parser.log`，`DumpTensor` 在 `dump_data/`。

- printf：打印标量。在 `ComplexMul` 读完 tiling 后（`op_kernel/complex_mul.cpp` 的 `complex_mul` 入口）可临时加上：

```c++
ComplexMulTilingData tilingData;
OgOpsKernel::CopyTilingData(&tilingData, tiling);
AscendC::printf("totalElems=%ld tileFloats=%ld coreNum=%ld blockIdx=%d\n",
                tilingData.totalElems, tilingData.tileFloats, tilingData.coreNum,
                static_cast<int>(GetBlockIdx()));
```

- DumpTensor：打印 UB 上的 LocalTensor。在 `ComplexMulKernel::ProcessTile` 写出 `wr` 之前（`op_kernel/complex_mul_impl.h`）可临时加上，第三参是元素个数，单核 dump 总量不要超过 1MB：

```c++
out = outQue_.DeQue<float>();
const uint32_t dumpN = static_cast<uint32_t>(cur < 32 ? cur : 32);
DumpTensor(out, 0, dumpN);
CopyOutPad(wrGm_[gmOff], out, cur);
```

卡死、GM / UB 越界等场景可用 [msDebug](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/devaids/optool/atlasopdev_16_0062.html) 单步调试。

---

## 本仓常见问题

### 自定义算子没有被调用

现象：设了 `OIL_GAS_OPS_COMPLEX_MUL_IMPL=ascendc`（或融合算子的 `OIL_GAS_OPS_FUSED_BIAS_SOFTMAX_IMPL=ascendc`），但结果/耗时仍像 PyTorch 参考实现。

原因：AscendC 调用失败时默认静默回退。禁止回退：

```bash
export OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL=1
# 融合注意力算子：
export OIL_GAS_OPS_REQUIRE_ASCENDC_FUSED_BIAS_SOFTMAX=1
```

环境变量全集见[接口列表](../api_list.md#环境变量)。

| 根因 | 排查 |
|------|------|
| pybind 扩展未编译或未安装 | `python3 -c "import complex_mul_npu"`（融合算子为 `fused_bias_softmax_npu`） |
| OPP 包未安装 | `ls ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib` |
| 动态库路径缺失 | `echo $LD_LIBRARY_PATH`，需含 `opp/vendors/customize/op_api/lib` |
| 张量不在 NPU 上 | `x.device.type == "npu"` |
| dtype / shape 不满足算子约束 | 见[算子列表](../op_list.md) |

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
```

### 找不到 aclnn 接口

自定义算子的 aclnn 名随 OpType 变化。本仓 `ComplexMul` 对应 `aclnnComplexMul`。改了 OpType 必须同步改 `ascendc/pybind/csrc/` 里的绑定。

### `.run` 包安装后不生效

OPP 安装到 `opp/vendors/<vendor_name>/`，默认 `customize`。多个 vendor 时确认 `opp/vendors/config.ini` 含 `customize`。

```bash
bash output/Ascend910B1/oil_gas_ops_prospect_opp_*.run --quiet
```

### 编译报错：缺少 CANN 内部框架头

从 CANN 原仓移植的 host 代码常依赖 `op_host/tiling_base.h`、`err/ops_err.h`、`log/log.h` 等，msopgen 自定义工程里没有这些头。处理方式见[算子开发指南 · 移植 CANN 框架算子](../develop/operator_development_guide.md#10-移植-cann-框架算子的注意事项)。

### Triton 算子在 CPU 上跑但没报错

`real_rfft` / `real_irfft` 在非 NPU 或 Triton 不可用时回退到 `torch.fft`。要确认走了 NPU kernel，检查张量 device 以及 `triton-ascend` 是否已安装。

---

## 性能采集

时延或显存异常时，用 `msprof op` 看 Kernel 耗时、核数和流水占比。下面命令已在真实 910B、CANN 9.0.0 上用本仓 `ComplexMul` 样例跑通。其它算子把可执行文件换成 `examples/aclnn/<op>/output/test_aclnn_<op>`，仿真 `--kernel-name` 换成该算子 OpType。

- **上板采集**：真实 NPU 上拿 Kernel 耗时、Block 数、流水占比。
- **流水图仿真**：看各核指令流水；比上板更细。

该样例是 `{2,2}` 小 tensor，上板 `Task Duration` 只有数微秒，用来验证采集链路，不能当业务性能数字。代表 shape 的 `ms/call` 仍走本节末尾的 Python `--perf`。

命令与结果文件含义见 [获取性能数据](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0008.html)。

### 方式 1：上板性能采集

先安装 customize vendor，再编样例。`run.sh` 会清掉 `build/`、`output/`，重新 cmake/make，并**立刻跑一遍**功能检查：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
bash examples/aclnn/complex_mul/run.sh
```

再在输出目录做性能采集（这是第二次 launch；默认预热 5 次）：

```bash
cd examples/aclnn/complex_mul/output
msprof op ./test_aclnn_complex_mul
```

结果目录为同级 `OPPROF_<时间戳>_*`。本仓实测摘要：

```text
Op Name: ComplexMul_923973e39f803f5bc6ddb804acabf320_0
Op Type: vector
Task Duration(us): 4.640093
Block Dim: 48
Device Id: 0
```

`Task Duration` 是当前 Kernel 耗时，`Block Dim` 是执行核数。本仓实测结果目录（时间戳与随机串每次不同）：

```text
OPPROF_<时间戳>_<随机串>/
├── OpBasicInfo.csv              # 核名、Op Type、总耗时、Block Dim、设备号
├── ArithmeticUtilization.csv    # 各 block 的 cube/vector 算力占比（vector 看 aiv_time / aiv_vec_ratio）
├── PipeUtilization.csv          # 各流水（VEC / MTE2 / MTE3 / SCALAR）耗时与占比
├── Memory.csv                   # GM ↔ UB 带宽、搬运量
├── MemoryL0.csv                 # L0A/L0B/L0C 带宽（纯 vector 核多为 NA）
├── MemoryUB.csv                 # UB 读写带宽
├── L2Cache.csv                  # L2 命中 / miss
├── ResourceConflictRatio.csv    # bank / MTE 冲突与 wait 占比
└── dump/
    ├── op_basic_info.txt        # 与 OpBasicInfo 对应的文本摘要
    └── pc_start_addr.txt        # Kernel PC 起始地址
```

`FusedSoftmaxGrad` 的 bf16 路径为 MIX，`Op Type` 可能不是纯 vector；该算子尚未用 `msprof op` 复测。

Python 精度 / 性能对比（相对误差与 `ms/call`，含业务 shape）：

```bash
bash build.sh -u --precision --perf --ops=complex_mul
# 或（会尝试安装 output/ 下的 OPP .run，再编 pybind、跑对比）
cd tests/ascendc/complex_mul && bash run_test.sh
```

### 方式 2：仿真流水图采集

本仓默认 `SOC_VERSION=Ascend910B1`。仿真前把仿真库加入 `LD_LIBRARY_PATH`（按实际 CANN 安装路径与 SOC 目录修改）。本仓实测 `--kernel-name=ComplexMul` 即可命中上板同一核名，不必改成入口符号 `complex_mul`：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/Ascend910B1/lib:${LD_LIBRARY_PATH}
cd examples/aclnn/complex_mul/output
msprof op simulator --output=$PWD/pipeline_auto --kernel-name=ComplexMul ./test_aclnn_complex_mul
```

结果在 `$PWD/pipeline_auto/OPPROF_*`（CANN 9.0.0 不落 `visualize_data.bin`）。本仓实测结构：

```text
pipeline_auto/OPPROF_<时间戳>_<随机串>/
├── dump/
│   ├── pc_start_addr.txt        # Kernel PC 起始地址
│   └── object_dump.txt          # 仿真 dump 路径与核名
└── simulator/
    ├── trace.json               # 全核 Chrome Tracing 事件（优先看这个）
    └── coreN.veccore0/          # 每个 vector 核一份；N 与 Block Dim 对应（本样例 0..47）
        ├── trace.json           # 单核流水时间线
        ├── coreN.veccore0_instr_exe.csv   # 指令级：流水、cycles、耗时
        └── coreN.veccore0_code_exe.csv    # 源码行耗时；未编 -g 时只有表头
```

用 Chrome `chrome://tracing` 或 MindStudio Insight 导入 `simulator/trace.json`。看某核堵在哪条指令时打开对应 `*_instr_exe.csv`。

---

## 本仓性能相关开关

### Triton FFT 分块

`OIL_GAS_OPS_FFT_BLOCK_M` 控制 DFT-matmul 的行分块（默认 512，受 910B L0C 中 fp32 累加器 ≤128KB 约束）。实现会按容量收缩，溢出时自动减半重试：

```bash
for m in 128 256 512; do
  OIL_GAS_OPS_FFT_BLOCK_M=$m python3 tests/triton/real_fft/verify_triton_fft_real_128_3d.py
done
```

`verify_triton_fft_real_128_3d.py` 的 `--block-m` 会覆盖该环境变量。

### AscendC 切分

切分算法在各算子的 `ascendc/operators/<op>/op_host/<op>_tiling_core.h`。改完先跑 ophost UT，再在 910B 上板看性能：

```bash
bash build.sh -u --ophost
bash build.sh -u --perf --ops=complex_mul
```
