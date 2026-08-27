---
inclusion: fileMatch
fileMatchPattern: 'ascendc/**'
---

# AscendC 算子工程细节

## 构建流水线

`ascendc/build.py` 的设计前提是**每次构建都从 msopgen 重新生成**与当前 CANN/SOC
匹配的脚手架，再把仓内算子合并进去。因此 `_work/` 是纯临时产物，可随时删。

```
msopgen gen（内嵌 DummyOp 占位 IR）          → 与 CANN/SOC 匹配的 CustomOp 工程
  ↓ patch_presets.py    修正 CMakePresets 的 ASCEND_COMPUTE_UNIT
  ↓ patch_package.py    写 vendor_name，并定义 .run 名前缀替换规则
  ↓ merge_operators.py  删 dummy、拷 op_host/op_kernel、生成多算子 kernel CMakeLists
  ↓ patch_op_host.py    注入额外 include；追加算子自带的 op_host/extra.cmake
  ↓ CustomOp/build.sh   msopgen 自带的 binary + package
```

新增算子：建 `operators/<name>/op_host|op_kernel/` → 在 `operators/manifest.tsv`
追加一行（**tab 分隔**，3 列：目录名 / OpType / kernel 入口文件名）。

## 移植 CANN 框架算子的注意事项

从 CANN 原仓移植的 host tiling 常依赖 CANN 内部框架头（`tiling_base.h`、
`err/ops_err.h`、`log/log.h` 等），这些在 msopgen 自定义算子工程的 opbuild 环境下
缺失，或会拉入 slog 依赖。处理原则是**保持源码改动最小**：

- 提供本地 compat shim（如 `conv3d_log_compat.h`），用等价/降级实现替换
  `OP_CHECK_IF`、`OP_CHECK_NULL_WITH_CONTEXT`、`OPS_REPORT_*`、`OP_LOG*`、
  `ASCENDC_EXTERN_C` 等宏；
- 只把不可用的 include 换成该 shim，**不改 tiling 算法本身**；
- 确实需要额外 CANN 头目录时，加到 `scripts/patch_op_host.py` 的
  `_EXTRA_INCLUDE_SUBDIRS`（会同时注入 opbuild 与 host 库的 include 路径）。

## ophost UT（`ascendc/tests/ophost/`）

入口 `bash build.sh -u --ophost` → `build.py ut-ophost` → cmake + ctest，
产物在 `_work/ophost_ut/`。无需 NPU。

### 为什么不能直接 UT 算子的 tiling 入口

`<Op>TilingFunc(gert::TilingContext*)` 和 `InferShape` 的签名依赖 gert，而
**CANN 二进制包不提供 gert faker**（`TilingContextFaker` 那套基建在 metadef 源码仓，
容器内 `find -name "*faker*"` 为空）。`Compute<Op>Tiling` 又依赖
`platform_ascendc::PlatformAscendC`，同样无法在 UT 里构造。

**约定**：每个算子把与平台无关的切分算法抽到 `op_host/<op>_tiling_core.h`，签名形如

```cpp
bool Compute<Op>TilingCore(int64_t totalElems, uint32_t aivNum, uint64_t libApiWorkspace,
                           <Op>TilingData &tiling, <Op>Plan &plan);
```

`<op>_tiling_impl.h` 只负责从 platform 读出 `GetCoreNumAiv()` /
`GetLibApiWorkSpaceSize()` 后委托给它。这样 UT 直接驱动核心函数，覆盖真实算法，
且 core 头是 header-only、只依赖 `<cstdint>`，**UT 不需要链接任何 CANN 库**。

新增算子时：把它的 `op_host` 目录加进 `tests/ophost/CMakeLists.txt` 的
`OPHOST_UT_INCLUDE_DIRS`，测试源加进 `ophost_ut` 目标。

> 如果将来要 UT 依赖 `platform_ascendc` 的代码（例如移植 CANN 框架算子的 tiling
> 引擎），链接依赖有几个坑：`platform_ascendc.h` 在 `$CANN/<arch>-linux/asc/include`，
> 它 include 的 `platform/soc_spec.h` 在**另一个** include 根
> `$CANN/<arch>-linux/include`；`platform_ascendc::PlatformAscendC{,Manager}` 的符号在
> `libtiling_api.a` 而**不在** `libplatform.so`；`libtiling_api.a` 还会调 slog 的
> `CheckLogLevel`，需要一起链 `libunified_dlog.so`。

## complex_mul / complex_mul_grad 算子要点

两个 **AIV-only** 逐元素算子（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`），
float32 / `ND` / 全部同 shape，`TILING_KEY` 固定为 0（单路径）。

```text
fwd:  wr = ur*vr - ui*vi          bwd:  gur =  gwr*vr + gwi*vi
      wi = ur*vi + ui*vr                gui = -gwr*vi + gwi*vr
                                        gvr =  gwr*ur + gwi*ui
                                        gvi = -gwr*ui + gwi*ur
```

- **切分**：host 只给 `totalElems / tileFloats / coreNum` 三个字段；核间划分在 kernel 侧
  `SplitByCore()` 完成 —— `per = ceil(total/coreNum)` 再上取整到 8 个 float（32B），
  `start = blockIdx * per`，越界核拿到 `len = 0` 直接返回。
- **UB 预算**：`tileFloats` 由 `UB_BUDGET_BYTES / (UB_SLOT_FACTOR * 4)` 反推再对齐到 8。
  `UB_SLOT_FACTOR` 是 double buffer 下的 tile 份数：前向 `2*(4进+2出)+1tmp = 13`，
  反向 `2*(6进+4出)+1tmp = 21`。**改动 in/out 路数或 buffer 深度时必须同步改这个常量**，
  否则会超 UB；ophost UT 有断言兜住。
- **单拍队列布局**：`inQue` / `outQue` 每槽是多路拼接的连续块，用 `in[k * t]` 取第 k 路
  （`t = tileFloats`），而不是每路一个 que。
- 搬运统一用 `DataCopyPad` + `DataCopyExtParams`（按字节给 `blockLen`），因此尾块不需要
  额外对齐处理。
- 没有手写 L2 aclnn 接口，直接用 msopgen 从 OpType 生成的 `aclnnComplexMulOg` /
  `aclnnComplexMulGradOg`（所以这两个算子**不需要** `op_host/extra.cmake`）。
