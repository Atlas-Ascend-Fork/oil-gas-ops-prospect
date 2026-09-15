# 算子开发指南

本文说明如何新增 AscendC / Triton-Ascend 算子，以及必须遵守的目录、命名、封装与验收约定。路径以 [`dir_structure.md`](../context/dir_structure.md) 为准。Issue / PR 流程见[贡献指南](../../../CONTRIBUTING.md)。

本仓有两类算子后端，交付件与开发流程不同。请先判断新算子属于哪一类：

| 场景 | 选择 |
|------|------|
| 需要精细控制核间切分、UB 分配与流水，追求极致性能 | AscendC 算子 → `ascendc/` |
| 用 Python 快速表达、依赖 autograd、形态多变 | Triton-Ascend 算子 → `triton-ascend/` |

---

## 一、AscendC 算子

### 1. 目录结构

```text
ascendc/operators/${op_name}/
├── op_host/
│   ├── ${op_name}.cpp                  # 算子原型定义（OpDef）、InferShape、TilingFunc
│   ├── ${op_name}_tiling_data_def.h    # TilingData 结构体定义
│   ├── ${op_name}_tiling_core.h        # 平台无关的切分算法（header-only）
│   └── ${op_name}_tiling_impl.h        # 对接 gert::TilingContext 的实现
└── op_kernel/
    ├── ${op_name}.cpp                  # kernel 入口
    ├── ${op_name}_impl.h               # kernel 实现
    └── ${op_name}_tiling.h             # kernel 侧 TilingData 视图
```

`operators/` 只放实现。C++ 测试与样例放在仓库根 `tests/`、`examples/`。多算子共用的 host / kernel 头放在 `ascendc/operators/_common/`，合并进 msopgen 工程时会平铺到 `op_host/` 与 `op_kernel/`。

一对前反向可以共用一个 Python 模块，但 **manifest 与 `operators/` 各占一行 / 一个目录**。共享逻辑放 `_common/`。

### 2. 注册算子

在 `ascendc/operators/manifest.tsv` 追加一行（**tab 分隔**：目录名、OpType、kernel 入口文件名）。可用如下命令确认注册成功：

```bash
cd ascendc && bash build.sh list-ops
```

### 3. 切分算法必须可测

`<Op>TilingFunc` / `InferShape` 的入口签名依赖 `gert::TilingContext`，`Compute<Op>Tiling` 依赖 `platform_ascendc::PlatformAscendC`；CANN 二进制包未提供 gert faker，两者都无法直接 UT。

因此约定：**把与平台无关的切分算法抽到 `op_host/<op>_tiling_core.h`**，UT 直接驱动核心函数，覆盖的是真实算法。

Host 侧用 `BEGIN_TILING_DATA_DEF`（opbuild 需要）；kernel 侧用 **plain struct** 镜像同一布局（`ccec` 不识别该宏）。`TilingFunc` 负责：形状 / dtype 校验 → 调 `*_tiling_core.h` → `SetBlockDim` / `SetTilingKey` → `SaveToBuffer`。`GetCoreNumAiv()` 为 0 必须失败或给出明确兜底，并在 ophost UT 覆盖该分支。

现有算子的切分策略可作参考：按元素总数在 AIV 核间均分（32B 对齐），核内再按 UB 预算分拍。

### 4. 命名约定

| 侧 | 规则 | 示例 |
|----|------|------|
| C++ OpType | PascalCase | `ComplexMul`、`FusedBiasSoftmax` |
| aclnn | `aclnn` + OpType | `aclnnComplexMul` |
| kernel 入口 / `opFile.value` | 蛇形，与源文件名一致 | `complex_mul` |
| Python 公开函数 | 蛇形，**不加** `npu_` 前缀 | `complex_mul`、`fused_bias_softmax` |
| 反向 OPP 算子 | `Grad` 后缀 | `ComplexMulGrad`、`FusedSoftmaxGrad` |
| 环境变量 | `OIL_GAS_OPS_` 前缀 | `OIL_GAS_OPS_COMPLEX_MUL_IMPL` |

OpType 改名不影响 kernel 源码（入口由 `opFile.value` / `opInterface.value` 固定），但 aclnn 符号随 OpType 变化。

### 5. OpDef 与 kernel 入口

使用 `ops::OpDef` + `OP_ADD`，不要用旧的 `REG_OP`。`opFile.value` / `opInterface.value` 必须等于 `op_kernel/` 下入口文件名（无扩展名）。需要 aclnn 时设 `aclnnSupport.value = support_aclnn`（或以该算子实测为准）。逐元素 float32/ND 算子优先复用 `_common/op_host/og_tiling_context.h` 的 `DeclareRequiredFloatNdIo` / `AddAscend910bElementwiseConfig`。

入口符号必须与 `opFile.value` 一致。Ascend 910B 向量指令以 **64 个元素**为粒度：`count` 不是 64 倍数时，尾部可能被跳过或写成 0。计算用 aligned 长度，`DataCopyPad` 写出用原始 `count`，禁止越界写。UB 按 32B 规划。

### 6. Python 封装

| 文件 | 作用 |
|------|------|
| `ascendc/pybind/<op_family>.py` | 公开 API，打包为 `oil_gas_ops_prospect.<op_family>` |
| `ascendc/pybind/csrc/<op_family>/bindings.cpp` | 直调 aclnn 两段式入口，编成 `<op_family>_npu` |

新建 `csrc/<op_family>/*.cpp` 后，`pybind_extensions.py` 会自动带上扩展，不必再手写 `setup.py` 的 `CppExtension`。写法以现有 `complex_mul.py` 为准：

| 规则 | 说明 |
|------|------|
| 函数式 API 走 `autograd.Function.apply` | 直接调 `ext.forward()` 会断梯度图 |
| 必须有独立 PyTorch 参考 | 默认 `IMPL=torch`；对拍依赖它 |
| 实现开关 | `OIL_GAS_OPS_<OP>_IMPL=torch\|ascendc\|auto`，默认 `torch` |
| 强制自定义算子 | `OIL_GAS_OPS_REQUIRE_ASCENDC_<OP>=1` 时禁止静默回退 |
| `@lru_cache` 缓存扩展 | 避免重复 import |
| 公开名蛇形、无 `npu_` | `complex_mul`，不要 `npu_complex_mul` / `ComplexMul` |

`attention_core` 这类组合封装可以放在同一模块，但必须在 [api_list.md](../api_list.md) 写明「不是独立 OPP 算子」。

### 7. 补充单元测试

AscendC 的 C++ 测试放在仓库根，算子目录只保留实现。本仓没有 ATK / `OP_API_UT`。

```text
tests/ut/op_host/${op_name}/test_${op_name}_tiling.cpp   # tiling UT（googletest）
tests/st/aclnn<OpType>/                                  # 多 case 两段式 aclnn + 真值（需要 NPU）
examples/aclnn/${op_name}/test_aclnn_${op_name}.cpp       # 内联小 tensor，两段式 aclnn（需要 NPU）
```

`tests/ascendc/ophost/CMakeLists.txt` 会自动收集 `tests/ut/op_host/*/test_*.cpp`
并加上各 `op_host` include。Python 精度对比放在 `tests/ascendc/${op_name}/`。
登记 ST / 精度 / 样例到 `tests/suites.json`。

ophost UT 建议覆盖：UB 预算约束、对齐与上限、小 tensor 收缩、退化输入（负元素数、`aivNum=0`、空张量），以及核间切分的覆盖性不变量（每个元素恰好被一个 AIV 核处理一次、区间无空洞无重叠）。融合算子另含 bias 广播 / 全尺寸、`Q≠S`、VEC / MIX 路径分发。

### 8. 编译与验证

```bash
# 打包（AscendC 侧只出 .run）
cd ascendc && bash build.sh --pkg --ops ${op_name}

# 仓库根目录
bash build.sh --pkg      # 只出 OPP .run
bash build.sh            # 编 .run 与 whl，不安装
bash build.sh --install  # 编译并安装

# 910B 上板：UT + ST + 精度 + 性能（写出 test_result/test_report.md）
bash build.sh -u --ut --st --precision --perf --ops=${op_name}
```

完整选项见 [build.md](../context/build.md)。

### 9. NPU 精度/性能测试

在 `tests/ascendc/${op_name}/` 下按现有结构组织，统一以等价的 PyTorch/torch_npu 表达式为基准：

```text
ascendc/pybind/${op_name}.py                 # Python API → oil_gas_ops_prospect.${op_name}
ascendc/pybind/csrc/${op_name}/bindings.cpp  # C++：通过自定义 aclnn 接口调用算子

tests/ascendc/${op_name}/
├── test_${op_name}_npu.py  # 精度/性能对比脚本
├── run_test.sh             # 安装 OPP 包 → 编译扩展 → 运行测试
└── README.md
```

`ascendc/pybind/${op_family}.py` 会直接变成 `oil_gas_ops_prospect.${op_family}`；
`ascendc/pybind/csrc/<op_family>/` 下的 `*.cpp` 会自动编译成扩展模块 `<op_family>_npu`，
无需改构建脚本，详见 [`ascendc/pybind/README.md`](../../../ascendc/pybind/README.md)。

精度标准以[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)为准。本仓已交付算子的验收阈值：

| 算子 | dtype | 阈值 | 出处 |
|------|-------|------|------|
| `ComplexMul` / `Grad` | fp32 | 相对误差默认 `1e-5` | `tests/ascendc/complex_mul/` |
| `FusedBiasSoftmax` / `FusedSoftmaxGrad` | fp32 | `max_diff ≤ 1e-5` | `tests/ascendc/fused_bias_softmax/` |
| 同上 | bf16 | `atol=2e-2, rtol=4e-3` | 同上 |

新算子在 PR 中按 [精度验收报告模板](precision_acceptance_template.md) 填写，通过率 100%。严重 / 主要遗留问题未闭环不得合入。算子在 NPU 上的精度、性能、显存必须在真实 910B 上验证通过。

### 10. 移植 CANN 框架算子的注意事项

部分从 CANN 原仓移植的算子的 host tiling/infershape 依赖 CANN 内部框架头（`op_host/tiling_base.h`、`op_host/tiling_templates_registry.h`、`err/ops_err.h`、`log/log.h` 等），这些头在 msopgen 自定义算子工程的 opbuild 编译环境下缺失或会拉入 slog 依赖。处理方式（保持源码改动最小）：

- 提供本地 compat shim，用等价/降级实现替换 `OP_CHECK_IF`、`OP_CHECK_NULL_WITH_CONTEXT`、`OPS_REPORT_*`、`OP_LOG*`、`ASCENDC_EXTERN_C` 等宏；
- 仅把不可用的 include 替换为该 shim，不改动 tiling 算法本身；
- 若算子确需额外 CANN 头目录，可在 `ascendc/scripts/patch_op_host.py` 的 `_EXTRA_INCLUDE_SUBDIRS` 中追加。

更多细节见 [`ascendc/README.md`](../../../ascendc/README.md)。

当前交付与真机验证面向 **Ascend 910B**（默认 `Ascend910B1`）。新增 SOC 必须同时写入 `SUPPORTED_SOC_VERSIONS`、在 OpDef 的 `AICore().AddConfig(...)` 中增加对应配置，并在该 SOC 上完成精度验收。

---

## 二、Triton-Ascend 算子

### 1. 目录结构

```text
triton-ascend/${op_family}/
├── __init__.py                         # 对外导出
├── triton_${op_family}.py              # Triton kernel 与 autograd 封装
└── README.md                           # 算子说明

tests/triton/${op_family}/
└── test_triton_${op_family}.py         # CPU 回退路径精度测试
```

### 2. 必须提供 CPU 回退

kernel 需提供 **PyTorch 回退实现**，接口语义与 NPU 路径一致。现有 `real_rfft` 的写法可作参考：

```python
def real_rfft(x, dim, norm="ortho", ...):
    if norm != "ortho":
        raise ValueError(f"only norm='ortho' is supported, got {norm!r}")
    if not (_is_npu_tensor(x) and _has_triton()):
        return _torch_rfft_real(x, dim=dim, norm=norm)
    return _RealRFFTFunction.apply(x, dim, norm, ...)
```

### 3. 注册进 whl

在 `setup.py` 的 `packages` 与 `package_dir` 中登记新子包，使其进入 whl：

```python
packages=[
    ...,
    "oil_gas_ops_prospect.${op_family}",
],
package_dir={
    ...,
    "oil_gas_ops_prospect.${op_family}": "triton-ascend/${op_family}",
},
```

不要加 `triton` / `ascendc` 二级包。裸 checkout 由 `scripts/local_pkg.py` 挂导入名；验 whl 时置 `OIL_GAS_OPS_WHL_TEST=1`。

### 4. 纳入 AOT 编译检查

新增 kernel 分支需能被 `triton-ascend/tools/aot_compile_check.py` 覆盖，保证全部分支可编译（TTIR → Linalg → bishengir → npubin）。

约束与数学定义写入 [op_list.md](../op_list.md)，签名写入 [api_list.md](../api_list.md)。

### 5. 本地源码 / whl 切换

验证脚本默认直接使用仓库里的源码（`scripts/local_pkg.py` 把各算子目录挂到
`oil_gas_ops_prospect.<算子名>`），改算子后不需要重新构建 whl：

```bash
python3 tests/triton/real_fft/verify_triton_fft_real_128_3d.py --print-shape-only
```

需要验证**已安装的 whl** 时置 `OIL_GAS_OPS_WHL_TEST=1`，避免测到仓内源码：

```bash
bash tests/run_whl_tests.sh
```

---

## 三、提交前自检

```bash
bash build.sh --install
bash build.sh -u
bash scripts/check_license_header.sh
```

新增源文件需带 Apache License 2.0 许可证头，格式参考仓内已有文件。算子语义或约束变了，必须同步 [op_list.md](../op_list.md)、[api_list.md](../api_list.md)、对应 `README.md`。

### 交付件清单

| 类别 | 路径 |
|------|------|
| Host / Kernel | `ascendc/operators/<op>/op_host/`、`op_kernel/`（含 `*_tiling_core.h`） |
| 注册表 | `ascendc/operators/manifest.tsv` 新行 |
| Python API | `ascendc/pybind/<family>.py`（蛇形公开名、autograd、torch 参考、`OIL_GAS_OPS_*` 开关） |
| aclnn 绑定 | `ascendc/pybind/csrc/<family>/bindings.cpp` |
| ophost UT | `tests/ut/op_host/<op>/test_<op>_tiling.cpp` |
| aclnn example / ST | `examples/aclnn/<op>/`、`tests/st/aclnn<OpType>/`；登记 `tests/suites.json` |
| NPU 对比 | `tests/ascendc/<op>/`（Triton 则为 `tests/triton/<family>/`） |
| 文档 | `docs/zh/op_list.md`、`api_list.md`；AscendC 还要 `ascendc/README.md` |
| 精度报告 | PR 描述附 [模板](precision_acceptance_template.md) |
| Triton 登记 | `setup.py` 的 `packages` / `package_dir`；AOT 检查覆盖 |

不要在 `operators/<op>/` 下再建 `tests/`、`examples/`、`testplan/`、`teste2e/`、`delivery/`。`output/`、`ascendc/_work/`、`build/`、`*.so` 等已在 `.gitignore`，不要提交。
