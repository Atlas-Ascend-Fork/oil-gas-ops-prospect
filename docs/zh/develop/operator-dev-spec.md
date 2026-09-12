# 算子开发与提交规范

> **适用范围**：本仓库新增或修改的 AscendC（`ascendc/operators/`）与 Triton-Ascend（`triton-ascend/`）算子。
>
> **与[开发指南](operator_development_guide.md)的分工**：指南讲步骤；本文讲必须遵守的约束、交付件与验收。路径以 [`dir_structure.md`](../context/dir_structure.md) 为准。
>
> **版本**：v2.0（按开源布局与当前实现刷新；不再沿用仓外 e3nn 的 `testplan/` / `teste2e/` / `delivery/` 目录约定）。

---

## 目录

1. [总则](#1-总则)
2. [目录与交付件](#2-目录与交付件)
3. [AscendC 源码规范](#3-ascendc-源码规范)
4. [Python 封装规范](#4-python-封装规范)
5. [Triton-Ascend 规范](#5-triton-ascend-规范)
6. [构建与打包](#6-构建与打包)
7. [测试与精度](#7-测试与精度)
8. [PR 产物清单](#8-pr-产物清单)
9. [审核与迭代](#9-审核与迭代)
10. [分支与提交](#10-分支与提交)
11. [附录：检查清单](#11-附录检查清单)

---

## 1. 总则

### 1.1 目标

每个算子从源码到真机运行形成可复现、可审核的闭环：编译通过不等于功能正确。

### 1.2 核心原则

- **闭环验证**：源码 → 编译 / 打包 → 安装 → 无卡 UT → 真机精度 / 性能 / 显存。
- **TDD**：先写 ophost UT 或 CPU 回退用例，再写实现。
- **双环境**：无卡覆盖切分与回退逻辑；NPU 覆盖精度、性能、内存安全。
- **测试不进算子目录**：用例一律放 `tests/`，`ascendc/operators/` 与 `triton-ascend/` 只放实现。

### 1.3 命名约定

| 侧 | 规则 | 示例 |
|----|------|------|
| C++ OpType | PascalCase | `ComplexMul`、`FusedBiasSoftmax` |
| aclnn | `aclnn` + OpType | `aclnnComplexMul` |
| kernel 入口 / `opFile.value` | 蛇形，与源文件名一致 | `complex_mul` |
| Python 公开函数 | 蛇形，**不加** `npu_` 前缀 | `complex_mul`、`fused_bias_softmax` |
| 反向 OPP 算子 | `Grad` 后缀 | `ComplexMulGrad`、`FusedSoftmaxGrad` |
| 环境变量 | `OIL_GAS_OPS_` 前缀 | `OIL_GAS_OPS_COMPLEX_MUL_IMPL` |

OpType 改名不影响 kernel 源码（入口由 `opFile.value` / `opInterface.value` 固定），但 aclnn 符号随 OpType 变化。

一对前反向可以共用一个 Python 模块（`fused_bias_softmax.py` 同时导出 `fused_bias_softmax` 与 `fused_softmax_grad`），但 **manifest 与 `operators/` 各占一行 / 一个目录**。

### 1.4 支持的 SoC

当前交付与真机验证面向 **Ascend 910B**（默认 `Ascend910B1`）。`ascendc/scripts/config.py` 的 `SUPPORTED_SOC_VERSIONS` 还可注册构建用 SOC；新增 SOC 必须同时：

1. 写入 `SUPPORTED_SOC_VERSIONS`；
2. 在 OpDef 的 `AICore().AddConfig(...)` 中增加对应配置；
3. 在该 SOC 上完成精度验收。

---

## 2. 目录与交付件

### 2.1 仓库对照（当前布局）

```text
oil-gas-ops-prospect/
├── ascendc/
│   ├── operators/
│   │   ├── manifest.tsv
│   │   ├── _common/                    # 多算子共用 host/kernel 头
│   │   └── <op_name>/op_host|op_kernel/
│   ├── pybind/                         # 打包为 oil_gas_ops_prospect
│   │   ├── <op_family>.py
│   │   └── csrc/<op_family>/bindings.cpp
│   ├── pybind_extensions.py            # 按 csrc/<op_family>/ 自动发现扩展
│   └── build.sh
├── triton-ascend/<op_family>/
├── tests/
│   ├── ascendc/ophost/                 # googletest，无卡
│   ├── ascendc/<op_name>/              # NPU 精度/性能
│   └── triton/<op_family>/
├── setup.py                            # whl：package_dir 组装 + 自动带上 pybind 扩展
└── build.sh                            # 顶层：OPP + whl + 冒烟；-u 无卡 UT
```

产物落在仓库根 `output/`（可用 `OIL_GAS_OPS_OUTPUT_DIR` 覆盖），**不要**再写 `ascendc/dist/`、`python/`、`tests/`。

### 2.2 AscendC 单算子目录

以现有 `complex_mul` / `fused_bias_softmax` 为准：

```text
ascendc/operators/<op_name>/
├── op_host/
│   ├── <OpType>.cpp 或 <op_name>.cpp   # OpDef + InferShape + TilingFunc 入口
│   ├── <op_name>_tiling.cpp            # 【可选】gert 入口较长时拆出
│   ├── <op_name>_tiling_data_def.h     # BEGIN_TILING_DATA_DEF
│   ├── <op_name>_tiling_core.h         # 【必须】平台无关切分算法
│   └── <op_name>_tiling_impl.h         # 对接 PlatformAscendC / TilingContext
└── op_kernel/
    ├── <op_name>.cpp                   # kernel 入口
    ├── <op_name>_impl.h                # 【可选】实现
    └── <op_name>_tiling.h              # device 侧 plain struct 镜像
```

切分算法**必须**抽到 `*_tiling_core.h`：`TilingFunc` 依赖 `gert::TilingContext`，平台读取依赖 `PlatformAscendC`，CANN 二进制包没有 gert faker，UT 只能直接驱动 core。

### 2.3 一个目录对应一个 OpType

`manifest.tsv` 一行一个 OpType（tab 分隔：目录名、OpType、kernel 入口文件名）：

```tsv
# 目录名	OpType(CANN)	kernel入口文件名
complex_mul	ComplexMul	complex_mul.cpp
complex_mul_grad	ComplexMulGrad	complex_mul_grad.cpp
fused_bias_softmax	FusedBiasSoftmax	fused_bias_softmax.cpp
fused_softmax_grad	FusedSoftmaxGrad	fused_softmax_grad.cpp
```

前反向即使 Python 同模块，也要分两个算子目录。共享逻辑放 `_common/` 或各自的 `*_kernel.h`，入口 `.cpp` 分别 `#include`。

---

## 3. AscendC 源码规范

### 3.1 OpDef（rt2.0）

使用 `ops::OpDef` + `OP_ADD`，不要用旧的 `REG_OP`。

```cpp
namespace ops {
class ComplexMul : public OpDef {
public:
    explicit ComplexMul(const char *name) : OpDef(name)
    {
        DeclareRequiredFloatNdIo(*this, {"ur", "ui", "vr", "vi"}, {"wr", "wi"});
        this->SetInferShape(ge::ComplexMulInferShape).SetInferDataType(ge::ComplexMulInferDataType);
        this->AICore().SetTiling(optiling::ComplexMulTilingFunc);
        AddAscend910bElementwiseConfig(*this, "complex_mul");  // opFile / opInterface
    }
};
OP_ADD(ComplexMul);
}
```

**约定**：

- `opFile.value` / `opInterface.value` 必须等于 `op_kernel/` 下入口文件名（无扩展名）。
- 需要 aclnn 时设 `aclnnSupport.value = support_aclnn`（或依赖 msopgen 默认生成两段式入口，以该算子实测为准，见 `FusedSoftmaxGrad`）。
- 每个 OpType 单独一份 OpDef 源文件。
- 逐元素 float32/ND 算子优先复用 `_common/op_host/og_tiling_context.h` 的 `DeclareRequiredFloatNdIo` / `AddAscend910bElementwiseConfig`。

### 3.2 Tiling

Host 侧用 `BEGIN_TILING_DATA_DEF`（opbuild 需要）；kernel 侧用 **plain struct** 镜像同一布局（`ccec` 不识别该宏）。

```cpp
// op_host/<op>_tiling_data_def.h
BEGIN_TILING_DATA_DEF(ComplexMulTilingDataDef)
    TILING_DATA_FIELD_DEF(int64_t, totalElems);
    TILING_DATA_FIELD_DEF(int64_t, tileFloats);
    TILING_DATA_FIELD_DEF(int64_t, coreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ComplexMul, ComplexMulTilingDataDef)
```

`TilingFunc` 负责：形状 / dtype 校验 → 调 `*_tiling_core.h` → `SetBlockDim` / `SetTilingKey` → `SaveToBuffer`。`GetCoreNumAiv()` 为 0 必须失败或给出明确兜底，并在 ophost UT 覆盖该分支。

### 3.3 Kernel 入口

```cpp
extern "C" __global__ __aicore__ void complex_mul(GM_ADDR ur, GM_ADDR ui, GM_ADDR vr, GM_ADDR vi,
    GM_ADDR wr, GM_ADDR wi, GM_ADDR workspace, GM_ADDR tiling)
{
    // 按 tiling 分核、分拍；AIV-only 算子在 kernel 侧声明 KERNEL_TYPE_AIV_ONLY
}
```

入口符号必须与 `opFile.value` 一致。

### 3.4 910B 向量 64 元素对齐

Ascend 910B 向量指令以 **64 个元素**为粒度。`count` 不是 64 倍数时，尾部可能被跳过或写成 0。

```cpp
uint32_t aligned = (count + 63u) & ~63u;
CompareScalar(mask, x, 0.0f, CMPMODE::GT, aligned);  // 计算用 aligned
DataCopyPad(outGm_[offset], out, outParams);         // 写出用原始 count，禁止越界写
```

`DataCopy` / `repeatTimes` 等其它对齐限制见 ascendc-api-best-practices；UB 按 32B 规划。

### 3.5 移植 CANN 框架算子

host tiling 若依赖 `tiling_base.h`、`ops_err.h`、`log/log.h` 等内部头，在 msopgen 工程里用本地 compat shim 替换宏，不要改切分算法。额外 include 目录加到 `ascendc/scripts/patch_op_host.py` 的 `_EXTRA_INCLUDE_SUBDIRS`。详见 [ascendc/README.md](../../../ascendc/README.md)。

---

## 4. Python 封装规范

### 4.1 位置与发现

| 文件 | 作用 |
|------|------|
| `ascendc/pybind/<op_family>.py` | 公开 API，打包为 `oil_gas_ops_prospect.<op_family>` |
| `ascendc/pybind/csrc/<op_family>/bindings.cpp` | 直调 aclnn 两段式入口，编成 `<op_family>_npu` |

新建 `csrc/<op_family>/*.cpp` 后，`pybind_extensions.py` 会自动带上扩展，**不必**再手写 `setup.py` 的 `CppExtension`。Triton 子包仍需在 `setup.py` 的 `packages` / `package_dir` 登记。

### 4.2 标准模式

以现有 `complex_mul.py` 为准：

```python
@lru_cache(maxsize=1)
def _load_extension():
    return importlib.import_module("complex_mul_npu")

def _require_custom() -> bool:
    return os.environ.get("OIL_GAS_OPS_REQUIRE_ASCENDC_COMPLEX_MUL", "0") == "1"

def _impl() -> str:
    return os.environ.get("OIL_GAS_OPS_COMPLEX_MUL_IMPL", "torch").strip().lower()

def torch_complex_mul(...):  # 纯 PyTorch 参考，CPU / 对拍
    ...

class _ComplexMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ur, ui, vr, vi, use_ascendc):
        ...
    @staticmethod
    def backward(ctx, gwr, gwi):
        ...

def complex_mul(ur, ui, vr, vi):
    use_ascendc = _impl() in ("ascendc", "auto") and ur.device.type == "npu"
    return _ComplexMulFn.apply(ur, ui, vr, vi, use_ascendc)
```

### 4.3 必须遵守

| 规则 | 说明 |
|------|------|
| 函数式 API 走 `autograd.Function.apply` | 直接调 `ext.forward()` 会断梯度图 |
| 必须有独立 PyTorch 参考 | 默认 `IMPL=torch`；对拍与无卡测试依赖它 |
| 实现开关 | `OIL_GAS_OPS_<OP>_IMPL=torch\|ascendc\|auto`，默认 `torch` |
| 强制自定义算子 | `OIL_GAS_OPS_REQUIRE_ASCENDC_<OP>=1` 时禁止静默回退 |
| `@lru_cache` 缓存扩展 | 避免重复 import |
| 公开名蛇形、无 `npu_` | `complex_mul`，不要 `npu_complex_mul` / `ComplexMul` |
| `from __future__ import annotations` | 兼容 `X \| None` |

`attention_core` 这类组合封装可以放在同一模块，但必须在 [api_list.md](../api_list.md) 写明「不是独立 OPP 算子」。

---

## 5. Triton-Ascend 规范

- 实现放 `triton-ascend/<op_family>/`，测试放 `tests/triton/<op_family>/`。
- **必须**提供非 NPU / 无 Triton 时的 PyTorch 回退，接口语义与 NPU 路径一致。
- 在 `setup.py` 登记为 `oil_gas_ops_prospect.<op_family>`，不要加 `triton` 二级包。
- 新 kernel 分支必须能被 `triton-ascend/tools/aot_compile_check.py` 覆盖。
- 约束与数学定义写入 [op_list.md](../op_list.md)，签名写入 [api_list.md](../api_list.md)。
- 裸 checkout 由 `scripts/local_pkg.py` 挂导入名；验 whl 时置 `OIL_GAS_OPS_WHL_TEST=1`。

步骤见[开发指南 · Triton](operator_development_guide.md#二triton-ascend-算子)。

---

## 6. 构建与打包

### 6.1 流程

```text
msopgen gen (DummyOp IR) → 与当前 CANN/SOC 匹配的脚手架
    → merge_operators.py（按 --ops / manifest 合并）
    → patch_*.py（CMakePresets / vendor / include）
    → CustomOp/build.sh → OPP .run
    → output/<SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run
```

顶层 `bash build.sh` 接着：安装 `.run` → 编 whl（含 pybind）→ 安装 whl → 冒烟。

### 6.2 添加 AscendC 算子后要做的事

1. `operators/<name>/op_host/` + `op_kernel/`（§2.2）。
2. `manifest.tsv` 追加一行。
3. `ascendc/pybind/<family>.py` + `csrc/<family>/bindings.cpp`。
4. ophost UT：改 `tests/ascendc/ophost/CMakeLists.txt`。
5. NPU 对比：新建 `tests/ascendc/<name>/`。
6. 更新 [op_list.md](../op_list.md)、[api_list.md](../api_list.md)、`ascendc/README.md`。
7. `bash ascendc/build.sh --pkg --ops <name>` 与 `bash build.sh -u`。

完整选项见 [build.md](../context/build.md)。

### 6.3 安装与扩展

```bash
# OPP（由顶层 build.sh 或手动安装）
bash output/Ascend910B1/oil_gas_ops_prospect_opp_<os>_<arch>.run --quiet

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib:${LD_LIBRARY_PATH}

# 就地编 pybind（NPU 测试用，不装 whl）
python3 ascendc/pybind_setup.py build_ext --inplace
```

`OIL_GAS_OPS_SKIP_EXT=1` 可跳过扩展编译。CANN 根目录默认 `/usr/local/Ascend/cann-9.0.0`，用 `ASCEND_HOME_PATH` 覆盖。

---

## 7. 测试与精度

### 7.1 分层（本仓库实际结构）

```text
无卡（CI JOB_llt / bash build.sh -u）
  ├─ tests/ascendc/ophost/          googletest：真实 tiling_core
  ├─ tests/triton/<family>/         pytest：CPU 回退精度
  └─ triton-ascend/tools/aot_compile_check.py

有卡（PR 描述必须附结果，门禁不代跑）
  ├─ tests/ascendc/<op>/run_test.sh
  └─ tests/triton/<family>/verify_*.py（如有）
```

不要在 `operators/<op>/` 下再建 `testplan/`、`teste2e/`、`delivery/`。

### 7.2 ophost UT 必须覆盖

- UB 预算与 tile 上限、32B 对齐
- 小 tensor 收缩、空张量、`aivNum=0`、非法 shape / dtype
- 核间切分不变量：每个元素恰好被一个核处理一次，无空洞无重叠
- 融合算子：bias 广播 / 全尺寸、`Q≠S`、VEC / MIX 路径分发

### 7.3 NPU 对比测试

每个 AscendC 算子（或前反向一对）在 `tests/ascendc/<op>/`：

```text
tests/ascendc/<op>/
├── test_<op>_npu.py
├── run_test.sh          # 安装 OPP → 就地编扩展 → 跑脚本
└── README.md
```

基准必须是等价的 PyTorch / torch_npu 表达式。记录相对误差或契约式 `atol/rtol`、时延与加速比、是否原地写回。

### 7.4 精度标准

以[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)为准。本仓已交付算子的验收阈值：

| 算子 | dtype | 阈值 | 出处 |
|------|-------|------|------|
| `ComplexMul` / `Grad` | fp32 | 相对误差默认 `1e-5` | `tests/ascendc/complex_mul/` |
| `FusedBiasSoftmax` / `FusedSoftmaxGrad` | fp32 | `max_diff ≤ 1e-5` | `tests/ascendc/fused_bias_softmax/` |
| 同上 | bf16 | `atol=2e-2, rtol=4e-3` | 同上（量化半 ulp 随幅值增长，不用纯 abs） |

新算子在 PR 中按 [精度验收报告模板](precision_acceptance_template.md) 填写，通过率 100%。严重 / 主要遗留问题未闭环不得合入。

### 7.5 真机性能与内存（推荐）

门禁不跑 NPU。合入前应在 910B 上：

- 精度脚本：预热后多次迭代，`torch.npu.synchronize()`，记 `ms/call`。
- 需要定位瓶颈时用 `msprof`（见 [op_debug_prof.md](../debug/op_debug_prof.md)）。
- 怀疑越界 / 踩踏时用 `mssanitizer --tool=memcheck --leak-check=no --check-device-heap=no`（PyTorch 缓存分配器会误报 leak / device heap）。

### 7.6 文档同步

算子语义或约束变了，必须同步 [op_list.md](../op_list.md)、[api_list.md](../api_list.md)、对应 `README.md`。

---

## 8. PR 产物清单

### 8.1 必须提交

| 类别 | 路径 |
|------|------|
| Host / Kernel | `ascendc/operators/<op>/op_host/`、`op_kernel/` |
| 注册表 | `ascendc/operators/manifest.tsv` 新行 |
| Python API | `ascendc/pybind/<family>.py` |
| aclnn 绑定 | `ascendc/pybind/csrc/<family>/bindings.cpp` |
| ophost UT | `tests/ascendc/ophost/test_<op>_tiling.cpp` + `CMakeLists.txt` |
| NPU 对比 | `tests/ascendc/<op>/`（Triton 则为 `tests/triton/<family>/`） |
| 文档 | `docs/zh/op_list.md`、`api_list.md`；AscendC 还要 `ascendc/README.md` |
| 精度报告 | PR 描述附 [模板](precision_acceptance_template.md) |
| Triton 登记 | `setup.py` 的 `packages` / `package_dir`；AOT 检查覆盖 |

不要求、也不要新增：算子目录内的 `delivery/` SHA256 副本、`testplan/`、`teste2e/`。

### 8.2 不提交

`output/`、`ascendc/_work/`、`build/`、`*.egg-info/`、`*.so`、`__pycache__/`、`.coverage`、`mindstudio_sanitizer_log/`。已在 `.gitignore`。

---

## 9. 审核与迭代

| 优先级 | 含义 | 处理 |
|--------|------|------|
| **P0** | 功能错误、精度超限、crash、梯度断裂 | 阻塞合入 |
| **P1** | 切分不可测、命名违规、静默回退掩盖失败 | 必须修 |
| **P2** | 文档不同步、缺边界用例 | 应修 |
| **P3** | 性能建议 | 可选 |

流程按 [CONTRIBUTING.md](../../../CONTRIBUTING.md)：先 Issue，再 PR；门禁 `compile`；Committer `/lgtm`，Maintainer `/approve`。

---

## 10. 分支与提交

| 类型 | 格式 | 示例 |
|------|------|------|
| 功能 | `feat/<op-name>` | `feat/complex-mul` |
| 修复 | `fix/<op-name>` | `fix/fused-softmax-grad-tiling` |
| 文档 | `docs/<topic>` | `docs/operator-dev-spec` |
| 布局 / 重构 | `refactor/<topic>` | `refactor/opensource-layout` |

Conventional Commits：`feat` / `fix` / `test` / `docs` / `refactor` / `chore`。描述可中文。

PR 标题写清算子或主题；描述含关联 Issue、测试统计、精度报告。CLA 邮箱与 `git config user.email` 一致。

---

## 11. 附录：检查清单

```
□ operators/<op>/op_host + op_kernel（含 tiling_core.h）
□ manifest.tsv 新行；list-ops 能看到
□ pybind/<family>.py：蛇形公开名、autograd、torch 参考、OIL_GAS_OPS_* 开关
□ pybind/csrc/<family>/bindings.cpp（直调 aclnn）
□ tests/ascendc/ophost 已接入；bash build.sh -u 通过
□ tests/ascendc/<op>/ 或 tests/triton/<family>/
□ op_list.md / api_list.md / 相关 README 已更新
□ 许可证头（Apache License 2.0）
□ bash scripts/check_license_header.sh
□ 910B 真机精度 + 性能写入 PR 模板；通过率 100%
□ 无卡门禁（compile / 静态检查 / UT）通过
```

### 环境变量速查

| 变量 | 默认 | 作用 |
|------|------|------|
| `ASCEND_HOME_PATH` | `/usr/local/Ascend/cann-9.0.0` | CANN 根目录 |
| `LD_LIBRARY_PATH` | — | 需含 `opp/vendors/customize/op_api/lib` |
| `OIL_GAS_OPS_<OP>_IMPL` | `torch` | `torch` / `ascendc` / `auto` |
| `OIL_GAS_OPS_REQUIRE_ASCENDC_<OP>` | `0` | `1` 禁止回退 |
| `OIL_GAS_OPS_OUTPUT_DIR` | `<repo>/output` | 产物目录 |
| `OIL_GAS_OPS_SKIP_EXT` | `0` | `1` 跳过 pybind 扩展 |
| `OIL_GAS_OPS_WHL_TEST` | `0` | `1` 时禁止导入仓内源码 |
| `OIL_GAS_OPS_RUN_PKG` | 自动探测 | NPU 测试指定 `.run` 包 |

完整列表见 [api_list.md · 环境变量](../api_list.md#环境变量)。
