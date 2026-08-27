# oil-gas-ops-prospect 算子开发与提交规范

> **适用范围**：本仓库 `ascendc/operators/` 下所有 AscendC 自定义算子的开发、测试、打包和提交。
>
> **版本**：v1.0（基于 e3nn_npu 四轮审核 + 真机验证闭环经验提炼）

---

## 目录

1. [总则](#1-总则)
2. [目录结构规划](#2-目录结构规划)
3. [算子源码开发规范](#3-算子源码开发规范)
4. [Python 封装规范](#4-python-封装规范)
5. [构建与打包](#5-构建与打包)
6. [PR 必须提交的产物清单](#6-pr-必须提交的产物清单)
7. [质量保障体系](#7-质量保障体系)
8. [代码审核与迭代](#8-代码审核与迭代)
9. [分支与提交规范](#9-分支与提交规范)
10. [附录：模板与检查清单](#10-附录模板与检查清单)

---

## 1. 总则

### 1.1 目标

确保每个自定义算子从源码到真机运行的完整闭环质量可追溯、可复现、可审核。

### 1.2 核心原则

- **闭环验证**：源码 → 编译 → 安装 → 真机验证，缺一不可。
- **TDD 先行**：先写测试（RED），再写实现（GREEN），最后重构。
- **审计可追溯**：所有关键文件有 SHA256 指纹，测试有原始输出证据。
- **CPU/NPU 双环境**：CPU 环境跑全量逻辑覆盖，NPU 环境跑精度+性能+内存安全。

### 1.3 OpType 命名约定

| 规则 | 示例 |
|------|------|
| 自研算子加 `Og`（oil & gas）后缀避免与 CANN 内置冲突 | `Conv3DGeluOg` |
| 或使用 `Custom` 前缀 | `CustomGRUV2` |
| 数学算子使用语义化命名 | `SoftUnitStep`, `ShLmax6` |
| 反向算子加 `Backward` 后缀 | `SoftUnitStepBackward` |

### 1.4 支持的 SoC

当前支持 `Ascend910B1` / `Ascend910B2`。新增 SoC 需在 `ascendc/scripts/config.py` 的 `SUPPORTED_SOC_VERSIONS` 中注册，并在 OpDef 的 `AddConfig()` 中添加对应配置。

---

## 2. 目录结构规划

### 2.1 仓库顶层结构

```
oil-gas-ops-prospect/
├── ascendc/                        # AscendC 算子（本规范主体）
│   ├── build.py                    # 构建入口
│   ├── operators/                  # 算子源码（每个算子一个子目录）
│   ├── scripts/                    # 构建系统 Python 模块
│   ├── tests/                      # C++ Plugin / aclnn 集成测试
│   └── dist/                       # 构建产物（.gitignore）
├── python/ai4fwi_ops/              # 统一 Python 包
│   └── ascendc/                    # 每个算子一个 .py 封装
├── triton-ascend/                  # Triton-Ascend 算子（独立规范）
├── tests/                          # whl 级别冒烟测试
├── build_install_ops.sh            # 一键构建+安装+测试
├── setup.py                        # Python 扩展编译
└── pyproject.toml
```

### 2.2 单个算子目录结构（标准模板）

以 `operators/<op_name>/` 为根，**按以下结构组织**：

```
operators/<op_name>/
├── op_host/                        # 【必须】Host 侧代码
│   ├── <op>_def.cpp                #   OpDef 注册（ops::OpDef + OP_ADD）
│   ├── <op>_tiling.cpp             #   Tiling 实现（IMPL_OP_OPTILING）
│   ├── <op>_tiling.h               #   Tiling 数据结构（BEGIN_TILING_DATA_DEF）
│   └── extra.cmake                 #   【可选】自定义 CMake 片段（如 aclnn L2 集成）
│
├── op_kernel/                      # 【必须】Device 侧代码
│   ├── <op>.cpp                    #   Kernel 入口（extern "C" __global__ __aicore__）
│   ├── <op>_tiling.h               #   Tiling 结构的 device 侧镜像（plain struct）
│   ├── <op>_kernel.h               #   【可选】共享 Kernel 类（多个 OpType 复用时）
│   └── <op>_coefficients.h         #   【可选】自动生成的常量表等
│
├── scripts/                        # 【可选】代码生成脚本
│   └── gen_<xxx>.py                #   生成 header / 表格等
│
├── testplan/                       # 【必须】测试计划与执行
│   ├── README.md                   #   测试概览
│   ├── test_plan.md                #   详细测试计划（用例表）
│   ├── test_evidence.md            #   测试证据（原始输出、git log、TDD 记录）
│   ├── conftest.py                 #   pytest 公共 fixture
│   ├── pytest.ini                  #   pytest 配置
│   ├── requirements.txt            #   测试依赖
│   ├── run_all_tests.sh            #   一键运行全量测试
│   ├── .gitignore                  #   忽略 .coverage, mindstudio_sanitizer_log/ 等
│   ├── unit/                       #   单元测试
│   ├── integration/                #   集成测试
│   ├── performance/                #   性能测试
│   └── coverage/                   #   覆盖率 + 工具测试脚本
│       ├── run_coverage.py         #     Python 覆盖率
│       ├── run_sanitizer.sh        #     msSanitizer memcheck
│       ├── sanitizer_test.py       #     memcheck 用的精简脚本
│       ├── run_prof.sh             #     msProf 性能采集
│       └── msopst_*.json           #     msOpST 测试用例
│
├── teste2e/                        # 【必须】真机端到端验证闭环
│   ├── README.md                   #   验证报告（含环境、结果、分析）
│   ├── run_e2e.sh                  #   一键复现脚本
│   ├── e2e_npu_verify.py           #   Python API 功能+性能验证
│   ├── e2e_memcheck.py             #   msSanitizer/msProf 用的精简脚本
│   ├── e2e_npu_verify_output.txt   #   功能验证原始输出
│   ├── mssanitizer_memcheck_output.txt  # msSanitizer 原始输出
│   ├── msopst/                     #   msOpST 用例 JSON
│   └── msprof/                     #   msProf 性能数据 CSV
│       ├── <kernel_name>/
│       │   ├── OpBasicInfo.csv
│       │   ├── PipeUtilization.csv
│       │   ├── ArithmeticUtilization.csv
│       │   └── Memory.csv
│
└── delivery/                       # 【必须】审核交付包
    ├── README.md                   #   SHA256 指纹清单 + PR 链接
    ├── <python_wrapper>.py         #   Python 封装的审核副本
    ├── bindings.cpp                #   C++ Plugin 的审核副本
    └── setup_<op>.py               #   setup.py 扩展配置的审核副本
```

### 2.3 一个目录注册多个 OpType

当多个 OpType 共享同一算子目录时（如 `e3nn_npu/` 注册 `ShLmax6`、`SoftUnitStep`、`SoftUnitStepBackward`），在 `manifest.tsv` 中每个 OpType 占一行，指向各自的 kernel 入口文件：

```tsv
# 目录名	OpType(CANN)	kernel入口文件名
e3nn_npu	ShLmax6	sh_lmax6.cpp
e3nn_npu	SoftUnitStep	soft_unit_step.cpp
e3nn_npu	SoftUnitStepBackward	soft_unit_step_backward.cpp
```

共享的 kernel 逻辑放在 `<op>_kernel.h` 中，各 OpType 的入口 `.cpp` 分别 `#include` 它。

---

## 3. 算子源码开发规范

### 3.1 OpDef 注册（op_host/）

使用 rt2.0 的 `ops::OpDef` + `OP_ADD` 模式（非旧的 `REG_OP`）：

```cpp
#include "register/op_def_registry.h"

namespace ops {
class MyOp : public OpDef {
public:
    explicit MyOp(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(false)
                .DynamicFormatFlag(false)
                .DynamicRankSupportFlag(false)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .ExtendCfgInfo("opFile.value", "my_op")       // 指向 op_kernel/my_op.cpp
                .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", aicConfig);
    }
};
OP_ADD(MyOp);
}  // namespace ops
```

**关键约定**：
- `opFile.value` 必须指向 `op_kernel/` 下的入口文件名（不含扩展名）。
- `aclnnSupport.value` 设为 `"support_aclnn"` 以让 opbuild 自动生成 aclnn API。
- 每个 OpType 单独一个 `_def.cpp` 文件（不要合并多个 OpDef 到一个文件）。

### 3.2 Tiling 实现（op_host/）

```cpp
// op_host/my_op_tiling.h
BEGIN_TILING_DATA_DEF(MyOpTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(MyOp, MyOpTilingData)
```

```cpp
// op_host/my_op_tiling.cpp
namespace optiling {
static ge::graphStatus TilingMyOp(gert::TilingContext* context) {
    auto tiling = context->GetTilingData<MyOpTilingData>();
    // ... 计算 tiling 参数 ...
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) coreNum = 24;  // 兜底
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(MyOp).Tiling(TilingMyOp);
}  // namespace optiling
```

**关键约定**：
- Tiling 数据结构必须在 `op_host/` 用 `BEGIN_TILING_DATA_DEF` 声明（Host 编译器需要）。
- 在 `op_kernel/` 中用 **plain struct** 镜像同一结构（Device 编译器 `ccec` 不识别 `BEGIN_TILING_DATA_DEF`）。
- `GetCoreNumAiv()` 返回 0 时必须有兜底值。

### 3.3 Kernel 入口（op_kernel/）

```cpp
// op_kernel/my_op_tiling.h — device 侧镜像
struct MyOpTilingData {
    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t lastTileLength;
};

// op_kernel/my_op.cpp — 入口
extern "C" __global__ __aicore__ void my_op(GM_ADDR x, GM_ADDR out, GM_ADDR tiling) {
    auto* t = (__gm__ MyOpTilingData*)tiling;
    KernelMyOp op;
    op.Init(x, out, t->totalLength, t->tileLength, t->lastTileLength);
    op.Process();
}
```

### 3.4 Ascend910B 向量操作 64 元素对齐（重要）

Ascend910B 的向量指令以 **64 个元素为单位**执行。当 `count` 非 64 倍数时，尾部元素会被跳过输出 0。

**必须使用以下模式**：

```cpp
uint32_t aligned = (count + 63u) & ~63u;

// 向量操作使用 aligned（覆盖尾部元素，多出的读垃圾但不影响输出）
CompareScalar(mask, x, 0.0f, CMPMODE::GT, aligned);
Reciprocal(t0, safeX, aligned);
Exp(t1, t0, aligned);

// DataCopyPad 仍使用原始 count（精确拷贝，不越界写）
DataCopyPad(outGm_[offset], out, outParams);  // blockLen = count * sizeof(float)
```

### 3.5 共享 Kernel 类

当 forward 和 backward 共享计算逻辑时，抽取为共享类：

```cpp
// op_kernel/my_op_kernel.h
class KernelMyOp {
public:
    __aicore__ inline void InitForward(...) { isBackward_ = false; ... }
    __aicore__ inline void InitBackward(...) { isBackward_ = true; ... }
    __aicore__ inline void Process() {
        if (isBackward_) ProcessTileBackward(...);
        else ProcessTileForward(...);
    }
private:
    // ... TQue, TBuf, GlobalTensor 成员 ...
    bool isBackward_;
};
```

### 3.6 C++ Plugin / Pybind 绑定（ascendc/tests/）

当 op-plugin 的 `EXEC_NPU_CMD_SYNC` 宏不可用（依赖未导出符号）时，使用手动两段式 aclnn 调用：

```cpp
// ascendc/tests/<op_name>/cpp/bindings.cpp
// 1. GetWorkspaceSize (Host)
auto workspaceSizeFunc = GetOpApiFuncAddr("aclnnMyOpGetWorkspaceSize");
auto launcherFunc = GetOpApiFuncAddr("aclnnMyOp");
// 2. Launch (Device)
aclnnMyOpGetWorkspaceSize(..., &workspaceSize, &executor);
aclnnMyOp(workspaceAddr, workspaceSize, executor, stream);
```

`setup.py` 中为每个算子定义 `CppExtension`，链接 `["cust_opapi", "ascendcl", "nnopbase", "opapi", "torch_npu"]`。

---

## 4. Python 封装规范

### 4.1 文件位置

每个算子的 Python 封装放在 `python/ai4fwi_ops/ascendc/<op_name>.py`。

### 4.2 标准封装模式

```python
from __future__ import annotations
import importlib, os
from functools import lru_cache
import torch, torch.nn as nn

@lru_cache(maxsize=1)
def _load_extension():
    return importlib.import_module("<op_name>_npu")

def _require_custom() -> bool:
    return os.environ.get("OILGAS_REQUIRE_ASCENDC_<OP_NAME>", "0") == "1"

def _is_npu_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "npu"

# ── 纯 PyTorch 参考（CPU fallback）──
def _my_op_pytorch(x: torch.Tensor) -> torch.Tensor:
    ...

# ── autograd.Function ──
class _MyOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ext = _load_extension()
        return ext.my_op_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        ext = _load_extension()
        return ext.my_op_backward(grad_output, x)

# ── 函数式 API（必须路由到 autograd.Function.apply）──
def my_op(x: torch.Tensor) -> torch.Tensor:
    if not _is_npu_tensor(x):
        if _require_custom():
            raise RuntimeError("...")
        return _my_op_pytorch(x)
    try:
        _load_extension()
        return _MyOpFunction.apply(x)          # ← 不能直接调用 ext.my_op_forward()
    except ImportError:
        if _require_custom():
            raise RuntimeError("...")
        return _my_op_pytorch(x)

# ── nn.Module 包装 ──
class MyOp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _is_npu_tensor(x):
            try:
                _load_extension()
                return _MyOpFunction.apply(x)
            except ImportError:
                pass
        return _my_op_pytorch(x)
```

### 4.3 关键规则

| 规则 | 说明 |
|------|------|
| **函数式 API 必须路由到 `autograd.Function.apply()`** | 直接调用 `ext.forward()` 会绕过 autograd 引擎，梯度图断裂 |
| **CPU fallback 必须独立实现** | 不能依赖 NPU 扩展 |
| **`_require_custom()` 环境变量** | 设为 `"1"` 时强制走 NPU 路径，CPU/导入失败则 raise |
| **`@lru_cache` 缓存扩展模块** | 避免重复 import |
| **`from __future__ import annotations`** | Python 3.9 兼容 `X | None` 类型注解 |

---

## 5. 构建与打包

### 5.1 构建流程

```
msopgen gen (DummyOp IR) → CANN 脚手架
    ↓
merge_operators.py → 合并多算子源码到脚手架
    ↓
patch_*.py → 补丁 CMakePresets / vendor_name / include 路径
    ↓
CustomOp/build.sh → 编译 kernel .o + 打包 .run
    ↓
patch_package.py → 重命名 custom_*.run → oil_gas_*.run
    ↓
输出 dist/<SOC>/oil_gas_opp_*.run
```

### 5.2 添加新算子

1. 创建 `operators/<name>/op_host/` + `op_kernel/`（遵循 §2.2 结构）。
2. 在 `operators/manifest.tsv` 追加一行（tab 分隔）。
3. 在 `setup.py` 中添加 `CppExtension` 定义。
4. 运行 `python3 ascendc/build.py -v Ascend910B2 --ops <name>`。
5. 运行 `bash build_install_ops.sh --soc Ascend910B2 --ops <name>` 完成安装+冒烟测试。

### 5.3 .run 包安装

```bash
# 安装 .run 包到 CANN OPP vendor
bash ascendc/dist/Ascend910B2/oil_gas_opp_bclinux_aarch64.run --install

# 验证安装
ls /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/<op_name>/
# 应看到 *.o + *.json
```

### 5.4 Python 扩展编译

```bash
# 编译 pybind 扩展
python3 setup.py build_ext --inplace

# 或通过 build_install_ops.sh 自动完成
```

### 5.5 构建系统注意事项

- `ascendc/scripts/config.py` 中 `from __future__ import annotations` 是 Python 3.9 兼容必需。
- `merge_operators.py` 中 kernel 文件直接复制到 `op_kernel/`（非子目录）。
- 使用 `add_kernels_compile()` API（非旧版 `npu_op_kernel_sources`）。

---

## 6. PR 必须提交的产物清单

### 6.1 必须提交

| 类别 | 路径 | 说明 |
|------|------|------|
| **OpDef** | `op_host/<op>_def.cpp` | 算子定义注册 |
| **Tiling** | `op_host/<op>_tiling.cpp` + `.h` | Tiling 实现 |
| **Kernel** | `op_kernel/<op>.cpp` + `_tiling.h` | Kernel 入口 + tiling 镜像 |
| **manifest** | `operators/manifest.tsv` | 追加新算子行 |
| **Python 封装** | `python/ai4fwi_ops/ascendc/<op>.py` | 函数式 API + nn.Module |
| **C++ Plugin** | `ascendc/tests/<op>/cpp/bindings.cpp` | pybind 绑定 |
| **setup.py** | `setup.py` | 追加 CppExtension |
| **测试计划** | `testplan/test_plan.md` | 详细用例表 |
| **测试证据** | `testplan/test_evidence.md` | 原始输出 + git log |
| **conftest.py** | `testplan/conftest.py` | 统一 `HAS_NPU_HW` 判定 |
| **单元测试** | `testplan/unit/test_*.py` | 精度 + 边界 + 覆盖路径 |
| **集成测试** | `testplan/integration/test_*.py` | 端到端场景 |
| **性能测试** | `testplan/performance/bench_*.py` | NPU vs CPU 基准 |
| **运行脚本** | `testplan/run_all_tests.sh` | 一键全量测试 |
| **E2E 验证** | `teste2e/` | 真机验证闭环（见 §7.4） |
| **交付包** | `delivery/` | SHA256 指纹 + 审核副本 |
| **delivery/README.md** | `delivery/README.md` | 指纹清单 + PR 链接 |

### 6.2 不提交（.gitignore）

| 路径 | 原因 |
|------|------|
| `build/`, `dist/`, `*.egg-info/` | 构建产物 |
| `ascendc/_work/` | msopgen 生成脚手架 |
| `*.so` | 编译后的共享库 |
| `__pycache__/`, `*.pyc` | Python 缓存 |
| `mindstudio_sanitizer_log/` | msSanitizer 日志 |
| `.coverage`, `.pytest_cache/` | 测试缓存 |
| `*.o` | 编译后的 kernel 二进制 |

### 6.3 delivery/ 审核交付包规范

`delivery/` 目录存放**审核副本**——原始文件位于仓库其他位置的拷贝，方便审核者在一个目录内完成审查。

`delivery/README.md` 必须包含：

```markdown
# <OpName> 审核交付包

## PR 链接
https://gitcode.com/cann/oil-gas-ops-prospect/pulls/<N>

## 分支
feat/<op-name>

## 文件指纹

| 文件 | 仓库原始路径 | SHA256 | 说明 |
|------|-------------|--------|------|
| <op>.py | python/ai4fwi_ops/ascendc/<op>.py | <sha256> | Python 封装 |
| bindings.cpp | ascendc/tests/<op>/cpp/bindings.cpp | <sha256> | C++ Plugin |
| setup_<op>.py | setup.py (excerpt) | <sha256> | 编译配置 |

> 审核副本，修改请到原始路径。
```

SHA256 通过 `sha256sum <file>` 计算，在 squash commit 后重新计算并更新。

---

## 7. 质量保障体系

### 7.1 测试分层架构

```
┌─────────────────────────────────────────────────┐
│  E2E 验证闭环 (teste2e/)                        │
│  msOpST + msSanitizer + msProf + Python API    │
├─────────────────────────────────────────────────┤
│  集成测试 (testplan/integration/)                │
│  端到端推理/训练场景                              │
├─────────────────────────────────────────────────┤
│  性能测试 (testplan/performance/)                │
│  NPU vs CPU 加速比 + 时延基准                     │
├─────────────────────────────────────────────────┤
│  单元测试 (testplan/unit/)                       │
│  精度 + 边界 + 数值稳定性 + 路径覆盖               │
└─────────────────────────────────────────────────┘
```

### 7.2 单元测试规范

#### 7.2.1 用例 ID 约定

| 前缀 | 含义 | 示例 |
|------|------|------|
| `UT-<CAT>-NNN` | 单元测试 | `UT-PREC-001`, `UT-EDGE-003` |
| `IT-<CAT>-NNN` | 集成测试 | `IT-E2E-001` |
| `PERF-<CAT>-NNN` | 性能测试 | `PERF-SH-004` |

#### 7.2.2 必须覆盖的测试类别

| 类别 | 说明 | 必须 |
|------|------|------|
| **精度测试** | NPU vs CPU 参考实现，max_diff < 阈值 | 是 |
| **边界测试** | N=0, N=1, N=64（对齐）, N=257（非对齐） | 是 |
| **数值稳定性** | 极大/极小输入, NaN, Inf | 是 |
| **路径覆盖** | `grad_fn` 路由, CPU fallback, `require_custom` 环境变量 | 是 |
| **形状覆盖** | 1D/2D/3D, 对齐/非对齐 | 是 |
| **autograd** | `torch.autograd.gradcheck` (CPU float64) | 是 |

#### 7.2.3 NPU 精度容差约定

| 算子类型 | 阈值 | 原因 |
|---------|------|------|
| 简单元素级 (SUS fwd/bwd) | `max_diff ≤ 2e-3` | float32 Reciprocal + Exp 硬件精度 |
| 高阶多项式 (SH lmax=6) | `rtol=1e-3, atol=1e-4`（标准输入）或 `max_diff ≤ 5e-2`（放大输入） | r^6 项放大舍入误差 |
| DataCopyPad 对齐 | `rtol=5e-3, atol=1e-3` | 尾部 padding 元素噪声 |

#### 7.2.4 conftest.py 规范

```python
def _try_import_torch_npu():
    try:
        import torch_npu  # noqa: F401
        return True
    except ImportError:
        return False

HAS_TORCH_NPU = _try_import_torch_npu()
HAS_NPU_HW = HAS_TORCH_NPU and torch.npu.is_available()

def npu_only():
    return pytest.mark.skipif(not HAS_NPU_HW, reason="NPU hardware required")

def get_device():
    return torch.device("npu:0") if HAS_NPU_HW else torch.device("cpu")
```

**关键**：使用统一的 `HAS_NPU_HW` 标志，不要在不同位置用不同的判定逻辑。

### 7.3 测试文档规范

#### 7.3.1 test_plan.md

必须包含以下章节：

1. **测试范围**：被测对象表 + 不测试范围
2. **测试分层**：单元/集成/性能，每层列出用例表（ID | 描述 | 输入 | 预期 | 方法）
3. **代码覆盖率策略**：Python 100%（CPU+NPU 合并）/ 80%（仅 CPU），C++ Kernel 100%
4. **测试环境**：SoC、CANN、torch、torch_npu、Python 版本
5. **通过/失败标准**
6. **测试统计**：总用例数、通过数、跳过数

#### 7.3.2 test_evidence.md

必须包含以下章节：

1. **Git 证据**：`HEAD` commit, 基准分支, `git status --porcelain`, squash 前 commit 历史
2. **TDD RED→GREEN 原始输出**：先让测试失败的输出，再修复后的输出
3. **CPU 环境测试输出**：`pytest -v` 完整输出
4. **NPU 环境测试输出**：`pytest -v` 完整输出 + 设备信息
5. **msSanitizer 原始输出**
6. **PR 链接**
7. **待补充证据**（如有）

### 7.4 真机 E2E 验证闭环（teste2e/）

#### 7.4.1 验证闭环流程

```
源码 (op_kernel/*.h, op_host/*.cpp)
  ↓ opbuild 编译
编译产物 (*.o kernel 二进制 + op_info.json)
  ↓ .run 包安装
安装到 OPP vendor (customize)
  ↓ 四工具验证
┌──────────────────────────────────────────┐
│ 1. Python API 功能验证 (21+ 用例)         │
│    torch_npu → pybind → aclnn → kernel.o │
├──────────────────────────────────────────┤
│ 2. msSanitizer memcheck (内存安全)        │
│    --leak-check=no --check-device-heap=no │
├──────────────────────────────────────────┤
│ 3. msProf (性能 profiling)                │
│    PipeUtilization/ArithUtil/Memory CSV   │
├──────────────────────────────────────────┤
│ 4. msOpST (ATC 编译 pipeline)             │
│    ATC → 代码生成 → 编译 → 真机执行        │
└──────────────────────────────────────────┘
```

#### 7.4.2 验证脚本要求

| 脚本 | 作用 | 关键点 |
|------|------|--------|
| `e2e_npu_verify.py` | 功能 + 性能 | 覆盖对齐/非对齐 shape，记录 max_diff 和 us/call |
| `e2e_memcheck.py` | msSanitizer/msProf 精简脚本 | 只做 kernel launch + sync，不做精度比较 |
| `run_e2e.sh` | 一键复现 | 4 阶段串联，每阶段独立 PASS/FAIL |

#### 7.4.3 msSanitizer 使用规范

```bash
mssanitizer --tool=memcheck \
    --leak-check=no \
    --check-device-heap=no \
    python3 e2e_memcheck.py
```

- **必须** `--leak-check=no`：PyTorch 缓存分配器会触发误报。
- **必须** `--check-device-heap=no`：同上。
- 通过标准：exit=0, 0 errors。
- WARN（缺 debug_line 信息）不影响通过。

#### 7.4.4 msProf 使用规范

```bash
msprof op \
    --application="python3 e2e_memcheck.py" \
    --aic-metrics=PipeUtilization,ArithmeticUtilization,Memory,Occupancy \
    --kernel-name=<KernelName> \
    --launch-count=1
```

- 每个 kernel 单独 profile（`--kernel-name` 过滤）。
- 输出 CSV 归档到 `teste2e/msprof/<kernel_name>/`。
- 性能报告中需包含：Task Duration、Block Dim、vec_ratio、scalar_ratio、mte2/mte3 bandwidth。

#### 7.4.5 msOpST 使用规范

```bash
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize
export IGNORE_INFER_ERROR=1

msopst run -i st_<op>.json -soc Ascend910B2 -out <output> -err_thr "[0.003,0.003]"
```

- ATC 编译阶段验证算子定义和 tiling 代码的正确性。
- 设备执行阶段可能因 ATC 动态编译 kernel 与预编译二进制不一致而失败——这是已知限制，不阻塞 PR。
- Python impl 模块路径需在 PYTHONPATH 中配置（`customize_impl/` + built-in `impl/`）。

### 7.5 性能测试规范

#### 7.5.1 基准测试要求

- 5 次 warmup + 100 次迭代取平均。
- `torch.npu.synchronize()` 确保同步。
- 记录 `us/call`。
- NPU vs CPU 加速比阈值在测试中设定。

#### 7.5.2 msProf 性能分析建议

| 指标 | 优化建议 |
|------|---------|
| MTE2/MTE3 bandwidth < 80% | memory-bound，考虑增大 TILE_SIZE |
| aivector compute < 20% | 计算密度低，瓶颈在访存 |
| scalar_ratio > 50% | tiling 循环开销大，考虑减少 tile 数量 |
| icache_miss_rate > 5% | kernel 代码过大，考虑拆分 |

---

## 8. 代码审核与迭代

### 8.1 审核轮次约定

| 轮次 | 审核重点 | 修复 commit 消息 |
|------|---------|-----------------|
| Round 1 | P0 功能正确性 + P1 代码质量 | `fix: 评审P0-P2修复` |
| Round 2 | P2 残留 + 文档同步 | `fix: P2残留问题修复` |
| Round 3 | 边界条件 + 性能 | `test: 添加代码路径覆盖测试` |
| Round 4 | autograd 路由 + kernel 对齐 + 构建系统 | `fix: Round-4审核修复` |

### 8.2 问题优先级

| 优先级 | 含义 | 处理 |
|--------|------|------|
| **P0** | 功能错误（梯度断裂、精度超限、crash） | 必须修复，阻塞 PR |
| **P1** | 代码质量（命名、注释、容差不合理） | 必须修复 |
| **P2** | 文档、风格、非关键测试缺失 | 应修复 |
| **P3** | 优化建议 | 可选 |

### 8.3 审核修复流程

1. 审核者提出意见，标注 P0/P1/P2。
2. 开发者修复，commit 消息引用审核轮次。
3. 修复后全量测试（`run_all_tests.sh`）必须通过。
4. `test_evidence.md` 补充本轮修复的测试输出。
5. `delivery/README.md` 更新 SHA256 指纹（如文件有变更）。
6. 所有审核轮次完成后，squash 为单一 commit。

### 8.4 squash 前保留历史

squash 前的分批 commit 历史必须记录到 `test_evidence.md` §1.5：

```markdown
### 1.5 squash 前分批 commit 历史

```
877d428 fix: 评审P0-P2修复
90a7511 fix: P2残留问题修复
c5bfd20 test: 添加代码路径覆盖测试
9cb4a42 feat: add e3nn_npu operators
```
```

---

## 9. 分支与提交规范

### 9.1 分支命名

| 类型 | 格式 | 示例 |
|------|------|------|
| 功能分支 | `feat/<op-name>` | `feat/e3nn-npu` |
| 修复分支 | `fix/<op-name>-<round>` | `fix/e3nn-npu-round4` |
| 文档分支 | `docs/<topic>` | `docs/operator-dev-spec` |

### 9.2 Commit 消息

使用 Conventional Commits：

```
<type>: <description>
```

| type | 用途 |
|------|------|
| `feat:` | 新增算子或功能 |
| `fix:` | 修复审核问题或 bug |
| `test:` | 新增或修改测试 |
| `docs:` | 文档变更 |
| `refactor:` | 重构（无功能变更） |
| `chore:` | 构建/配置变更 |

描述可中文可英文，审核修复需标注轮次：
```
fix: Round-4审核修复 — autograd路由+SUS kernel 64元组对齐+构建系统+测试容差
```

### 9.3 PR 要求

- PR 标题：`feat: add <OpType> operator` 或 `fix: <description>`
- PR 描述包含：算子功能概述、测试统计（N passed, 0 failed）、PR 链接
- 最终 squash 为单一 commit
- CLA 签署邮箱与 git config user.email 一致

---

## 10. 附录：模板与检查清单

### 10.1 PR 提交前检查清单

```
□ op_host/<op>_def.cpp     — OpDef 注册（ops::OpDef + OP_ADD）
□ op_host/<op>_tiling.cpp  — Tiling 实现
□ op_host/<op>_tiling.h    — Tiling 数据结构
□ op_kernel/<op>.cpp       — Kernel 入口
□ op_kernel/<op>_tiling.h  — Tiling device 镜像（plain struct）
□ manifest.tsv             — 追加新行
□ python/ai4fwi_ops/ascendc/<op>.py — Python 封装（autograd.Function.apply 路由）
□ ascendc/tests/<op>/cpp/bindings.cpp — C++ Plugin
□ setup.py                 — CppExtension
□ testplan/test_plan.md    — 测试计划
□ testplan/test_evidence.md — 测试证据（含 git log + TDD + 原始输出）
□ testplan/conftest.py     — 统一 HAS_NPU_HW
□ testplan/unit/           — 精度+边界+稳定性+路径覆盖
□ testplan/integration/    — 端到端场景
□ testplan/performance/    — NPU vs CPU 基准
□ testplan/run_all_tests.sh — 一键测试
□ teste2e/                 — 真机验证闭环（4 工具）
□ delivery/                — SHA256 指纹 + 审核副本
□ .gitignore               — 新增产物目录已忽略
□ 全量测试通过             — run_all_tests.sh: 0 failed
□ 真机 E2E 通过            — run_e2e.sh: Python API + memcheck PASS
□ squash 前历史已记录       — test_evidence.md §1.5
```

### 10.2 精度容差速查表

| 场景 | 阈值 | 算子示例 |
|------|------|---------|
| float32 元素级 (Reciprocal + Exp) | `max_diff ≤ 2e-3` | SUS forward/backward |
| float32 高阶多项式 (r^6) | `rtol=1e-3, atol=1e-4` | ShLmax6 |
| float32 DataCopyPad 对齐 | `rtol=5e-3, atol=1e-3` | DataCopy alignment |
| float64 gradcheck (CPU) | `atol=1e-6` | 所有算子 autograd |
| NPU 加速比下限 | `ratio ≥ 1.5`（小 N）或 `ratio ≥ 3.0`（大 N） | 性能基准 |

### 10.3 环境变量速查

| 变量 | 作用 | 示例值 |
|------|------|--------|
| `LD_LIBRARY_PATH` | 加载 cust_opapi.so | `.../opp/vendors/customize/op_api/lib` |
| `PYTHONPATH` | 加载 ai4fwi_ops + op_impl | `python/:...:op_impl/ai_core/tbe` |
| `OILGAS_REQUIRE_ASCENDC_<OP>` | 强制 NPU 路径 | `"1"` |
| `DDK_PATH` | msOpST CMake include 路径 | `/usr/local/Ascend/ascend-toolkit/latest` |
| `NPU_HOST_LIB` | msOpST CMake lib 路径 | `.../runtime/lib64/stub` |
| `ASCEND_CUSTOM_OPP_PATH` | ATC 查找 vendor 算子 | `.../opp/vendors/customize` |
| `IGNORE_INFER_ERROR` | 跳过 ATC InferShape 检查 | `"1"` |
