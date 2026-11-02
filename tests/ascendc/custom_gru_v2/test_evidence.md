# CustomGRU v2 测试证据文档

> 本文档响应第四轮审核放行条件，修复 P0-3（环境陈述分离）和 P1-8（TDD 证据）。

## 1. Git 三项原始回显

### 1.1 `git rev-parse HEAD`

```
f7b953cca1a5487c289318268b1994c758d3fd83
```

### 1.2 `git rev-parse origin/master`

```
2d8eb04f75a6e02c64c890f736ae1fae665f1e09
```

### 1.3 `git status --porcelain`

```
 M tests/ascendc/custom_gru_v2/pack_submission.sh
 M tests/ascendc/custom_gru_v2/test_evidence.md
?? tests/ascendc/custom_gru_v2/.coverage
```

> `.coverage` 为覆盖率运行产物，非源代码。`pack_submission.sh` 和 `test_evidence.md` 的改动将在本次提交中包含。

## 2. 分支归并状态

- **当前分支**: `feat/custom-gru-v2`
- **基准分支**: `origin/master` (commit `2d8eb04`)
- **交付阶段**: **中期交付，未合入 main/master**

### 分批 commit 列表（origin/master..HEAD，共 23 个 commit）

| # | Commit | 说明 | 修复项 |
| --- | --- | --- | --- |
| 1 | `a1eebfe` | Initial commit | — |
| 2 | `c7242da` | Add AI4FWI AscendC and Triton custom operator package | — |
| 3 | `f2aab4a` | feat: add CustomGRUV2 fused kernel operator with comprehensive QA testplan | — |
| 4 | `9060b75` | fix: 评审P0-P2修复 + API一致性 + 验证问题修复 | R1-P0×6 |
| 5 | `c280522` | fix: P2残留问题修复 | R1-P2 |
| 6 | `31cec36` | test: 添加代码路径覆盖测试 + conftest.py 修复 | R1-P1 |
| 7 | `67a26f7` | fix: 第二轮审核修复 | R2-P0×2, R2-P1×3 |
| 8 | `71bb275` | fix: P0-遗留+P1-新-4+P1-遗留 | R2-P0遗留 |
| 9 | `4b3f891` | test: 新增手写GRU反向公式gradcheck验证 | R2-P1 |
| 10 | `9a3985b` | fix: UT-BWD-009改为v1精确错误公式 | R2-P1 |
| 11 | `6ed5aff` | fix: 第三轮审核修复 | R3-P0×2, R3-P1×2 |
| 12 | `450b9a8` | fix: npu_hardware_only 装饰器 | R3-P1 |
| 13 | `796de49` | fix: 性能测试 retain_graph + 超时优化 | R3-P1 |
| 14 | `c28a35c` | fix(perf): skip ratio assertion when no compiled NPU kernel | R3-P1 |
| 15 | `7552774` | fix: import pytest in test_e2e_inference | R3-P1 |
| 16 | `e60f11a` | fix(coverage): detect compiled extension before targeting 100% | R3-P1 |
| 17 | `c08d813` | docs: add test_evidence.md | R3-P0 |
| 18 | `003d528` | fix(build): CANN 8.5.0 compatibility | R3-P0 |
| 19 | `531d47f` | fix(npu): backward bias=None + 2D input + NPU precision threshold | R3-P1 |
| 20 | `eb49324` | test(npu): msOpST 5/5 + msSanitizer + msProf results | R3-P0 |
| 21 | `6e99e01` | Merge remote-tracking branch 'origin/master' | 冲突解决 |
| 22 | `f7b953c` | fix(round4): delivery/ + npu_only + requires_compiled_kernel + PipeBarrier + host validation | R4-P0-1, R4-P1-1~4, R4-P2-1~2 |
| 23 | (本次) | docs: update test_evidence.md + pack_submission.sh | R4-P0-2, R4-P0-3, R4-P1-8 |

> 提交者邮箱: `slamdunk111@163.com`（全部 commit）

## 3. `pytest --collect-only -q` 完整输出

```
95 tests collected in 0.11s
```

> 较第三轮新增 4 个 TDD 测试（`test_tdd_round4.py`），总数 91 → 95。

### 收集数交叉核对

| 来源 | 数量 |
| --- | --- |
| `pytest --collect-only -q` | **95** |
| 包内 `def test_` grep | **95** |

## 4. CPU 环境测试运行（commit `e60f11a`，无编译扩展）

> **以下结果来自 commit `e60f11a`，在该 commit 上 `custom_gru_v2_npu` 扩展未编译安装。**
> **torch 版本为 `2.7.1+cpu`，torch_npu 可导入但 `torch.npu.is_available()=False`。**
> **此节的测试在 CPU fallback 模式下运行，验证 fallback 正确性。**

### [CPU-1/4] Unit Tests — 73 passed

```
============================= 73 passed in 22.53s ==============================
```

### [CPU-2/4] Integration Tests — 10 passed, 1 skipped

```
======================= 10 passed, 1 skipped in 15.86s ========================
```

> `test_it_infer_006_npu_graph_capture` SKIPPED：无编译好的 NPU 扩展。

### [CPU-3/4] Performance Tests — 7 passed

```
======================= 7 passed in 360.08s (0:06:00) =========================
```

> PERF ratio 断言在无编译扩展时正确跳过（fallback == reference）。

### [CPU-4/4] Coverage — 83 passed, 1 skipped, 85% (target: 80%)

```
TOTAL                                                                    117     18    85%
[coverage] No NPU — running CPU-only coverage
```

### CPU 环境汇总

| 段 | 收集 | 通过 | 跳过 | 失败 |
| --- | --- | --- | --- | --- |
| Unit | 73 | 73 | 0 | 0 |
| Integration | 11 | 10 | 1 | 0 |
| Performance | 7 | 7 | 0 | 0 |
| Coverage | 84 | 83 | 1 | 0 |
| **合计** | **91** | **90** | **1** | **0** |

## 5. NPU 环境测试运行（commit `f7b953c`，编译扩展已安装）

> **以下结果来自 commit `f7b953c`（第四轮修复后），在 NPU 910B2 / CANN 8.5.0 环境运行。**
> **`custom_gru_v2_npu` 扩展已编译安装，`torch.npu.is_available()=True`。**
> **本轮新增 4 个 TDD 测试，总数 95。**

### [NPU-1/4] Unit Tests — 77 passed

```
======================== 77 passed in 85.68s (0:01:25) =========================
```

> 含 4 个 TDD 测试（P1-1 npu_only 硬件检查 + P1-2 requires_compiled_kernel）。
> 精度测试在 NPU kernel 可用时激活，不再走 fallback 自证。

### [NPU-2/4] Integration Tests — 10 passed, 1 skipped

```
================== 10 passed, 1 skipped in 153.17s (0:02:33) ===================
```

> `test_it_infer_006_npu_graph_capture` SKIPPED：autograd.Function Python backward 与 `make_graphed_callables` 不兼容，已知限制。

### [NPU-3/4] Performance Tests — 部分（前向性能耗时较长，单独运行）

> NPU 前向 kernel 性能：B=1,T=1000,H=24 avg=27251ms（标量三重循环，aiv_scalar_ratio=99.3%）。
> B=32,T=100,H=24 avg=2239ms（32 核并行）。
> 完整性能测试因耗时超 5 分钟未在本节完整重跑，msProf 数据见 §7.4。

### NPU 环境汇总

| 段 | 收集 | 通过 | 跳过 | 失败 |
| --- | --- | --- | --- | --- |
| Unit | 77 | 77 | 0 | 0 |
| Integration | 11 | 10 | 1 | 0 |
| **合计** | **88** | **87** | **1** | **0** |

## 6. 环境说明（CPU 与 NPU 分列）

### CPU 环境（§4 数据来源）

| 项 | 值 |
| --- | --- |
| commit | `e60f11a` |
| Python | 3.11.14 |
| pytest | 9.1.1 |
| torch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2（可导入，`is_available()=False`） |
| CANN | 8.5.0 |
| 自定义扩展 | **未编译安装** |
| 测试模式 | CPU fallback（`_torch_gru_forward`） |

### NPU 环境（§5 和 §7 数据来源）

| 项 | 值 |
| --- | --- |
| commit | `f7b953c` |
| Python | 3.11.14 |
| pytest | 9.1.1 |
| torch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2（`is_available()=True`） |
| CANN | 8.5.0 |
| 平台 | NPU 910B2 × 8 |
| 自定义扩展 | `custom_gru_v2_npu` **已编译安装** |
| LD_LIBRARY_PATH | `/usr/local/Ascend/cann-8.5.0/opp/vendors/customize/op_api/lib/` |
| 测试模式 | NPU kernel（`_CustomGRUV2Function.apply`） |

## 7. NPU 工具链实测结果（commit `eb49324`，第三轮 NPU 执行）

> **以下工具链结果来自 commit `eb49324` 的 NPU 执行。**
> **第四轮代码修改（PipeBarrier、host 校验、UB_LIMIT）需要重新编译才能验证，**
> **但本节结果证明 kernel 在第三轮代码上已通过全部工具链。**

### 7.1 msOpST — 5/5 PASS

```
[PASS] standard_shape: y=[1, 1, 100, 24] h=[1, 1, 24] nan=False inf=False
[PASS] batch32:        y=[1, 32, 100, 24] h=[1, 32, 24] nan=False inf=False
[PASS] bidirectional:  y=[2, 1, 50, 8] h=[2, 1, 8] nan=False inf=False
[PASS] single_step:    y=[1, 1, 1, 8] h=[1, 1, 8] nan=False inf=False
[PASS] no_bias:        y=[1, 1, 10, 8] h=[1, 1, 8] nan=False inf=False
Total: 5/5 passed
```

### 7.2 msSanitizer memcheck — PASS

```
memcheck standard: y.shape=torch.Size([1, 1, 100, 24]), h.shape=torch.Size([1, 1, 24])
memcheck bidir: y2.shape=torch.Size([2, 1, 50, 8]), h2.shape=torch.Size([2, 1, 8])
memcheck single_step: y3.shape=torch.Size([1, 1, 1, 8])
memcheck: ALL PASSED
```

### 7.3 msSanitizer racecheck — PASS

```
racecheck batch32: y.shape=torch.Size([1, 32, 100, 24])
racecheck: PASSED
```

### 7.4 msProf — 性能采集

| 场景 | 平均耗时 | blockDim | aiv_scalar_ratio |
| --- | --- | --- | --- |
| Single (B=1, T=1000, H=24) | 24.73ms | 1 | 99.3% |
| Batch (B=32, T=100, H=24) | 2.53ms | 32 | 99.3% |

> `aiv_scalar_ratio = 99.3%` — 标量三重循环是性能瓶颈，后续优化方向。

## 8. TDD RED→GREEN 证据（P1-8）

### 8.1 P1-1: npu_only() 硬件检查

**RED（修复前）**：

```
$ python3 -m pytest unit/test_tdd_round4.py::test_npu_only_uses_hardware_check -v

unit/test_tdd_round4.py::test_npu_only_uses_hardware_check FAILED        [ 50%]
E   AssertionError: npu_only() should skip when torch_npu imports but no NPU hardware. Got condition=False
E   assert False is True

unit/test_tdd_round4.py::test_npu_hardware_only_consistent_with_npu_only FAILED
E   AssertionError: npu_only() condition (False) != npu_hardware_only() condition (True).
============================== 2 failed in 0.14s ===============================
```

**GREEN（修复后）**：

```
$ python3 -m pytest unit/test_tdd_round4.py::test_npu_only_uses_hardware_check \
    unit/test_tdd_round4.py::test_npu_hardware_only_consistent_with_npu_only -v

unit/test_tdd_round4.py::test_npu_only_uses_hardware_check PASSED        [ 50%]
unit/test_tdd_round4.py::test_npu_hardware_only_consistent_with_npu_only PASSED [100%]
============================== 2 passed in 0.03s ===============================
```

**修复内容**：`conftest.py` 中 `npu_only()` 从 `not HAS_TORCH_NPU` 改为 `not _has_npu_hw()`，其中 `_has_npu_hw() = HAS_TORCH_NPU and torch.npu.is_available()`。

### 8.2 P1-2: requires_compiled_kernel 标记

**RED（修复前）**：

```
$ python3 -m pytest unit/test_tdd_round4.py::test_has_compiled_kernel_helper_exists \
    unit/test_tdd_round4.py::test_requires_compiled_kernel_marker_skips_without_kernel -v

unit/test_tdd_round4.py::test_has_compiled_kernel_helper_exists FAILED
E   AssertionError: conftest must define _has_compiled_kernel() — P1-2 requires detecting
E   whether the compiled NPU extension is available to avoid self-proving tests.

unit/test_tdd_round4.py::test_requires_compiled_kernel_marker_skips_without_kernel FAILED
E   AssertionError: conftest must define requires_compiled_kernel() marker.
============================== 2 failed in 0.16s ===============================
```

**GREEN（修复后）**：

```
$ python3 -m pytest unit/test_tdd_round4.py -v

unit/test_tdd_round4.py::test_npu_only_uses_hardware_check PASSED        [ 25%]
unit/test_tdd_round4.py::test_npu_hardware_only_consistent_with_npu_only PASSED [ 50%]
unit/test_tdd_round4.py::test_has_compiled_kernel_helper_exists PASSED   [ 75%]
unit/test_tdd_round4.py::test_requires_compiled_kernel_marker_skips_without_kernel PASSED [100%]
============================== 4 passed in 0.03s ===============================
```

**修复内容**：
- 新增 `_has_compiled_kernel()`：检查 NPU 硬件 + 编译扩展导入
- 新增 `requires_compiled_kernel()` marker：无编译扩展时 skip
- `test_forward_precision.py` 12 个精度测试标记为 `@requires_compiled_kernel()`
- `test_backward_precision.py` 1 个对比测试标记为 `@requires_compiled_kernel()`

## 9. 送审包信息

### 包文件

| 项 | 值 |
| --- | --- |
| 包名 | `oil-gas-ops-prospect-f7b953cca1a5487c289318268b1994c758d3fd83-ascendc-operators-custom_gru_v2.zip` |
| SHA256 | `ea00da09ee3072cf8da8fc7e113233663dd355952e44b4f260a5cab5fb94189f` |
| 字节数 | 53,972 |
| 文件数 | 33 |
| 包名内 commit | `f7b953cca1a5487c289318268b1994c758d3fd83` |
| 证据 §1.1 HEAD | `f7b953cca1a5487c289318268b1994c758d3fd83` |

> **包名 commit = 证据 HEAD**，一致性已确认。

### 被测主体文件确认

| 文件 | 在包内路径 | 字节数 | SHA256 |
| --- | --- | --- | --- |
| `custom_gru_v2.py` | `delivery/custom_gru_v2.py` + `ascendc/pybind/custom_gru_v2.py` | 9,443 | `a427535586a9499e781369f934793d1f629aebe9c8a1c3f23e8780071ab504b5` |
| `bindings.cpp` | `delivery/bindings.cpp` + `ascendc/pybind/csrc/custom_gru_v2/bindings.cpp` | 2,662 | `f11ed8bfc039dc704c0b4e49b60d17dfbe0c5c7c5a93a64fcd860afc0d800ae4` |

> 两文件均在包内（`delivery/` 副本 + 原始路径），可通过 SHA256 对拍。

## 10. 第四轮审核放行条件逐条回应

| 条件 | 级别 | 回应 |
| --- | --- | --- |
| 1. 被测主体入包 | P0 | ✅ `delivery/` 目录含 `custom_gru_v2.py` + `bindings.cpp` + SHA256 README |
| 2. 包名 commit = 证据 HEAD | P0 | ✅ 均为 `f7b953c`，§9 附 SHA256 |
| 3. 环境陈述统一 | P0 | ✅ §4 CPU 环境 / §5 NPU 环境 / §6 环境表分列，各标注 commit |
| 4. npu_only() 硬件判定 | P1 | ✅ `npu_only()` 改用 `_has_npu_hw()`，TDD RED→GREEN §8.1 |
| 5. CPU 精度/性能 skip | P1 | ✅ `requires_compiled_kernel()` marker，TDD RED→GREEN §8.2 |
| 6. kernel 同步 + 向量化 | P1 | ✅ 添加 3 处 `PipeBarrier<PIPE_V>` + GM 标量读优化注释；完整向量化为后续优化 |
| 7. host 输入校验 | P1 | ✅ nullptr 检查 + hiddenSize>0 + weight_hh 一致性 |
| 8. TDD 证据 + commit 映射 | P1 | ✅ §8 TDD RED→GREEN + §2 commit↔修复项映射表 |
