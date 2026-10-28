# ascendc — 基于 msopgen 的多算子构建

`oil-gas-ops-prospect/ascendc`：**每次构建都从 msopgen 重新生成**与当前 CANN、SOC 匹配的工程，再合并多算子源码。

## 设计思路

```
install.sh 单算子流程:
  msopgen gen → cp 单算子实现 → build.sh

本目录多算子扩展:
  msopgen gen (内嵌 DummyOp IR)  →  与 CANN/SOC 匹配的脚手架
  merge_operators (Python)        →  拷贝 --ops 选中算子 + overlay 多 kernel CMake
  CustomOp/build.sh               →  msopgen 自带 binary + package（仍为 shell）
```

## 目录结构

```
ascendc/
├── build.py                 # 打包实现（msopgen + merge + CustomOp）
├── build.sh                 # 入口：`--pkg` 打包 / `-u --ophost` 跑 UT
├── operators/
│   ├── manifest.tsv
│   └── <name>/op_host|op_kernel/
├── scripts/
│   ├── config.py            # 常量、SOC 列表、CMake 模板、msopgen 占位 IR
│   ├── manifest.py          # 读取 manifest.tsv
│   ├── merge_operators.py   # 合并算子进 CustomOp
│   ├── patch_presets.py     # SOC / AddConfig 对齐
│   └── cann_env.py          # CANN 环境 + 子 shell 执行
└── _work/                   # msopgen 临时工程（含 ophost_ut/ 构建目录）
```

编译产物不落在本目录，统一放到**仓库根目录 `output/`**：

```
output/
├── <SOC>/oil_gas_ops_prospect_opp_<os>_<arch>.run   # OPP 算子包（build.sh --pkg）
└── wheels/oil_gas_ops_prospect-*.whl                # whl（<repo>/build.sh / tests/run_whl_tests.sh）
```

`OIL_GAS_OPS_OUTPUT_DIR` 可覆盖该目录。

## 编译与 UT

在本目录执行：

```bash
# 编译 OPP .run 与 Python whl，不安装（无参数默认）
# → <repo>/output/Ascend910B1/oil_gas_ops_prospect_opp_<os>_<arch>.run
#   <repo>/output/wheels/oil_gas_ops_prospect-*.whl
bash build.sh

# 只编译并打包 OPP .run（默认 SOC=Ascend910B1，默认全部算子）
bash build.sh --pkg

# 运行 ophost 侧单元测试（googletest）
bash build.sh -u --ophost
```

C++ 测试放在仓库根 `tests/ut`、`tests/st`；样例在 `examples/aclnn/`。`tests/ascendc/ophost/` 只做汇总编译。`operators/` 只保留 `op_host` / `op_kernel`。

| 测试文件 | 覆盖内容 |
|----------|----------|
| `tests/ut/op_host/complex_mul/test_complex_mul_tiling.cpp` | 前向 `*_tiling_core.h`：UB 预算、32B 对齐、小 tensor 收缩、退化输入、核间覆盖性不变量 |
| `tests/ut/op_host/complex_mul_grad/test_complex_mul_grad_tiling.cpp` | 反向 UB 份数（21）与单拍更短 |
| `tests/ut/op_host/fused_bias_softmax/test_fused_bias_softmax_tiling.cpp` | 对齐 padding、bias 广播/全尺寸 UB 预算、非法输入拒绝 |
| `tests/ut/op_host/fused_softmax_grad/test_fused_softmax_grad_tiling.cpp` | MIX/VEC 路径分发、V plane 驻留上限、`Q≠S` |
| `examples/aclnn/<op>/` | 两段式 `aclnn` 小样例（需要 NPU） |
| `tests/st/aclnn<OpType>/` | 多 case 两段式 `aclnn` + 真值对比（需要 NPU） |

> `<Op>TilingFunc` / `InferShape` 的入口签名依赖 `gert::TilingContext`，`Compute<Op>Tiling`
> 依赖 `platform_ascendc::PlatformAscendC`，CANN 二进制包未提供 gert faker（该基建在
> metadef 源码仓），两者都无法直接 UT。因此各算子把与平台无关的切分算法抽到
> `op_host/<op>_tiling_core.h`，UT 直接驱动核心函数，覆盖的是真实算法。产物在
> `_work/ophost_ut/`。
>
> 新增算子时：在 `tests/ut/op_host/<op>/` 放下 `test_*.cpp`，CMake 会自动收集。

常用打包参数：

```bash
bash build.sh --pkg -v Ascend910B1 --ops complex_mul complex_mul_grad fused_bias_softmax fused_softmax_grad -j 16
bash build.sh --pkg --vendor-name customize --package-name oil_gas_ops_prospect
bash build.sh clean
bash build.sh list-ops
```

`--ops` 支持空格分隔或逗号；当前 manifest 包含 `complex_mul`、`complex_mul_grad`、`fused_bias_softmax`、`fused_softmax_grad`、`conv3d_gelu`。

等价的 Python 入口：

```bash
python3 build.py -v Ascend910B1 --ops complex_mul complex_mul_grad
```

## 编译与安装 OPP + whl

如果目标是直接使用自定义算子（含 whl），优先在仓库根目录执行：

```bash
bash build.sh --install
```

该脚本会自动完成：

1. 调用本目录的 `bash build.sh --pkg` 编译 AscendC OPP `.run` 包。
2. 安装生成的 `oil_gas_ops_prospect_opp_*.run` 到 CANN。
3. 构建统一的 `oil_gas_ops_prospect` whl 包。
4. 安装 whl 包。
5. 默认运行 `tests/test_oil_gas_ops_prospect_whl.py` 冒烟测试。

常用参数：

```bash
bash build.sh --pkg                 # 只编译 OPP .run
bash build.sh                       # 编译 .run 与 whl，不安装
bash build.sh --install             # 编译并安装
bash build.sh --clean               # 先清理 ascendc/_work 与 output/
bash build.sh --ops=complex_mul     # 只构建指定 AscendC 算子
bash build.sh --install --skip-tests  # 安装后跳过 whl 冒烟测试
bash build.sh --soc=Ascend910B1 -j16
bash build.sh -u                    # 全部测试套件（需 910B）
bash build.sh -u --ophost --noexec  # 只编译 ophost UT
```

如 CANN 安装路径不是默认的 `/usr/local/Ascend/cann-9.0.0`，可通过环境变量指定：

```bash
ASCEND_HOME_PATH=/path/to/cann bash build.sh --install
```

### 包名与 vendor

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vendor-name` | `customize` | OPP 安装目录 `opp/vendors/<name>/` |
| `--package-name` | `oil_gas_ops_prospect` | 替换 `.run` 名中的 `custom` 前缀，如 `oil_gas_ops_prospect_opp_openEuler_aarch64.run` |

```bash
# 默认：vendor=customize，产物 oil_gas_ops_prospect_opp_<os>_<arch>.run
python3 build.py -v Ascend910B1

# 自定义前缀
python3 build.py -v Ascend910B1 --package-name my_prefix
# → my_prefix_opp_openEuler_aarch64.run
```

## 脚本模块说明

| 模块　　　　　　　　　　　　 | 作用　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| ------------------------------| ---------------------------------------------------------------|
| `build.py`　　　　　　　　　 | CLI、多 SOC 循环、调用 msopgen / merge / CustomOp/build.sh　　|
| `scripts/manifest.py`　　　　| 解析 `operators/manifest.tsv`　　　　　　　　　　　　　　　　 |
| `scripts/merge_operators.py` | 删 dummy、拷 host/kernel、写多算子 `op_kernel/CMakeLists.txt` |
| `scripts/patch_presets.py`　 | 修正 `CMakePresets.json` 中 ASCEND_COMPUTE_UNIT　　　　　　　 |
| `scripts/patch_package.py`　 | 设置 vendor_name / package_name 与 .run 命名规则　　　　　　　|
| `scripts/cann_env.py`　　　　| 查找 CANN 路径，`source set_env.sh` 后执行命令　　　　　　　　|

单独调试：

```bash
python3 scripts/merge_operators.py _work/CustomOp_Ascend910B1 complex_mul complex_mul_grad
python3 scripts/patch_presets.py _work/CustomOp_Ascend910B1 Ascend910B
```

## 添加新算子

1. `operators/<name>/op_host/` + `op_kernel/`
2. `operators/manifest.tsv` 追加一行（tab 分隔）
3. `python3 build.py -v Ascend910B1 --ops <name>`

### 移植 CANN 框架算子的注意事项

`conv3d_gelu` 的 cube 与 tiling 不放在本仓。构建时下载
[ops-nn v9.0.0](https://raw.gitcode.com/cann/ops-nn/archive/refs/heads/v9.0.0.tar.gz)，
只把 `conv/conv3d_v2` 的 910B 源码原样拷进本次 CustomOp，和本仓的 GELU 入口一起编。
不编译 ops-nn 工程，也不改它的源码。`arch35`、`conv3d_v2.cpp` 和 Conv3DV2 的 op 注册不拷。
`log/log.h` 由本仓 `operators/conv3d_gelu/op_host/log/log.h` 提供空宏，避免 msopgen 去拉 slog。

部分从 CANN 原仓移植的算子（如 `ffn`）的 host tiling/infershape 依赖
CANN 内部框架头（`op_host/tiling_base.h`、`op_host/tiling_templates_registry.h`、
`err/ops_err.h`、`log/log.h` 等），这些头在 msopgen 自定义算子工程的 opbuild 编译
环境下缺失或会拉入 slog 依赖。处理方式（保持源码改动最小）：

- 提供本地 compat shim（如 `ffn_log_compat.h`），用等价/降级
  实现替换 `OP_CHECK_IF`、`OP_CHECK_NULL_WITH_CONTEXT`、`OPS_REPORT_*`、`OP_LOG*`、
  `ASCENDC_EXTERN_C` 等宏；
- 仅把不可用的 include 替换为该 shim，不改动 tiling 算法本身；
- 若算子确需额外 CANN 头目录，可在 `scripts/patch_op_host.py` 的 `_EXTRA_INCLUDE_SUBDIRS`
  中追加（会注入到 opbuild 与 host 库的 include 路径）。

## 已集成算子

`operators/manifest.tsv` 当前五行：

| 目录名 | OpType | aclnn 接口 | Python | 说明 |
|--------|--------|-----------|--------|------|
| `complex_mul` | `ComplexMul` | `aclnnComplexMul` | `oil_gas_ops_prospect.complex_mul.complex_mul` | 四路 float 复数逐点乘（AIV-only） |
| `complex_mul_grad` | `ComplexMulGrad` | `aclnnComplexMulGrad` | 由 `complex_mul` 的 autograd 调用 | 上者的反向（AIV-only） |
| `fused_bias_softmax` | `FusedBiasSoftmax` | `aclnnFusedBiasSoftmax` | `oil_gas_ops_prospect.fused_bias_softmax.fused_bias_softmax` | `softmax(x+bias1+bias2,-1)`；底层 aclnn 可原地写回传入扩展的连续 `x` |
| `fused_softmax_grad` | `FusedSoftmaxGrad` | `aclnnFusedSoftmaxGrad` | `oil_gas_ops_prospect.fused_bias_softmax.fused_softmax_grad` | 注意力 softmax JVP；底层 aclnn 可原地覆写传入扩展的连续 `probs` |
| `conv3d_gelu` | `Conv3DGelu` | `aclnnConv3dGeluCustom` | `oil_gas_ops_prospect.conv3d_gelu.conv3d_gelu` | conv3d + tanh GELU，cube+vector，省掉中间激活 |
| `conv3d_gelu_grad` | `Conv3DGeluGrad` | `aclnnConv3dGeluGrad` | 由 `conv3d_gelu` 的 autograd 调用 | 重算 conv3d，再 GELU 反向，再 conv3d 反向 |

更完整的数学定义、输入输出与约束见 [`docs/zh/op_list.md`](../docs/zh/op_list.md)；Python / aclnn 签名见 [`docs/zh/api_list.md`](../docs/zh/api_list.md)。下面按算子列出本目录实现侧必须守住的规格。

### ComplexMul / ComplexMulGrad

```text
fwd:  wr = ur*vr - ui*vi          bwd:  gur =  gwr*vr + gwi*vi
      wi = ur*vi + ui*vr                gui = -gwr*vi + gwi*vr
                                        gvr =  gwr*ur + gwi*ui
                                        gvi = -gwr*ui + gwi*ur
```

| 项 | 内容 |
|----|------|
| 输入 / 输出 | 前向 `ur,ui,vr,vi → wr,wi`；反向 `gwr,gwi,ur,ui,vr,vi → gur,gui,gvr,gvi` |
| dtype / format | 全部 `float32` / `ND` / **同 shape**；支持动态 rank |
| 核类型 | AIV-only。按元素总数在 AIV 核间均分（32B 对齐），核内按 UB 分拍 |
| UB 预算 | 前向 13 份 tile，反向 21 份 tile（double buffer + tmp），见各 `*_tiling_core.h` |
| Python | 只有 `complex_mul`；反向不单独导出函数 |

NPU 精度/性能对比：[`tests/ascendc/complex_mul/`](../tests/ascendc/complex_mul/)，相对误差阈值默认 `1e-5`。

### FusedBiasSoftmax

```text
y = softmax(x + bias1(广播或全尺寸) + bias2, -1)   # 底层 aclnn 原地写回传入扩展的连续 x
```

| 项 | 内容 |
|----|------|
| 输入 / 输出 | `x [B,H,Q,S]`；可选 `bias1 [B,H,1,S]` 或 `[B,H,Q,S]`；可选 `bias2 [B,H,Q,S]`；输出 `y` 与传入扩展的连续 `x` 别名 |
| dtype / format | `fp32` / `fp16` / `bf16`，`ND`，存在的 bias 必须与 `x` 同 dtype |
| 核类型 | AIV-only。按行 `B*H*Q` 切核，行宽 `S` 对齐到 32B；SoftMax 预留 32KB UB |
| 形状约束 | **必须 4D**；`1 ≤ S ≤ 4096`；`bias2` 若存在必须全尺寸 |
| tiling key | `0=fp32`，`1=fp16`，`2=bf16` |

### FusedSoftmaxGrad

```text
g = go @ V^T
probs ← probs ⊙ (g − rowsum(probs ⊙ g))   # fp32 域；底层 aclnn 原地覆写传入扩展的连续 probs
```

| 项 | 内容 |
|----|------|
| 输入 / 输出 | `probs [B,H,Q,S]`，`gradOutput [B,H,Q,D]`，`values [B,H,S,D]` → `out` 与传入扩展的连续 `probs` 别名 |
| dtype / format | 三输入同 dtype：`fp32` / `fp16` / `bf16`，`ND` |
| 形状约束 | **必须 4D**；`Q ≠ S` 合法；`D` 必须一致且大于 0 |
| 路径分发 | `fp32`/`fp16`/非对齐 `bf16` → 纯向量（key 0/1/2）；`bf16` 且 `S%16==0` 且 `D%16==0` → MIX_AIC_1_2（key 3：AIC 矩乘写 workspace，再 AIV JVP） |
| workspace | 系统预留 16MB + MIX 路径 `B*H*Q*S*sizeof(half)` 的 `g` 缓冲 |
| 精度注意 | fp32/fp16 不走 cube（910B fixpipe C 16bit 上限）；MIX 的 half-`g` 量化误差要求 `probs` 为 softmax 分布 |

NPU 精度/性能对比（前向 + 反向 + e2e）：[`tests/ascendc/fused_bias_softmax/`](../tests/ascendc/fused_bias_softmax/)。容差：fp32 `max_diff ≤ 1e-5`；bf16 `atol=2e-2, rtol=4e-3`。

Python 还提供 `attention_core(q, k, v, bias1, bias2)`，把上述两算子按 openfold `AttentionCoreFunction` 语义串起来，**不是**独立 OPP 算子。

> C++ 侧 OpType 为 PascalCase（`ComplexMul`），aclnn 为 `aclnnComplexMul`。
> Python 侧公开函数为蛇形命名（`complex_mul`），不加 `npu_` 前缀。
> kernel 入口通过 `opFile.value`/`opInterface.value` 固定为蛇形文件名
> （`complex_mul` / `complex_mul_grad` /
> `fused_bias_softmax` / `fused_softmax_grad`），
> OpType 改名不影响 kernel 源码；aclnn 接口随 OpType 变化。

## 测试用例

各算子的 NPU 对比测试在 `<repo>/tests/ascendc/<op>/`，统一以等价的 **PyTorch/torch_npu**
表达式为基准对比 **精度 / 性能**。每个测试目录结构一致：

```
tests/ascendc/<op>/
├── test_<op>_npu.py   # 精度/性能对比脚本
├── run_test.sh        # 安装 OPP 包 → 编译扩展 → 运行测试
└── README.md
```

| 目录 | 覆盖 |
|------|------|
| [`tests/ascendc/complex_mul/`](../tests/ascendc/complex_mul/) | `ComplexMul` / `ComplexMulGrad` 前向、反向、autograd、时延 |
| [`tests/ascendc/fused_bias_softmax/`](../tests/ascendc/fused_bias_softmax/) | `FusedBiasSoftmax` / `FusedSoftmaxGrad` 前向、反向、attention e2e、时延 |

测试用到的 pybind 扩展是算子交付件而非测试资产，源码在
[`pybind/`](pybind/)，详见 [`pybind/README.md`](pybind/README.md)。

另有 ophost tiling UT，见上文「编译与 UT」。

### 前置条件

1. 已构建出 OPP 包（见上文「编译与 UT」），产物在 `<repo>/output/<SOC>/oil_gas_ops_prospect_opp_*.run`。
2. **测试需在带 NPU 设备的机器上运行**，且环境内有 `torch` 与 `torch_npu`。

### 运行

```bash
# 复数乘
cd ../tests/ascendc/complex_mul
bash run_test.sh
bash run_test.sh --shape 1 1 96 160 160

# 融合注意力（前反向一对）
cd ../tests/ascendc/fused_bias_softmax
bash run_test.sh

# 指定 OPP 包（默认取 <repo>/output/Ascend910B1/oil_gas_ops_prospect_opp_openEuler_aarch64.run）
OIL_GAS_OPS_RUN_PKG=/path/to/oil_gas_ops_prospect_opp_openEuler_aarch64.run bash run_test.sh
```

`run_test.sh` 两步：安装/更新 OPP 包 → 编译 pybind 扩展并跑对应 `test_*_npu.py`。也可手动分步：

```bash
bash ../../../output/Ascend910B1/oil_gas_ops_prospect_opp_openEuler_aarch64.run --quiet
python3 test_complex_mul_npu.py --build-ext
python3 test_complex_mul_npu.py
```

### 输出

`complex_mul` 脚本打印前向/反向相对误差、时延与加速比，以及 autograd 封装的梯度误差：

```text
[fwd] shape=... rel_err wr=... wi=...
[bwd] rel_err gur=... gui=... gvr=... gvi=...
[perf] fwd torch=...ms ascendc=...ms speedup=...x
[autograd] gur=... gui=... gvr=... gvi=...
PASS
```

`fused_bias_softmax` 脚本按 forward / backward / e2e 打印 `max_diff`、超容差元素数与是否原地写回。

退出码：`0` 精度达标；`1` 超阈值 / 无 NPU / 扩展缺失。

详见 [`tests/ascendc/complex_mul/README.md`](../tests/ascendc/complex_mul/README.md) 与
[`tests/ascendc/fused_bias_softmax/README.md`](../tests/ascendc/fused_bias_softmax/README.md)。
