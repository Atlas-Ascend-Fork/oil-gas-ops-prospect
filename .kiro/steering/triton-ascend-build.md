---
inclusion: fileMatch
fileMatchPattern: 'triton-ascend/**'
---

# Triton-Ascend 算子开发与无卡验证

## 源码 / whl 双模式

验证脚本默认直接用仓内 `triton-ascend/` 的 Python 文件，改算子后**不需要重建 whl**。
机制在 `triton-ascend/_local_ops_package.py::use_local_triton_sources()`，由
`OIL_GAS_OPS_TRITON_SOURCE` 控制：未设/`local` 用仓内源码，`whl` 用已安装包。

`setup.py` 把 `triton-ascend/real_fft/` 映射进
`oil_gas_ops_prospect.triton.*`，所以同一份文件既是源码又是包内容。

## 无卡 AOT 编译验证（本仓最重要的编译期门禁）

没有 NPU 时 `kernel[grid](...)` 无法执行，Triton 算子的编译期回归（tl 语义、
constexpr 分支、`tl.dot` 形状、UB/L0C 超限）没有其他门禁。用：

```bash
python3 triton-ascend/tools/aot_compile_check.py            # 全部分支
python3 triton-ascend/tools/aot_compile_check.py --filter irfft
python3 triton-ascend/tools/aot_compile_check.py --dump-dir /tmp/ir
```

它把每个 kernel 的每个关注分支跑完整链路 `TTIR → Linalg(ttadapter) → bishengir →
npubin`。**改动 kernel 或新增 constexpr 分支后，必须在这里加用例并跑通。**

### 三个让它在无卡环境成立的关键点

1. `TRITON_ASCEND_ARCH=Ascend910B1` 让后端不查卡就能构造
   `GPUTarget("npu", arch, 0)`（`triton/backends/ascend/utils.py::get_ascend_arch_from_env`）。
2. bishengir 阶段 `linalg_to_bin_enable_npu_compile_A2_A3` **硬编码**调用
   `NPUUtils().get_arch()` → `rtGetSocVersion()`，无卡必失败并报
   `rtGetSocVersion failed, 0x7bc80`。必须把 `NPUUtils.get_arch` 打桩为返回 env arch
   （`stub_soc_runtime_queries()`）。仅设环境变量是不够的。
3. 走 `triton.compile(ASTSource(...), target=...)` 而非 `kernel[grid](...)`，绕开
   driver 的 device / stream 查询。

### ASTSource 用法（此处 triton 3.2）

签名是 `ASTSource(fn, signature, constants, attrs)`（旧式）：`signature` 是
参数名 → 类型串的 dict，constexpr 参数的类型串填 `"constexpr"`，其值放进
`constants`。指针参数用 `"*fp32"` 之类。

### 怎么判断 IR 正确

- `tl.dot` 应降级为 `linalg.matmul`（走 cube）。数量要对得上：`rfft` 有 2 个
  （cos/sin 基）。
- metadata 里 `mix_mode=mix`、`parallel_mode=simd`。
- `npubin` 应是 Ascend ELF（`file` 显示 `*unknown arch 0x1029*`）。

## 算子约定

- `@triton.jit` 函数**不能读非 constexpr 全局量**，常量必须内联写进 kernel。
- 小维度在 kernel 内 pad 到 16 的倍数以喂满 cube。
- grid 固定为物理 AI core 数，核内循环逻辑 tile（见各 kernel 的 `pid` /
  `tl.num_programs(0)` 用法）。
- kernel 定义放在 `if _has_triton():` 块内，保证无 triton 环境仍可导入并走 CPU 回退。

## CPU 回退

`real_fft` 有纯 PyTorch 参考实现，非 NPU 张量自动回退。这让
无卡环境能验证 API、形状、autograd 接线和参考实现本身的正确性：

```bash
python3 -m pytest triton-ascend/real_fft/test_triton_fft_real.py -q
```

回退路径下误差为 0 是**预期**的（两边都在跑参考实现），它验证的是接线而不是 kernel
数值。kernel 数值必须上卡验证。

## 下游模型交叉校验（opt-in）

这些用例默认 skip，不要为了"让测试跑起来"而硬编码下游路径：

```bash
export OIL_GAS_OPS_MODEL_REPO=/path/to/model-repo
export OIL_GAS_OPS_MODEL_CONFIG=/abs/path/to/config.yaml
export OIL_GAS_OPS_FFT_BACKBONE=my_pkg.backbone:MyNet   # 覆盖默认导入路径
export OIL_GAS_OPS_SPECTRAL_CONV=my_pkg.spectral:MyConv # --bench-3d 用
```

`verify_triton_fft_real_128_3d.py` 的独立用法（无需下游仓）：`--shape B D H W C
--axes ...`，或两个预设 `--run-latent-3d-profiled` / `--run-spectral-3d-smoke`。
