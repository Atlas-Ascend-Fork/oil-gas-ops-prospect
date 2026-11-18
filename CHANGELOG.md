# 变更记录

本文件记录 oil-gas-ops-prospect 的主要变更。版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- Composite-Torch：`complex_addcmul`，复数三输入逐元素 `input + value*tensor1*tensor2`，纯 PyTorch 组合实现，复数乘委托 `complex_mul`。用于规避 NPU 上 `torch.addcmul` 不支持 `complex64` / `complex128` 的限制（`aclnnAddcmul` 报 `161002`，`DT_COMPLEX64` 不在支持列表内），支持广播、`value` 取实数或复数标量与 autograd（`value` 本身不参与求导）；`complex128` 按 `torch.promote_types` 与 `complex64` 混合提升。对拍 CPU 原生 `torch.addcmul` 的最大绝对误差：`complex64` ≤ 1e-5、`complex128` ≤ 2e-14（均为浮点舍入级，非逐位一致）。
- Triton-Ascend `real_fft`：新增二维组合接口 `real_rfft2` / `real_irfft2`，语义对齐 `torch.fft.rfft2` / `irfft2`，支持 `norm="backward"`（默认）与 `"ortho"` 及 autograd。由现有 1-D `real_rfft` / `real_irfft` 组合而成（线性 + 共轭反转 + Hermitian 半谱补全），不新增 kernel；非末轴且长度较大时把轴搬到末轴（正向、反向都内置），末轴半谱补全前先把该轴搬到 0 以避开 910B 上 stride-1 `flip` 的慢路径。返回实数 `(real, imag)` 对，不构造 complex dtype。非 NPU / Triton 不可用 / 非 `float32` 时回退 `torch.fft`。910B2 实测收益按行数 `numel/(h*w)` 分档：行数 ≲64 时慢 3~5 倍，约 256 行起反超（N=288 时 1.30×、N=576 时 1.17×）；收益来自实数输出表示与连续布局（原生 `rfft2` 返回非连续 `complex64`，其上逐点乘加约慢 200 倍），不是变换本身更快。
- AscendC：`Conv3DGelu`，cube conv3d 与 vector tanh-GELU 融合，float32 / NCDHW。前向不保存二者之间的预激活。反向是 `Conv3DGeluGrad`：重算 conv3d，再做 GELU 反向，再做 conv3d 反向。
- 融合注意力算子 `FusedBiasSoftmax` / `FusedSoftmaxGrad`，以及组合接口 `attention_core`。
- 新增 Triton-Ascend 复数算子 `complex_reciprocal`（复数倒数 `1/z`，支持实虚分离入口与自定义反向）与 `complex_exp`（复数指数 `e^z`，自定义反向），替代 NPU 上走 Ai_Cpu 的 `aclnnReciprocal` / `aclnnExp`。
- 统一 Python 包 `oil_gas_ops_prospect`：AscendC 经 `ascendc/pybind/` 暴露，Triton 暴露 `real_rfft` / `real_irfft`。
- Triton-Ascend `interpolate`：2D/3D 线性插值；5D fp32 trilinear 按空间轴拆成批量 Cube GEMM，支持一阶 autograd、大张量分块与 CPU 回退，4D 使用原生实现。
- 根目录三种构建入口：`bash build.sh` 只编译出包，`--pkg` 只出 OPP `.run`，`--install` 安装并冒烟。
- 统一测试入口 `bash build.sh -u`（切分、上板、精度、性能、示例、FFT、文档门禁）；`--opensource-report` 生成开源测试报告。

### Changed

- `Conv3DGelu`：1×1 直写只在输入通道不少于 8、输出通道不超过 64 时启用。其余形状的前向按一个输出平面切，私有格式缓冲不再按整幅分配。反向由 `Conv3DGeluGrad` 一次完成。
- AscendC OpType 去掉 `Og` 后缀（`ComplexMul` 等）；Python 接口为蛇形命名，不加 `npu_` 前缀。
- `operators/` 只保留实现；测试在 `tests/`，aclnn 样例在 `examples/aclnn/`。
- 许可证改为 Apache License 2.0。

### Fixed

- `complex_mul` 的反向不再重复 clone 梯度输入与输出。`gwr` / `gwi` 在 `(wr + wi).sum()` 这类调用下互为同一张量的 view（可能是 stride 0 的广播布局），`_kernel_in` 的 `contiguous()` 已把它们物化成独立缓冲，额外 `clone()` 只是多一次整张量拷贝。前向仍保留 clone：kernel 把两路输出写进同一块缓冲，不 clone 会让 `autograd.Function` 返回共享 storage 的多 view，下游原地写报错。
- `complex_exp` 的数值语义对齐 `torch.exp(complex64)`（`cexpf`）：纯实数上溢不再得到 `inf+nanj`，`exp(a)` 溢出带内（如 `89+1j`）实部不再丢失为 `inf`，`b` 为 `±inf`/NaN 时按 `cexpf` 特判；补上二阶导（`create_graph=True`）与 conjugate-view 梯度；AOT 门禁补齐反向 int64 分支。
- `complex_mul` 的 Python autograd 接线，反向梯度可正确回传到输入。
- `bash build.sh -u` 单 flag 入口无法进入测试的问题。

## [0.1.0] - 2026-08-27

### Added

- 首次发布油气勘探昇腾算子库。
- AscendC：`ComplexMul` / `ComplexMulGrad`（float32 / ND，产出 OPP `.run`）。
- Triton-Ascend：`real_rfft` / `real_irfft`（`norm="ortho"`，支持 autograd）。
