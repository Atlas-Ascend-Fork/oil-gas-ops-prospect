# 变更记录

本文件记录 oil-gas-ops-prospect 的主要变更。版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- Triton-Ascend `spectral_mlp`：频域低秩门控两层 MLP（第 1 层 `d→rank→dm` 带 GELU，第 2 层 `dm→rank→d` 无激活），输入为 rFFT 之后的实虚分离频谱 `(B,L,*spatial,G,d)`，按分组共享权重、按 `(L,G,rank)` 逐模数门控。两层的矩阵乘、门控、偏置与 GELU 融合在单个 kernel 内，层间激活不落 GM；前向不保存任何中间量，反向从输入重算两层的上投影与激活，权重/门控/偏置梯度按「每核 8 槽位部分和 + host 归约」累加。非 NPU / Triton 不可用 / 非 `float32` 时回退等价 PyTorch 实现（含 autograd）。显存收益是主要卖点：业务形态 `B=1, L=64, S=128×128, G=8, d=16, dm=16, rank=8` 下，两层之间不再同时驻留的中间量为 `16·B·L·S·G·(rank+dm)` 字节 ≈ 3.0 GB，训练单步峰值显存 7.17 GB → 4.12 GB（-42.5%）。基线为同设备 fp32 的两层 `einsum` + `npu_gelu`：同一形态下前向 28.7 ms → 12.8 ms（2.24×），前向+反向 189.1 ms → 165.0 ms（1.15×）。
- Composite-Torch：`complex_index_select`，复数版 `torch.index_select`，纯 PyTorch 组合实现。用于补齐 NPU 上复数路径的两处缺陷：原生反向落在 `aclnnIndexAdd`，其 dtype 支持列表不含复数，训练路径直接报 `161002`；原生 complex64 前向派发到 AiCPU 标量核（`GatherV2`），大形状比 float32 的 `GatherV3` 慢两个数量级（kernel 级 9.66ms vs 47.3µs）。连续输入且 `dim` 非末轴走 flat 路径（`view(float)` 后 gather，带宽与 float32 基线一致），末轴 / 1-D / 非连续输入走 real/imag 路径；反向在实数视图上 `zeros + index_add_`（零张量按 float dtype 分配，避开 complex fill 的 AiCPU 慢路径），与原生反向同构，重复索引按出现次数累加；索引 `int32` / `int64`，dtype `complex64` / `complex128`。对拍 CPU 原生 `torch.index_select`：前向逐位一致，反向最大绝对误差 `complex64` ≤ 1e-5、`complex128` ≤ 2e-14。910B2 公开 API 实测（前向 vs 原生）：频率体 `(4,64,80,144,144)×dim=1` 约 115~135×、行聚合约 124~144×、长条约 113~124×、中块约 30~39×（派发开销主导）、1-D 兜底约 22~23×；反向（前向+反向合计）与同字节量 float32 原生基线对比：大形状 0.99~1.00×、中块 0.68×（常数级派发开销，随数据量摊薄）。见 [ISSUE #61](https://gitcode.com/cann/oil-gas-ops-prospect/issues/61)。
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

- 融合注意力 Python API 将无下划线版本定义为非原地接口，并新增连续输入专用的下划线原地接口。
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
