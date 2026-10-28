# 变更记录

本文件记录 oil-gas-ops-prospect 的主要变更。版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

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

- `complex_exp` 的数值语义对齐 `torch.exp(complex64)`（`cexpf`）：纯实数上溢不再得到 `inf+nanj`，`exp(a)` 溢出带内（如 `89+1j`）实部不再丢失为 `inf`，`b` 为 `±inf`/NaN 时按 `cexpf` 特判；补上二阶导（`create_graph=True`）与 conjugate-view 梯度；AOT 门禁补齐反向 int64 分支。
- `complex_mul` 的 Python autograd 接线，反向梯度可正确回传到输入。
- `bash build.sh -u` 单 flag 入口无法进入测试的问题。

## [0.1.0] - 2026-08-27

### Added

- 首次发布油气勘探昇腾算子库。
- AscendC：`ComplexMul` / `ComplexMulGrad`（float32 / ND，产出 OPP `.run`）。
- Triton-Ascend：`real_rfft` / `real_irfft`（`norm="ortho"`，支持 autograd）。
