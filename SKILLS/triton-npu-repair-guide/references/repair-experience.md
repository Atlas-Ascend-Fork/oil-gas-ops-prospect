# Repair experience（本仓库）

## 1. L0C / L1 overflow

日志含 `cc overflow`、`cbuf overflow`、`overflow`，或被包成 `ERR99999` / `bisheng` / `mlircompilation`。

- 先让 `_cap_fft_tiles` / `_cap_block_m_for_l0c` / `_cap_block_m_for_cbuf` 生效。
- 再靠 `_launch_with_block_m_backoff` 减半 `BLOCK_M`，`cache_key` 必须含 n、freq、tile、device。
- 不要为过编译把 `tile_k`/`tile_f` 缩到盖不住整段 K（axis kernel 注释已写：一次 `tl.dot` 覆盖整段）。

## 2. 轴上 kernel 的 tile_k / tile_f

axis 路径 `tile_k`/`tile_f`/`tile_t` 必须 `round_up` 到覆盖 n 或 freq_len。缩小它们会算错，不是合法的 overflow 修复。只缩 `tile_b` / `BLOCK_M`。

## 3. `allow_internal_format` / resize 警告

`torch.fft` 在 NPU 上可能打 resize / internal format 警告。那是标杆实现，不是本算子失败。不要为此改官方阈值。

## 4. 精度对拍

- 标杆：CPU 或同设备 `torch.fft`，`norm="ortho"`。
- 原生 NPU `torch.fft` 反向有时与 CPU 不一致；官方以 CPU 为受理标。
- 相对误差目标约 `1e-4`～`3e-3`，见测试文件，不要放宽。

## 5. 编译环境

AOT：`python3 triton-ascend/tools/aot_compile_check.py`。无卡 CI 靠它抓 tl.dot / constexpr 回归。
