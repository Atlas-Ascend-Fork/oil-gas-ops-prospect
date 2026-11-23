# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------------------------------------

"""频域低秩门控两层 MLP 的 Triton-Ascend 算子。

输入是 rFFT 之后的实虚分离频谱 ``(B, L, *spatial, G, d)``：``L`` 为保留的
低频模数，``*spatial`` 为其余空间轴展平后的 ``S``，``G`` 为通道分组，``d``
为每组特征维。算子把两层的小矩阵乘（Cube ``tl.dot``）、逐频点门控、偏置与
GELU 融合进一个 kernel，两层之间的激活不落 GM。

记 ``x = xr + i·xi``，第 1 层把 ``d`` 投影到 ``rank`` 再升回 ``dm``，第 2 层
把 ``dm`` 投影到 ``rank`` 再降回 ``d``（无激活）::

    X1r = xr · U1                X1i = xi · U1
    sum1 = X1r ⊙ g1r + X1i ⊙ g1i
    diff1 = X1r ⊙ g1r − X1i ⊙ g1i
    h1r = diff1 · V1 + b1r       h1i = sum1 · V1 + b1i
    o1r = GELU(h1r)              o1i = GELU(h1i)

    X2r = o1r · U2               X2i = o1i · U2
    sum2 = X2r ⊙ g2r + X2i ⊙ g2i
    diff2 = X2r ⊙ g2r − X2i ⊙ g2i
    yr  = diff2 · V2 + b2r       yi  = sum2 · V2 + b2i

两个矩阵乘都按 ``G`` 分组做（每个分组一套权重），``U1/U2`` 的收缩轴是特征
维，``V1/V2`` 的收缩轴是低秩维；门控按 ``(L, G, rank)`` 逐模数、逐分组生效。

实现要点
--------
- 一个逻辑 tile = 一个 ``(b, l, g)`` 三元组 + ``S`` 轴上一段连续切片，tile 内
  门控与偏置都是常量，可直接广播；kernel 以物理 AI Core 数作为 grid，在核内
  循环处理逻辑 tile。
- ``d / dm / rank`` 在 kernel 内向上取到 2 的幂（且不小于 16），以满足
  ``tl.arange`` 与 Cube 的 16 对齐要求。
- 前向不保存任何中间量；反向从输入重算 ``x1/h1/o1/x2``（6 个小 GEMM），
  训练显存只与输入/输出有关。权重/门控/偏置梯度先写进「每核 8 槽位」的部分和
  缓冲，再由 host 求和，避免跨核原子竞争。
- GELU 使用 ``x / (1 + exp(-1.5957691 * (x + 0.044715 * x^3)))`` 近似，纯
  PyTorch 回退路径与之逐位一致。

回退与约束
----------
- 非 NPU、Triton 不可用或输入不是 ``float32`` 时回退到等价的 PyTorch
  参考实现（``einsum`` + 同一 GELU 近似），接口语义与 NPU 路径一致。
- NPU kernel 仅支持 ``float32``；所有输入需同设备、同 dtype。
- ``block_m`` / ``block_m_bwd`` 会向下取到 2 的幂，并按 UB 预算钳制；默认
  512 / 256，是 128³ 业务形态下实测最优的档位。
"""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments can still import this module.
    triton = None
    tl = None


# GELU 近似常数（与 torch 参考实现共用）。
GELU_C0 = 1.5957691
GELU_C1 = 0.044715

# 910B 单核 UB 约 192KB，留出掩码/索引/编译器临时量的余量后按 160KB 估算。
_USABLE_UB_BYTES = 160 * 1024
_FP32_BYTES = 4

_DEFAULT_BLOCK_M = 512
_DEFAULT_BLOCK_M_BWD = 256

# 反向权重/门控/偏置梯度按 (核, 槽位) 分片部分和，槽位数决定单槽顺序累加长度。
_GRAD_SLOTS = 8


def _has_triton() -> bool:
    return triton is not None and tl is not None


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


def _get_aicore_num(default: int = 20) -> int:
    """当前 NPU 的 AI Core 数，作为 persistent kernel 的 grid。"""
    try:
        import torch_npu  # type: ignore[import-not-found]
        import triton.runtime.driver as driver
    except ImportError:
        return default
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return default
    try:
        device = torch_npu.npu.current_device()
        properties = driver.active.utils.get_device_properties(device)
        return int(properties.get("num_aicore", default))
    except Exception:
        return default


def _pad_dim(x: int) -> int:
    """向上取到不小于 16 的 2 的幂（``tl.arange`` / Cube 切片的约束）。"""
    x = max(16, int(x))
    return 1 << (x - 1).bit_length()


def _round_block_m(block_m: int) -> int:
    """把 tile 大小向下取到 2 的幂，保证 ``tl.arange`` 合法。"""
    block_m = max(16, int(block_m))
    return 1 << (block_m.bit_length() - 1)


def _ub_tile_budget_elements(live_tile_count: int) -> int:
    """给每个 tile 的 ``M * K`` 元素数预算（按同时驻留的 fp32 tile 数折减）。"""
    return max(1024, _USABLE_UB_BYTES // _FP32_BYTES // max(1, int(live_tile_count)))


def _safe_block_m(block_m: int, pad_dims, budget: int) -> int:
    """按 UB 预算钳制 ``block_m``：``block_m * max(pad_dims) <= budget``。"""
    block_m = _round_block_m(block_m)
    max_dim = max(int(d) for d in pad_dims)
    cap = max(16, budget // max(1, max_dim))
    cap = _round_block_m(cap)
    return max(16, min(block_m, cap))


def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    """与 kernel 逐位一致的 GELU 近似（供参考实现与测试共用）。"""
    return x / (1.0 + torch.exp(-GELU_C0 * (x + GELU_C1 * x * x * x)))


if _has_triton():

    @triton.jit
    def _gelu_fwd(x):
        # constants inlined: Triton @jit functions cannot read non-constexpr globals.
        return x / (1.0 + tl.exp(-1.5957691 * (x + 0.044715 * x * x * x)))

    @triton.jit
    def _gelu_bwd(x):
        # d/dx of x * sigmoid(c0*(x + c1*x^3)); constants inlined.
        u = 1.5957691 * (x + 0.044715 * x * x * x)
        s = tl.sigmoid(u)
        dudx = 1.5957691 * (1.0 + 3.0 * 0.044715 * x * x)
        return s + x * s * (1.0 - s) * dudx

    @triton.jit
    def _spectral_mlp_fwd_kernel(
        xr_ptr, xi_ptr,
        up1_ptr, down1t_ptr, gate1r_ptr, gate1i_ptr, bias1r_ptr, bias1i_ptr,
        up2_ptr, down2t_ptr, gate2r_ptr, gate2i_ptr, bias2r_ptr, bias2i_ptr,
        yr_ptr, yi_ptr,
        B: tl.constexpr, L: tl.constexpr, S: tl.constexpr, G: tl.constexpr,
        D: tl.constexpr, DM: tl.constexpr, R: tl.constexpr,
        D_PAD: tl.constexpr, DM_PAD: tl.constexpr, R_PAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_s_blocks = tl.cdiv(S, BLOCK_M)
        total = B * L * G * num_s_blocks

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D_PAD)
        offs_dm = tl.arange(0, DM_PAD)
        offs_r = tl.arange(0, R_PAD)
        d_mask = offs_d < D
        dm_mask = offs_dm < DM
        r_mask = offs_r < R

        for tile in range(pid, total, num_core):
            sb = tile % num_s_blocks
            t2 = tile // num_s_blocks
            g = t2 % G
            bl = t2 // G
            mode = bl % L
            s_idx = sb * BLOCK_M + offs_m
            s_mask = s_idx < S

            x_off = bl * S * G * D + g * D + s_idx[:, None] * (G * D) + offs_d[None, :]
            x_mask = s_mask[:, None] & d_mask[None, :]
            xr = tl.load(xr_ptr + x_off, mask=x_mask, other=0.0)
            xi = tl.load(xi_ptr + x_off, mask=x_mask, other=0.0)

            # ---- layer 1 ----
            up1 = tl.load(up1_ptr + g * D * R + offs_d[:, None] * R + offs_r[None, :],
                         mask=d_mask[:, None] & r_mask[None, :], other=0.0)
            x1r = tl.dot(xr, up1)
            x1i = tl.dot(xi, up1)
            gate1_off = mode * G * R + g * R + offs_r
            gate1r = tl.load(gate1r_ptr + gate1_off, mask=r_mask, other=0.0)
            gate1i = tl.load(gate1i_ptr + gate1_off, mask=r_mask, other=0.0)
            sum1 = x1r * gate1r[None, :] + x1i * gate1i[None, :]
            diff1 = x1r * gate1r[None, :] - x1i * gate1i[None, :]
            down1t = tl.load(down1t_ptr + g * R * DM + offs_r[:, None] * DM + offs_dm[None, :],
                          mask=r_mask[:, None] & dm_mask[None, :], other=0.0)
            bias1r = tl.load(bias1r_ptr + g * DM + offs_dm, mask=dm_mask, other=0.0)
            bias1i = tl.load(bias1i_ptr + g * DM + offs_dm, mask=dm_mask, other=0.0)
            h1r = tl.dot(diff1, down1t) + bias1r[None, :]
            h1i = tl.dot(sum1, down1t) + bias1i[None, :]
            o1r = _gelu_fwd(h1r)
            o1i = _gelu_fwd(h1i)

            # ---- layer 2 ----
            up2 = tl.load(up2_ptr + g * DM * R + offs_dm[:, None] * R + offs_r[None, :],
                         mask=dm_mask[:, None] & r_mask[None, :], other=0.0)
            x2r = tl.dot(o1r, up2)
            x2i = tl.dot(o1i, up2)
            gate2_off = mode * G * R + g * R + offs_r
            gate2r = tl.load(gate2r_ptr + gate2_off, mask=r_mask, other=0.0)
            gate2i = tl.load(gate2i_ptr + gate2_off, mask=r_mask, other=0.0)
            sum2 = x2r * gate2r[None, :] + x2i * gate2i[None, :]
            diff2 = x2r * gate2r[None, :] - x2i * gate2i[None, :]
            down2t = tl.load(down2t_ptr + g * R * D + offs_r[:, None] * D + offs_d[None, :],
                          mask=r_mask[:, None] & d_mask[None, :], other=0.0)
            bias2r = tl.load(bias2r_ptr + g * D + offs_d, mask=d_mask, other=0.0)
            bias2i = tl.load(bias2i_ptr + g * D + offs_d, mask=d_mask, other=0.0)
            y1r = tl.dot(diff2, down2t) + bias2r[None, :]
            y1i = tl.dot(sum2, down2t) + bias2i[None, :]

            tl.store(yr_ptr + x_off, y1r, mask=x_mask)
            tl.store(yi_ptr + x_off, y1i, mask=x_mask)

    @triton.jit
    def _spectral_mlp_bwd_kernel(
        xr_ptr, xi_ptr,
        up1t_ptr, down1_ptr, gate1r_ptr, gate1i_ptr, bias1r_ptr, bias1i_ptr,
        up1_ptr, down1t_ptr,
        up2t_ptr, down2_ptr, gate2r_ptr, gate2i_ptr, up2_ptr,
        gyr_ptr, gyi_ptr,
        gxr_ptr, gxi_ptr,
        # per-tile partial accumulators (leading NUM_TILES dim) -> reduced on host.
        g_up1_ptr, g_down1_ptr, g_gate1r_ptr, g_gate1i_ptr, g_bias1r_ptr, g_bias1i_ptr,
        g_up2_ptr, g_down2_ptr, g_gate2r_ptr, g_gate2i_ptr, g_bias2r_ptr, g_bias2i_ptr,
        B: tl.constexpr, L: tl.constexpr, S: tl.constexpr, G: tl.constexpr,
        D: tl.constexpr, DM: tl.constexpr, R: tl.constexpr,
        D_PAD: tl.constexpr, DM_PAD: tl.constexpr, R_PAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        NUM_SLOTS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_s_blocks = tl.cdiv(S, BLOCK_M)
        total = B * L * G * num_s_blocks

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D_PAD)
        offs_dm = tl.arange(0, DM_PAD)
        offs_r = tl.arange(0, R_PAD)
        d_mask = offs_d < D
        dm_mask = offs_dm < DM
        r_mask = offs_r < R

        for tile in range(pid, total, num_core):
            # 每个 (核, 槽位) 写一片部分和并轮转累加：核间没有原子竞争，且把单个
            # 槽位的顺序累加长度压到 1/NUM_SLOTS；host 侧对所有槽位求和。bias/gate
            # 这类跨 S 全量规约的 fp32 舍入误差主要来自这串顺序累加。
            slot = pid * NUM_SLOTS + ((tile // num_core) % NUM_SLOTS)
            tile_bd = slot * (G * D)        # bias over D  (g_bias2)
            tile_bdm = slot * (G * DM)      # bias over DM (g_bias1)
            tile_dr = slot * (G * D * R)    # (G, D, R) buffers: g_up1, g_down2
            tile_dmr = slot * (G * DM * R)  # (G, DM, R) buffers: g_down1, g_up2
            tile_g = slot * (L * G * R)     # gate buffers
            sb = tile % num_s_blocks
            t2 = tile // num_s_blocks
            g = t2 % G
            bl = t2 // G
            mode = bl % L
            s_idx = sb * BLOCK_M + offs_m
            s_mask = s_idx < S

            x_off = bl * S * G * D + g * D + s_idx[:, None] * (G * D) + offs_d[None, :]
            x_mask = s_mask[:, None] & d_mask[None, :]
            xr = tl.load(xr_ptr + x_off, mask=x_mask, other=0.0)
            xi = tl.load(xi_ptr + x_off, mask=x_mask, other=0.0)

            # ---- 重算前向中间量：forward 不再把 x1/h1/x2 写回 GM ----
            gate1_off = mode * G * R + g * R + offs_r
            gate1r = tl.load(gate1r_ptr + gate1_off, mask=r_mask, other=0.0)
            gate1i = tl.load(gate1i_ptr + gate1_off, mask=r_mask, other=0.0)
            up1t_off = g * R * D + offs_r[:, None] * D + offs_d[None, :]
            up1t = tl.load(up1t_ptr + up1t_off, mask=r_mask[:, None] & d_mask[None, :], other=0.0)
            up1 = tl.load(up1_ptr + g * D * R + offs_d[:, None] * R + offs_r[None, :],
                          mask=d_mask[:, None] & r_mask[None, :], other=0.0)
            x1r = tl.dot(xr, up1)
            x1i = tl.dot(xi, up1)
            sum1 = x1r * gate1r[None, :] + x1i * gate1i[None, :]
            diff1 = x1r * gate1r[None, :] - x1i * gate1i[None, :]
            down1_off = g * DM * R + offs_dm[:, None] * R + offs_r[None, :]
            down1_mask = dm_mask[:, None] & r_mask[None, :]
            down1 = tl.load(down1_ptr + down1_off, mask=down1_mask, other=0.0)
            bias1r_v = tl.load(bias1r_ptr + g * DM + offs_dm, mask=dm_mask, other=0.0)
            bias1i_v = tl.load(bias1i_ptr + g * DM + offs_dm, mask=dm_mask, other=0.0)
            down1t = tl.load(down1t_ptr + g * R * DM + offs_r[:, None] * DM + offs_dm[None, :],
                             mask=r_mask[:, None] & dm_mask[None, :], other=0.0)
            h1r = tl.dot(diff1, down1t) + bias1r_v[None, :]
            h1i = tl.dot(sum1, down1t) + bias1i_v[None, :]
            o1r = _gelu_fwd(h1r)
            o1i = _gelu_fwd(h1i)
            gate2_off = mode * G * R + g * R + offs_r
            gate2r = tl.load(gate2r_ptr + gate2_off, mask=r_mask, other=0.0)
            gate2i = tl.load(gate2i_ptr + gate2_off, mask=r_mask, other=0.0)
            up2t_off = g * R * DM + offs_r[:, None] * DM + offs_dm[None, :]
            up2t = tl.load(up2t_ptr + up2t_off, mask=r_mask[:, None] & dm_mask[None, :], other=0.0)
            up2 = tl.load(up2_ptr + g * DM * R + offs_dm[:, None] * R + offs_r[None, :],
                          mask=dm_mask[:, None] & r_mask[None, :], other=0.0)
            x2r = tl.dot(o1r, up2)
            x2i = tl.dot(o1i, up2)
            sum2 = x2r * gate2r[None, :] + x2i * gate2i[None, :]
            diff2 = x2r * gate2r[None, :] - x2i * gate2i[None, :]

            gy_r = tl.load(gyr_ptr + x_off, mask=x_mask, other=0.0)  # (M, D_PAD)
            gy_i = tl.load(gyi_ptr + x_off, mask=x_mask, other=0.0)

            # ---- layer2 backward (private RMW into per-core buffers) ----
            g_bias2r_p = g_bias2r_ptr + tile_bd + g * D + offs_d
            g_bias2i_p = g_bias2i_ptr + tile_bd + g * D + offs_d
            tl.store(g_bias2r_p, tl.load(g_bias2r_p, mask=d_mask, other=0.0) + tl.sum(gy_r, axis=0), mask=d_mask)
            tl.store(g_bias2i_p, tl.load(g_bias2i_p, mask=d_mask, other=0.0) + tl.sum(gy_i, axis=0), mask=d_mask)
            down2_off = g * D * R + offs_d[:, None] * R + offs_r[None, :]
            down2_mask = d_mask[:, None] & r_mask[None, :]
            down2 = tl.load(down2_ptr + down2_off, mask=down2_mask, other=0.0)  # (D_PAD, R_PAD)
            grad_diff2 = tl.dot(gy_r, down2)
            grad_sum2 = tl.dot(gy_i, down2)
            g_down2 = tl.dot(tl.trans(gy_r), diff2) + tl.dot(tl.trans(gy_i), sum2)
            g_down2_p = g_down2_ptr + tile_dr + down2_off
            tl.store(g_down2_p, tl.load(g_down2_p, mask=down2_mask, other=0.0) + g_down2, mask=down2_mask)
            grad_x2r_g = grad_sum2 + grad_diff2
            grad_x2i_g = grad_sum2 - grad_diff2
            g_gate2r_p = g_gate2r_ptr + tile_g + gate2_off
            g_gate2i_p = g_gate2i_ptr + tile_g + gate2_off
            acc = tl.load(g_gate2r_p, mask=r_mask, other=0.0) + tl.sum(grad_x2r_g * x2r, axis=0)
            tl.store(g_gate2r_p, acc, mask=r_mask)
            acc = tl.load(g_gate2i_p, mask=r_mask, other=0.0) + tl.sum(grad_x2i_g * x2i, axis=0)
            tl.store(g_gate2i_p, acc, mask=r_mask)
            grad_x2r = grad_x2r_g * gate2r[None, :]
            grad_x2i = grad_x2i_g * gate2i[None, :]
            grad_o1r = tl.dot(grad_x2r, up2t)  # (M, DM_PAD)
            grad_o1i = tl.dot(grad_x2i, up2t)
            up2_off = g * DM * R + offs_dm[:, None] * R + offs_r[None, :]
            g_up2 = tl.dot(tl.trans(o1r), grad_x2r) + tl.dot(tl.trans(o1i), grad_x2i)
            g_up2_p = g_up2_ptr + tile_dmr + up2_off
            up2_mask = dm_mask[:, None] & r_mask[None, :]
            acc = tl.load(g_up2_p, mask=up2_mask, other=0.0) + g_up2
            tl.store(g_up2_p, acc, mask=up2_mask)

            # ---- GELU backward ----
            grad_h1r = grad_o1r * _gelu_bwd(h1r)
            grad_h1i = grad_o1i * _gelu_bwd(h1i)

            # ---- layer1 backward (private RMW into per-core buffers) ----
            g_bias1r_p = g_bias1r_ptr + tile_bdm + g * DM + offs_dm
            g_bias1i_p = g_bias1i_ptr + tile_bdm + g * DM + offs_dm
            tl.store(g_bias1r_p, tl.load(g_bias1r_p, mask=dm_mask, other=0.0) + tl.sum(grad_h1r, axis=0), mask=dm_mask)
            tl.store(g_bias1i_p, tl.load(g_bias1i_p, mask=dm_mask, other=0.0) + tl.sum(grad_h1i, axis=0), mask=dm_mask)
            grad_diff1 = tl.dot(grad_h1r, down1)
            grad_sum1 = tl.dot(grad_h1i, down1)
            g_down1 = tl.dot(tl.trans(grad_h1r), diff1) + tl.dot(tl.trans(grad_h1i), sum1)
            g_down1_p = g_down1_ptr + tile_dmr + down1_off
            tl.store(g_down1_p, tl.load(g_down1_p, mask=down1_mask, other=0.0) + g_down1, mask=down1_mask)
            grad_x1r_g = grad_sum1 + grad_diff1
            grad_x1i_g = grad_sum1 - grad_diff1
            g_gate1r_p = g_gate1r_ptr + tile_g + gate1_off
            g_gate1i_p = g_gate1i_ptr + tile_g + gate1_off
            acc = tl.load(g_gate1r_p, mask=r_mask, other=0.0) + tl.sum(grad_x1r_g * x1r, axis=0)
            tl.store(g_gate1r_p, acc, mask=r_mask)
            acc = tl.load(g_gate1i_p, mask=r_mask, other=0.0) + tl.sum(grad_x1i_g * x1i, axis=0)
            tl.store(g_gate1i_p, acc, mask=r_mask)
            grad_x1r = grad_x1r_g * gate1r[None, :]
            grad_x1i = grad_x1i_g * gate1i[None, :]
            grad_xr = tl.dot(grad_x1r, up1t)
            grad_xi = tl.dot(grad_x1i, up1t)
            tl.store(gxr_ptr + x_off, grad_xr, mask=x_mask)
            tl.store(gxi_ptr + x_off, grad_xi, mask=x_mask)
            up1_off = g * D * R + offs_d[:, None] * R + offs_r[None, :]
            g_up1 = tl.dot(tl.trans(xr), grad_x1r) + tl.dot(tl.trans(xi), grad_x1i)
            g_up1_p = g_up1_ptr + tile_dr + up1_off
            up1_mask = d_mask[:, None] & r_mask[None, :]
            acc = tl.load(g_up1_p, mask=up1_mask, other=0.0) + g_up1
            tl.store(g_up1_p, acc, mask=up1_mask)




def _check_inputs(
    xr, xi,
    up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
    up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
):
    """校验形状并返回 ``(B, L, S, G, D, DM, R, spatial)``。"""
    if xr.shape != xi.shape:
        raise ValueError(f"xr and xi must match, got {tuple(xr.shape)} vs {tuple(xi.shape)}")
    if xr.ndim < 3:
        raise ValueError(f"xr must be (B, L, *spatial, G, d), got {tuple(xr.shape)}")
    B, L = int(xr.shape[0]), int(xr.shape[1])
    G, D = int(xr.shape[-2]), int(xr.shape[-1])
    spatial = tuple(int(v) for v in xr.shape[2:-2])
    S = 1
    for s in spatial:
        S *= s

    if up1.shape != (G, D, up1.shape[2]):
        raise ValueError(f"up1 must be (G, d, rank), got {tuple(up1.shape)}")
    R = int(up1.shape[2])
    DM = int(down1.shape[1])
    for name, t, want in (
        ("down1", down1, (G, DM, R)),
        ("up2", up2, (G, DM, R)),
        ("down2", down2, (G, D, R)),
        ("gate1_r", gate1_r, (L, G, R)),
        ("gate1_i", gate1_i, (L, G, R)),
        ("gate2_r", gate2_r, (L, G, R)),
        ("gate2_i", gate2_i, (L, G, R)),
        ("bias1_r", bias1_r, (G, DM)),
        ("bias1_i", bias1_i, (G, DM)),
        ("bias2_r", bias2_r, (G, D)),
        ("bias2_i", bias2_i, (G, D)),
    ):
        if tuple(t.shape) != want:
            raise ValueError(f"{name} must be {want}, got {tuple(t.shape)}")
    return B, L, S, G, D, DM, R, spatial


def _torch_reference(
    xr, xi,
    up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
    up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
):
    """纯 PyTorch 等价实现（einsum + 同一 GELU 近似），供回退与对拍。"""
    B, L, S, G, D, DM, R, spatial = _check_inputs(
        xr, xi, up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
        up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
    )
    xr2 = xr.reshape(B, L, S, G, D)
    xi2 = xi.reshape(B, L, S, G, D)

    x1r = torch.einsum("blsgf,gfr->blsgr", xr2, up1)
    x1i = torch.einsum("blsgf,gfr->blsgr", xi2, up1)
    gate1r = gate1_r.view(1, L, 1, G, R)
    gate1i = gate1_i.view(1, L, 1, G, R)
    sum1 = x1r * gate1r + x1i * gate1i
    diff1 = x1r * gate1r - x1i * gate1i
    h1r = torch.einsum("blsgr,gdr->blsgd", diff1, down1) + bias1_r.view(1, 1, 1, G, DM)
    h1i = torch.einsum("blsgr,gdr->blsgd", sum1, down1) + bias1_i.view(1, 1, 1, G, DM)
    o1r = gelu_approx(h1r)
    o1i = gelu_approx(h1i)

    x2r = torch.einsum("blsgf,gfr->blsgr", o1r, up2)
    x2i = torch.einsum("blsgf,gfr->blsgr", o1i, up2)
    gate2r = gate2_r.view(1, L, 1, G, R)
    gate2i = gate2_i.view(1, L, 1, G, R)
    sum2 = x2r * gate2r + x2i * gate2i
    diff2 = x2r * gate2r - x2i * gate2i
    yr = torch.einsum("blsgr,gdr->blsgd", diff2, down2) + bias2_r.view(1, 1, 1, G, D)
    yi = torch.einsum("blsgr,gdr->blsgd", sum2, down2) + bias2_i.view(1, 1, 1, G, D)

    out_shape = (B, L, *spatial, G, D)
    return yr.reshape(out_shape), yi.reshape(out_shape)


def _forward(
    xr, xi,
    up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
    up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
    block_m: int,
):
    B, L, S, G, D, DM, R, spatial = _check_inputs(
        xr, xi, up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
        up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
    )
    xr_c = xr.reshape(B * L, S, G, D).contiguous()
    xi_c = xi.reshape(B * L, S, G, D).contiguous()
    up1_c = up1.contiguous()
    down1t_c = down1.transpose(-2, -1).contiguous()  # (G, rank, dm)
    up2_c = up2.contiguous()
    down2t_c = down2.transpose(-2, -1).contiguous()  # (G, rank, d)
    gate1r_c, gate1i_c = gate1_r.contiguous(), gate1_i.contiguous()
    gate2r_c, gate2i_c = gate2_r.contiguous(), gate2_i.contiguous()
    bias1r_c, bias1i_c = bias1_r.contiguous(), bias1_i.contiguous()
    bias2r_c, bias2i_c = bias2_r.contiguous(), bias2_i.contiguous()

    # 用 torch.empty(shape) 而不是 empty_like：torch_npu 的 empty_like 会尝试沿用
    # 输入的 internal format，触发 "Cannot create tensor with interal format" 告警。
    # 前向不保存任何中间量；反向从输入重算。
    yr = torch.empty((B * L, S, G, D), device=xr.device, dtype=torch.float32)
    yi = torch.empty((B * L, S, G, D), device=xr.device, dtype=torch.float32)

    d_pad, dm_pad, r_pad = _pad_dim(D), _pad_dim(DM), _pad_dim(R)
    # 两层融合 kernel 同时驻留两层 tile，按 5 个 live tile 折减 UB 预算。
    block_m = _safe_block_m(block_m, (d_pad, dm_pad), budget=_ub_tile_budget_elements(5))

    _spectral_mlp_fwd_kernel[(_get_aicore_num(),)](
        xr_c, xi_c,
        up1_c, down1t_c, gate1r_c, gate1i_c, bias1r_c, bias1i_c,
        up2_c, down2t_c, gate2r_c, gate2i_c, bias2r_c, bias2i_c,
        yr, yi,
        B, L, S, G, D, DM, R,
        d_pad, dm_pad, r_pad,
        block_m,
    )
    out_shape = (B, L, *spatial, G, D)
    return yr.reshape(out_shape), yi.reshape(out_shape)


def _backward(
    gyr, gyi,
    xr, xi,
    up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
    up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
    block_m: int,
):
    B, L, S, G, D, DM, R, _ = _check_inputs(
        xr, xi, up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
        up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
    )
    xr_c = xr.reshape(B * L, S, G, D).contiguous()
    xi_c = xi.reshape(B * L, S, G, D).contiguous()
    gyr_c = gyr.reshape(B * L, S, G, D).contiguous()
    gyi_c = gyi.reshape(B * L, S, G, D).contiguous()
    up1t_c = up1.transpose(-2, -1).contiguous()  # (G, rank, d)
    down1_c = down1.contiguous()                 # (G, dm, rank)
    up2t_c = up2.transpose(-2, -1).contiguous()  # (G, rank, dm)
    down2_c = down2.contiguous()                 # (G, d, rank)
    up1_c = up1.contiguous()                     # (G, d, rank)
    down1t_c = down1.transpose(-2, -1).contiguous()  # (G, rank, dm)
    up2_c = up2.contiguous()                     # (G, dm, rank)
    gate1r_c, gate1i_c = gate1_r.contiguous(), gate1_i.contiguous()
    gate2r_c, gate2i_c = gate2_r.contiguous(), gate2_i.contiguous()
    bias1r_c, bias1i_c = bias1_r.contiguous(), bias1_i.contiguous()

    grad_xr = torch.empty((B * L, S, G, D), device=xr.device, dtype=torch.float32)
    grad_xi = torch.empty((B * L, S, G, D), device=xr.device, dtype=torch.float32)

    # 每个 (核, 槽位) 一片部分和，host 侧对槽位维求和；既避免跨核原子竞争，也把
    # 单槽顺序累加长度压到 1/_GRAD_SLOTS，降低 bias/gate 规约的 fp32 误差累积。
    num_core = _get_aicore_num()
    num_slots = num_core * _GRAD_SLOTS
    f32 = dict(device=xr.device, dtype=torch.float32)
    grad_up1 = torch.zeros((num_slots, G, D, R), **f32)
    grad_down1 = torch.zeros((num_slots, G, DM, R), **f32)
    grad_gate1r = torch.zeros((num_slots, L, G, R), **f32)
    grad_gate1i = torch.zeros((num_slots, L, G, R), **f32)
    grad_bias1r = torch.zeros((num_slots, G, DM), **f32)
    grad_bias1i = torch.zeros((num_slots, G, DM), **f32)
    grad_up2 = torch.zeros((num_slots, G, DM, R), **f32)
    grad_down2 = torch.zeros((num_slots, G, D, R), **f32)
    grad_gate2r = torch.zeros((num_slots, L, G, R), **f32)
    grad_gate2i = torch.zeros((num_slots, L, G, R), **f32)
    grad_bias2r = torch.zeros((num_slots, G, D), **f32)
    grad_bias2i = torch.zeros((num_slots, G, D), **f32)

    d_pad, dm_pad, r_pad = _pad_dim(D), _pad_dim(DM), _pad_dim(R)
    # 反向同时算两层梯度，按 10 个 live tile 折减 UB 预算。
    block_m = _safe_block_m(block_m, (d_pad, dm_pad), budget=_ub_tile_budget_elements(10))

    _spectral_mlp_bwd_kernel[(num_core,)](
        xr_c, xi_c,
        up1t_c, down1_c, gate1r_c, gate1i_c, bias1r_c, bias1i_c,
        up1_c, down1t_c,
        up2t_c, down2_c, gate2r_c, gate2i_c, up2_c,
        gyr_c, gyi_c,
        grad_xr, grad_xi,
        grad_up1, grad_down1, grad_gate1r, grad_gate1i, grad_bias1r, grad_bias1i,
        grad_up2, grad_down2, grad_gate2r, grad_gate2i, grad_bias2r, grad_bias2i,
        B, L, S, G, D, DM, R,
        d_pad, dm_pad, r_pad,
        block_m,
        _GRAD_SLOTS,
    )
    return (
        grad_xr.reshape(xr.shape),
        grad_xi.reshape(xi.shape),
        grad_up1.sum(0), grad_down1.sum(0),
        grad_gate1r.sum(0), grad_gate1i.sum(0),
        grad_bias1r.sum(0), grad_bias1i.sum(0),
        grad_up2.sum(0), grad_down2.sum(0),
        grad_gate2r.sum(0), grad_gate2i.sum(0),
        grad_bias2r.sum(0), grad_bias2i.sum(0),
    )


if _has_triton():

    class _SpectralMLPFunction(torch.autograd.Function):
        """两层融合前向/反向（仅 NPU + Triton + float32）。"""

        @staticmethod
        def forward(ctx, xr, xi, up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
                    up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
                    block_m, block_m_bwd):
            ctx.block_m_bwd = block_m_bwd
            with torch.no_grad():
                y1r, y1i = _forward(
                    xr, xi, up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
                    up2, down2, gate2_r, gate2_i, bias2_r, bias2_i, block_m,
                )
            # 前向不保存中间量；反向从输入重算，训练显存只与输入/输出有关。
            ctx.save_for_backward(
                xr, xi, up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
                up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
            )
            return y1r, y1i

        @staticmethod
        def backward(ctx, grad_y1r, grad_y1i):
            inputs = ctx.saved_tensors
            grads = _backward(
                grad_y1r.contiguous(), grad_y1i.contiguous(), *inputs, ctx.block_m_bwd,
            )
            out = [g if (g is not None and t.requires_grad) else None for t, g in zip(inputs, grads)]
            # block_m / block_m_bwd 是非张量参数，不需要梯度。
            return (*out, None, None)


def spectral_mlp(
    xr: torch.Tensor,
    xi: torch.Tensor,
    up1: torch.Tensor,
    down1: torch.Tensor,
    gate1_r: torch.Tensor,
    gate1_i: torch.Tensor,
    bias1_r: torch.Tensor,
    bias1_i: torch.Tensor,
    up2: torch.Tensor,
    down2: torch.Tensor,
    gate2_r: torch.Tensor,
    gate2_i: torch.Tensor,
    bias2_r: torch.Tensor,
    bias2_i: torch.Tensor,
    *,
    block_m: int = _DEFAULT_BLOCK_M,
    block_m_bwd: int = _DEFAULT_BLOCK_M_BWD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """频域低秩门控两层 MLP：``(xr, xi) -> (yr, yi)``。

    第 1 层 ``d -> rank -> dm`` 带 GELU，第 2 层 ``dm -> rank -> d`` 无激活；
    两层之间不落 GM。支持一阶 autograd（输入与全部权重/门控/偏置都可导）。

    :param xr: 频谱实部，``(B, L, *spatial, G, d)``，``float32``。
    :param xi: 频谱虚部，形状同 ``xr``。
    :param up1: 第 1 层上投影 ``(G, d, rank)``。
    :param down1: 第 1 层下投影 ``(G, dm, rank)``。
    :param gate1_r: 第 1 层实部门控 ``(L, G, rank)``。
    :param gate1_i: 第 1 层虚部门控 ``(L, G, rank)``。
    :param bias1_r: 第 1 层实部偏置 ``(G, dm)``。
    :param bias1_i: 第 1 层虚部偏置 ``(G, dm)``。
    :param up2: 第 2 层上投影 ``(G, dm, rank)``。
    :param down2: 第 2 层下投影 ``(G, d, rank)``。
    :param gate2_r: 第 2 层实部门控 ``(L, G, rank)``。
    :param gate2_i: 第 2 层虚部门控 ``(L, G, rank)``。
    :param bias2_r: 第 2 层实部偏置 ``(G, d)``。
    :param bias2_i: 第 2 层虚部偏置 ``(G, d)``。
    :param block_m: 前向 tile 的 ``S`` 方向大小（取 2 的幂，默认 512）。
    :param block_m_bwd: 反向 tile 大小（默认 256）。
    :return: ``(yr, yi)``，形状同 ``xr``，``float32``。

    非 NPU、Triton 不可用或输入不是 ``float32`` 时回退到等价的 PyTorch 实现。
    """
    if not (_is_npu_tensor(xr) and _has_triton()):
        return _torch_reference(
            xr, xi, up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
            up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
        )
    tensors = (
        xr, xi, up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
        up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
    )
    if any(t.dtype != torch.float32 for t in tensors):
        return _torch_reference(
            xr, xi, up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
            up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
        )
    return _SpectralMLPFunction.apply(
        xr, xi, up1, down1, gate1_r, gate1_i, bias1_r, bias1_i,
        up2, down2, gate2_r, gate2_i, bias2_r, bias2_i,
        _round_block_m(block_m), _round_block_m(block_m_bwd),
    )


__all__ = ["spectral_mlp"]
