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

import math
import os
from typing import Dict, Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments can still import this module.
    triton = None
    tl = None


_BasisKey = Tuple[int, torch.device, str]
_basis_cache: Dict[_BasisKey, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_aicore_num(default: int = 20) -> int:
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


def _is_npu_tensor(x: torch.Tensor) -> bool:
    return x.device.type == "npu"


def _has_triton() -> bool:
    return triton is not None and tl is not None


def _round_up(x: int, mult: int = 16) -> int:
    """Round ``x`` up to a multiple of ``mult`` (>= mult); used to size tl.dot tiles."""
    return max(mult, (int(x) + mult - 1) // mult * mult)


# Row-tile (M) size for the DFT matmul kernels. These are tall-skinny GEMMs
# (rows ~= millions, n/freq_len tiny), so a big row-tile amortizes the per-block
# launch/index overhead and turns the work into a few large tl.dot calls per
# core. 512 was the fastest across a 128/256/512 sweep on 910B for the profiled
# n=41/freq_len=21 shape (rfft/irfft fwd ~24ms, fwd+bwd ~38ms) and fits on-chip
# (fp32 accumulators <=128KB in L0C). Override with OIL_GAS_OPS_FFT_BLOCK_M to re-sweep
# on other shapes/hardware (drop to 256/128 if a shape overflows UB/L0C).
_DEFAULT_FFT_BLOCK_M = 512


def _default_block_m() -> int:
    try:
        v = int(os.environ.get("OIL_GAS_OPS_FFT_BLOCK_M", ""))
        return v if v > 0 else _DEFAULT_FFT_BLOCK_M
    except (TypeError, ValueError):
        return _DEFAULT_FFT_BLOCK_M


# Ascend 910B L0C (cube accumulator) capacity. The kernels keep their fp32
# ``tl.dot`` accumulators (``acc``/``acc_r``/``acc_i``) resident in L0C, so their
# total size must fit here or BiShengIR aborts with "cc overflow". The default
# BLOCK_M=512 was tuned for freq_len~21 (2 * 512 * 32 * 4B = 128KB, right at the
# limit); larger freq_len/n (e.g. size 64 -> freq_len 33 -> padded 48) would need
# 192KB and overflow. We therefore shrink BLOCK_M per shape so the accumulators
# always fit, independent of the (fixed) other tile dim.
_ASCEND_L0C_ACC_BYTES = 128 * 1024
# 910B L1/cbuf ≈ 512KB。tl.dot 操作数（含 double-buffer）按 2 份计。
_ASCEND_L1_CBUF_BYTES = 512 * 1024
_FP32_BYTES = 4


def _cap_block_m_for_l0c(block_m: int, tile_other: int, num_acc: int) -> int:
    """Shrink ``block_m`` so ``num_acc`` fp32 ``(block_m, tile_other)`` accumulators fit in L0C.

    ``tile_other`` is the accumulator's other (non-row) tile dim in elements
    (already 16-rounded), ``num_acc`` the number of simultaneously live fp32
    accumulators in the kernel. Keeps the result 16-aligned for ``tl.dot``.
    """
    if tile_other <= 0 or num_acc <= 0:
        return block_m
    max_m = _ASCEND_L0C_ACC_BYTES // (num_acc * int(tile_other) * _FP32_BYTES)
    max_m -= max_m % 16
    max_m = max(16, max_m)
    return max(16, min(int(block_m), max_m))


def _cap_block_m_for_cbuf(block_m: int, tile_k: int, tile_n: int, num_operands: int = 6) -> int:
    """Shrink ``block_m`` so ``tl.dot`` L1/cbuf operands fit in 910B L1.

    ``num_operands`` counts simultaneously live tiles after multi-buffering
    plus compiler temporaries (default 6: two (M,K) + two (K,N) + extras).
    BiShengIR ``--enable-auto-multi-buffer`` on N=256 needed ~1.35× the
    4-operand estimate (4980736 bits vs 4194304 available), so we also keep
    a 65% L1 safety factor.
    """
    if tile_k <= 0 or tile_n <= 0 or num_operands <= 0:
        return block_m
    tile = max(int(tile_k), int(tile_n))
    budget = int(_ASCEND_L1_CBUF_BYTES * 0.65)
    max_m = budget // (num_operands * tile * _FP32_BYTES)
    max_m -= max_m % 16
    max_m = max(16, max_m)
    return max(16, min(int(block_m), max_m))


# (K,N) 单拍上限：n=128 已验证的 128×64。n=256 的 256×144 与 BLOCK_M 无关，
# 单拍就要 4980736 bits（L1 仅 4194304）。kernel 已按 tile_k 循环，缩小 K/N 即可。
_ASCEND_L1_MAX_KN = 128 * 64


def _align16(x: int) -> int:
    x = max(16, int(x))
    return x - (x % 16)


def _cap_block_kn_for_cbuf(tile_k: int, tile_n: int, max_kn: int = _ASCEND_L1_MAX_KN) -> tuple[int, int]:
    """Shrink K/N tiles so a (K,N) operand pair fits in 910B L1."""
    tk, tn = _align16(tile_k), _align16(tile_n)
    while tk * tn > max_kn and (tk > 16 or tn > 16):
        if tk >= tn and tk > 16:
            tk = _align16(tk // 2)
        elif tn > 16:
            tn = _align16(tn // 2)
        else:
            break
    return tk, tn


def _cap_fft_tiles(block_m: int, tile_k: int, tile_n: int, num_acc: int) -> tuple[int, int, int]:
    """Cap K/N first (L1), then M (L0C + remaining L1)."""
    tile_k, tile_n = _cap_block_kn_for_cbuf(tile_k, tile_n)
    block_m = _cap_block_m_for_l0c(block_m, tile_n, num_acc=num_acc)
    block_m = _cap_block_m_for_cbuf(block_m, tile_k, tile_n)
    return block_m, tile_k, tile_n


# Substrings identifying BiShengIR on-chip buffer overflows that scale with the
# row tile: "cc overflow" (L0C accumulators) and "cbuf overflow" (L1 tl.dot
# operands). Both are fixed by shrinking BLOCK_M.
_BLOCK_M_OVERFLOW_MARKERS = ("cbuf overflow", "cc overflow", "overflow")


def _exception_text(err: BaseException) -> str:
    parts: list[str] = []
    cur: BaseException | None = err
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(str(cur))
        parts.append(repr(cur))
        parts.extend(str(a) for a in getattr(cur, "args", ()))
        for attr in ("error_detail", "detail", "message", "stderr", "output", "stdout"):
            val = getattr(cur, attr, None)
            if val:
                parts.append(str(val) if not isinstance(val, bytes) else val.decode("utf-8", "replace"))
        cur = cur.__cause__ or cur.__context__
    return "\n".join(parts)


def _is_block_m_overflow(err: BaseException) -> bool:
    msg = _exception_text(err).lower()
    if any(marker in msg for marker in _BLOCK_M_OVERFLOW_MARKERS):
        return True
    # torch_npu 有时把 BiShengIR 失败包成 ERR99999，正文不含 overflow
    # 字样；缩小 BLOCK_M 不改变数值，对编译失败做一次回退是安全的。
    return any(tok in msg for tok in ("bisheng", "mlircompilation", "ccec", "ttadapter"))


# Maps a caller-supplied ``cache_key`` (kernel site + shape-dependent tile
# sizes + device) to the last ``BLOCK_M`` that actually compiled. Only the
# static L0C cap (see ``_cap_block_m_for_l0c``) is cheap to predict; the L1/cbuf
# footprint (tl.dot operands, multi-buffering) is not, so the *first* call for
# a given shape may still need to back off through one or more failing sizes
# before landing on a compilable one. Without this cache, every subsequent
# call would blindly retry from the capped default and re-pay for the failing
# ``bishengir-compile`` subprocess invocation(s) on *every single call* (Triton
# only caches successful compiles, never failed ones) -- this was measured to
# make some kernels ~30x slower than their (first-try-successful) counterparts
# in steady state. Caching the resolved value lets repeat calls jump straight
# to the known-good ``BLOCK_M``.
_BLOCK_M_RESOLVED_CACHE: Dict[tuple, int] = {}


def _launch_with_block_m_backoff(launch, block_m: int, min_block_m: int = 16, cache_key: tuple | None = None):
    """Call ``launch(block_m)``; on an NPU on-chip buffer overflow, halve the
    (16-aligned) row tile and retry until it compiles or reaches ``min_block_m``.

    The FFT kernels keep both their ``tl.dot`` operands (L1/cbuf) and fp32
    accumulators (L0C/cc) on-chip, and both scale with the row tile ``BLOCK_M``.
    The exact footprint (multi-buffering, compiler temporaries) is hard to
    predict per shape/hardware, so beyond the static L0C cap we also back off on
    the actual compile error. Kernel math is independent of ``BLOCK_M``, so
    retries never change results.

    If ``cache_key`` is given, the last successfully-compiled ``BLOCK_M`` for
    that key is reused as the starting point (capped at the caller's
    requested ``block_m``), so repeat calls with the same shape skip straight
    past sizes already known to overflow instead of re-attempting (and
    re-paying compile cost for) them every time.
    """
    bm = max(min_block_m, int(block_m))
    bm -= bm % 16
    bm = max(min_block_m, bm)
    if cache_key is not None:
        cached_bm = _BLOCK_M_RESOLVED_CACHE.get(cache_key)
        if cached_bm is not None and cached_bm < bm:
            bm = cached_bm
    while True:
        try:
            result = launch(bm)
            if cache_key is not None:
                _BLOCK_M_RESOLVED_CACHE[cache_key] = bm
            return result
        except Exception as err:  # noqa: BLE001 - re-raised unless it is a row-tile overflow
            if bm <= min_block_m or not _is_block_m_overflow(err):
                raise
            bm = max(min_block_m, (bm // 2) - ((bm // 2) % 16))


def make_basis(n: int, device: torch.device, norm: str):
    if norm != "ortho":
        raise ValueError(f"only norm='ortho' is supported, got {norm!r}")

    key = (n, device, norm)
    cached = _basis_cache.get(key)
    if cached is not None:
        return cached

    freq_len = n // 2 + 1
    compute_dtype = torch.float32
    t = torch.arange(n, dtype=compute_dtype)
    k = torch.arange(freq_len, dtype=compute_dtype)
    phase = (2.0 * math.pi / float(n)) * k[:, None] * t[None, :]
    scale = n ** -0.5

    # rfft computes X @ basis where basis has shape (n, freq_len).
    rfft_cos = (torch.cos(phase).transpose(0, 1) * scale).contiguous().to(device=device)
    rfft_sin = (-torch.sin(phase).transpose(0, 1) * scale).contiguous().to(device=device)

    # irfft reconstructs the omitted conjugate half exactly like torch.fft.irfft.
    weight = torch.ones(freq_len, dtype=compute_dtype)
    weight[1:(n + 1) // 2] = 2.0
    irfft_cos = (torch.cos(phase).transpose(0, 1) * weight[None, :] * scale).contiguous().to(device=device)
    irfft_sin = (torch.sin(phase).transpose(0, 1) * weight[None, :] * scale).contiguous().to(device=device)

    cached = (rfft_cos.contiguous(), rfft_sin.contiguous(), irfft_cos.contiguous(), irfft_sin.contiguous())
    _basis_cache[key] = cached
    return cached


if _has_triton():

    @triton.jit
    def _rfft_matmul_kernel(
        x,
        cos_t,
        sin_t,
        out_r,
        out_i,
        rows: tl.constexpr,
        n: tl.constexpr,
        freq_len: tl.constexpr,
        num_blocks: tl.constexpr,
        tile_m: tl.constexpr,
        tile_n: tl.constexpr,
        tile_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_blocks_n = tl.cdiv(freq_len, tile_n)

        offs_m = tl.arange(0, tile_m)
        offs_n = tl.arange(0, tile_n)
        offs_k = tl.arange(0, tile_k)

        for block_idx in range(pid, num_blocks, num_core):
            block_m = block_idx // num_blocks_n
            block_n = block_idx - block_m * num_blocks_n
            rows_idx = block_m * tile_m + offs_m
            freq_idx = block_n * tile_n + offs_n

            acc_r = tl.zeros((tile_m, tile_n), tl.float32)
            acc_i = tl.zeros((tile_m, tile_n), tl.float32)
            for k_start in range(0, n, tile_k):
                k_idx = k_start + offs_k
                a = tl.load(
                    x + rows_idx[:, None] * n + k_idx[None, :],
                    mask=(rows_idx[:, None] < rows) & (k_idx[None, :] < n),
                    other=0.0,
                )
                b_r = tl.load(
                    cos_t + k_idx[:, None] * freq_len + freq_idx[None, :],
                    mask=(k_idx[:, None] < n) & (freq_idx[None, :] < freq_len),
                    other=0.0,
                )
                b_i = tl.load(
                    sin_t + k_idx[:, None] * freq_len + freq_idx[None, :],
                    mask=(k_idx[:, None] < n) & (freq_idx[None, :] < freq_len),
                    other=0.0,
                )
                acc_r += tl.dot(a, b_r)
                acc_i += tl.dot(a, b_i)

            out_offsets = rows_idx[:, None] * freq_len + freq_idx[None, :]
            out_mask = (rows_idx[:, None] < rows) & (freq_idx[None, :] < freq_len)
            tl.store(out_r + out_offsets, acc_r, mask=out_mask)
            tl.store(out_i + out_offsets, acc_i, mask=out_mask)


    @triton.jit
    def _irfft_matmul_kernel(
        xr,
        xi,
        cos_t,
        sin_t,
        out,
        rows: tl.constexpr,
        n: tl.constexpr,
        freq_len: tl.constexpr,
        num_blocks: tl.constexpr,
        tile_m: tl.constexpr,
        tile_n: tl.constexpr,
        tile_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_blocks_n = tl.cdiv(n, tile_n)

        offs_m = tl.arange(0, tile_m)
        offs_n = tl.arange(0, tile_n)
        offs_k = tl.arange(0, tile_k)

        for block_idx in range(pid, num_blocks, num_core):
            block_m = block_idx // num_blocks_n
            block_n = block_idx - block_m * num_blocks_n
            rows_idx = block_m * tile_m + offs_m
            time_idx = block_n * tile_n + offs_n

            acc = tl.zeros((tile_m, tile_n), tl.float32)
            for k_start in range(0, freq_len, tile_k):
                k_idx = k_start + offs_k
                a_r = tl.load(
                    xr + rows_idx[:, None] * freq_len + k_idx[None, :],
                    mask=(rows_idx[:, None] < rows) & (k_idx[None, :] < freq_len),
                    other=0.0,
                )
                a_i = tl.load(
                    xi + rows_idx[:, None] * freq_len + k_idx[None, :],
                    mask=(rows_idx[:, None] < rows) & (k_idx[None, :] < freq_len),
                    other=0.0,
                )
                b_r = tl.load(
                    cos_t + k_idx[:, None] * n + time_idx[None, :],
                    mask=(k_idx[:, None] < freq_len) & (time_idx[None, :] < n),
                    other=0.0,
                )
                b_i = tl.load(
                    sin_t + k_idx[:, None] * n + time_idx[None, :],
                    mask=(k_idx[:, None] < freq_len) & (time_idx[None, :] < n),
                    other=0.0,
                )
                acc += tl.dot(a_r, b_r) - tl.dot(a_i, b_i)

            out_offsets = rows_idx[:, None] * n + time_idx[None, :]
            out_mask = (rows_idx[:, None] < rows) & (time_idx[None, :] < n)
            tl.store(out + out_offsets, acc, mask=out_mask)


    @triton.jit
    def _rfft_backward_matmul_kernel(
        grad_r,
        grad_i,
        cos_f,
        sin_f,
        grad_x,
        rows: tl.constexpr,
        n: tl.constexpr,
        freq_len: tl.constexpr,
        num_blocks: tl.constexpr,
        tile_m: tl.constexpr,
        tile_n: tl.constexpr,
        tile_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_blocks_n = tl.cdiv(n, tile_n)

        offs_m = tl.arange(0, tile_m)
        offs_n = tl.arange(0, tile_n)
        offs_k = tl.arange(0, tile_k)

        for block_idx in range(pid, num_blocks, num_core):
            block_m = block_idx // num_blocks_n
            block_n = block_idx - block_m * num_blocks_n
            rows_idx = block_m * tile_m + offs_m
            time_idx = block_n * tile_n + offs_n

            acc = tl.zeros((tile_m, tile_n), tl.float32)
            for k_start in range(0, freq_len, tile_k):
                k_idx = k_start + offs_k
                gr = tl.load(
                    grad_r + rows_idx[:, None] * freq_len + k_idx[None, :],
                    mask=(rows_idx[:, None] < rows) & (k_idx[None, :] < freq_len),
                    other=0.0,
                )
                gi = tl.load(
                    grad_i + rows_idx[:, None] * freq_len + k_idx[None, :],
                    mask=(rows_idx[:, None] < rows) & (k_idx[None, :] < freq_len),
                    other=0.0,
                )
                br = tl.load(
                    cos_f + k_idx[:, None] * n + time_idx[None, :],
                    mask=(k_idx[:, None] < freq_len) & (time_idx[None, :] < n),
                    other=0.0,
                )
                bi = tl.load(
                    sin_f + k_idx[:, None] * n + time_idx[None, :],
                    mask=(k_idx[:, None] < freq_len) & (time_idx[None, :] < n),
                    other=0.0,
                )
                acc += tl.dot(gr, br) + tl.dot(gi, bi)

            out_offsets = rows_idx[:, None] * n + time_idx[None, :]
            out_mask = (rows_idx[:, None] < rows) & (time_idx[None, :] < n)
            tl.store(grad_x + out_offsets, acc, mask=out_mask)


    @triton.jit
    def _irfft_backward_matmul_kernel(
        grad_y,
        cos_t,
        sin_t,
        grad_r,
        grad_i,
        rows: tl.constexpr,
        n: tl.constexpr,
        freq_len: tl.constexpr,
        num_blocks: tl.constexpr,
        tile_m: tl.constexpr,
        tile_n: tl.constexpr,
        tile_k: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_blocks_n = tl.cdiv(freq_len, tile_n)

        offs_m = tl.arange(0, tile_m)
        offs_n = tl.arange(0, tile_n)
        offs_k = tl.arange(0, tile_k)

        for block_idx in range(pid, num_blocks, num_core):
            block_m = block_idx // num_blocks_n
            block_n = block_idx - block_m * num_blocks_n
            rows_idx = block_m * tile_m + offs_m
            freq_idx = block_n * tile_n + offs_n

            acc_r = tl.zeros((tile_m, tile_n), tl.float32)
            acc_i = tl.zeros((tile_m, tile_n), tl.float32)
            for k_start in range(0, n, tile_k):
                k_idx = k_start + offs_k
                gy = tl.load(
                    grad_y + rows_idx[:, None] * n + k_idx[None, :],
                    mask=(rows_idx[:, None] < rows) & (k_idx[None, :] < n),
                    other=0.0,
                )
                br = tl.load(
                    cos_t + k_idx[:, None] * freq_len + freq_idx[None, :],
                    mask=(k_idx[:, None] < n) & (freq_idx[None, :] < freq_len),
                    other=0.0,
                )
                bi = tl.load(
                    sin_t + k_idx[:, None] * freq_len + freq_idx[None, :],
                    mask=(k_idx[:, None] < n) & (freq_idx[None, :] < freq_len),
                    other=0.0,
                )
                acc_r += tl.dot(gy, br)
                acc_i -= tl.dot(gy, bi)

            out_offsets = rows_idx[:, None] * freq_len + freq_idx[None, :]
            out_mask = (rows_idx[:, None] < rows) & (freq_idx[None, :] < freq_len)
            tl.store(grad_r + out_offsets, acc_r, mask=out_mask)
            tl.store(grad_i + out_offsets, acc_i, mask=out_mask)

    # ------------------------------------------------------------------ #
    # Transpose-free "axis" kernels.
    #
    # When the FFT axis is not the last dim, the (rows, n) kernels above force two
    # movedim(...).contiguous() transposes of the whole tensor per call (once in,
    # once out) so the contracted axis is innermost. These kernels instead operate
    # directly on the natural ``(A, mid, B)`` view (A = product of dims before the
    # axis, B = product of the trailing, already-contiguous dims), contracting the
    # middle axis via a single tl.dot with the tiny Fourier basis and writing the
    # output in-place layout. No transpose / extra GM round-trip is needed. The
    # big contiguous ``B`` axis becomes the tl.dot N dimension (good cube/vector
    # utilization); the tiny basis is the M/K operand.
    # ------------------------------------------------------------------ #
    @triton.jit
    def _rfft_axis_matmul_kernel(
        x, cos_t, sin_t, out_r, out_i,
        n_lead: tl.constexpr, n: tl.constexpr, freq_len: tl.constexpr, n_tail: tl.constexpr,
        num_blocks: tl.constexpr,
        tile_b: tl.constexpr, tile_f: tl.constexpr, tile_k: tl.constexpr,
    ):
        # out[a, f, b] = sum_k cos_t[k, f] * x[a, k, b] ; out_i uses sin_t.
        # cos_t/sin_t are the rfft basis in natural (n, freq_len) layout.
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_b_blocks = tl.cdiv(n_tail, tile_b)
        offs_f = tl.arange(0, tile_f)
        offs_k = tl.arange(0, tile_k)
        offs_b = tl.arange(0, tile_b)
        f_mask = offs_f < freq_len
        k_mask = offs_k < n
        # basis as [F, K]: element (f, k) lives at k * freq_len + f.
        basis_off = offs_k[None, :] * freq_len + offs_f[:, None]
        basis_mask = k_mask[None, :] & f_mask[:, None]
        cos_fk = tl.load(cos_t + basis_off, mask=basis_mask, other=0.0)
        sin_fk = tl.load(sin_t + basis_off, mask=basis_mask, other=0.0)
        for block_idx in range(pid, num_blocks, num_core):
            a = block_idx // num_b_blocks
            bb = block_idx - a * num_b_blocks
            b_idx = bb * tile_b + offs_b
            b_mask = b_idx < n_tail
            x_kb = tl.load(
                x + a * n * n_tail + offs_k[:, None] * n_tail + b_idx[None, :],
                mask=k_mask[:, None] & b_mask[None, :], other=0.0,
            )  # [K, B]
            acc_r = tl.dot(cos_fk, x_kb)  # [F, B]
            acc_i = tl.dot(sin_fk, x_kb)
            out_off = a * freq_len * n_tail + offs_f[:, None] * n_tail + b_idx[None, :]
            out_mask = f_mask[:, None] & b_mask[None, :]
            tl.store(out_r + out_off, acc_r, mask=out_mask)
            tl.store(out_i + out_off, acc_i, mask=out_mask)

    @triton.jit
    def _rfft_axis_backward_kernel(
        grad_r, grad_i, cos_t, sin_t, grad_x,
        n_lead: tl.constexpr, n: tl.constexpr, freq_len: tl.constexpr, n_tail: tl.constexpr,
        num_blocks: tl.constexpr,
        tile_b: tl.constexpr, tile_k: tl.constexpr, tile_f: tl.constexpr,
    ):
        # grad_x[a, k, b] = sum_f cos_t[k, f] * grad_r[a, f, b] + sin_t[k, f] * grad_i[a, f, b]
        # cos_t/sin_t are the rfft basis in natural (n, freq_len) layout.
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_b_blocks = tl.cdiv(n_tail, tile_b)
        offs_k = tl.arange(0, tile_k)
        offs_f = tl.arange(0, tile_f)
        offs_b = tl.arange(0, tile_b)
        k_mask = offs_k < n
        f_mask = offs_f < freq_len
        # basis as [K, F]: element (k, f) lives at k * freq_len + f.
        basis_off = offs_k[:, None] * freq_len + offs_f[None, :]
        basis_mask = k_mask[:, None] & f_mask[None, :]
        cos_kf = tl.load(cos_t + basis_off, mask=basis_mask, other=0.0)
        sin_kf = tl.load(sin_t + basis_off, mask=basis_mask, other=0.0)
        for block_idx in range(pid, num_blocks, num_core):
            a = block_idx // num_b_blocks
            bb = block_idx - a * num_b_blocks
            b_idx = bb * tile_b + offs_b
            b_mask = b_idx < n_tail
            fb_off = a * freq_len * n_tail + offs_f[:, None] * n_tail + b_idx[None, :]
            fb_mask = f_mask[:, None] & b_mask[None, :]
            gr_fb = tl.load(grad_r + fb_off, mask=fb_mask, other=0.0)  # [F, B]
            gi_fb = tl.load(grad_i + fb_off, mask=fb_mask, other=0.0)
            acc = tl.dot(cos_kf, gr_fb) + tl.dot(sin_kf, gi_fb)  # [K, B]
            out_off = a * n * n_tail + offs_k[:, None] * n_tail + b_idx[None, :]
            out_mask = k_mask[:, None] & b_mask[None, :]
            tl.store(grad_x + out_off, acc, mask=out_mask)

    @triton.jit
    def _irfft_axis_matmul_kernel(
        xr, xi, icos, isin, out,
        n_lead: tl.constexpr, n: tl.constexpr, freq_len: tl.constexpr, n_tail: tl.constexpr,
        num_blocks: tl.constexpr,
        tile_b: tl.constexpr, tile_t: tl.constexpr, tile_f: tl.constexpr,
    ):
        # out[a, t, b] = sum_f icos[t, f] * xr[a, f, b] - isin[t, f] * xi[a, f, b]
        # icos/isin are the irfft basis in natural (n, freq_len) layout.
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_b_blocks = tl.cdiv(n_tail, tile_b)
        offs_t = tl.arange(0, tile_t)
        offs_f = tl.arange(0, tile_f)
        offs_b = tl.arange(0, tile_b)
        t_mask = offs_t < n
        f_mask = offs_f < freq_len
        # basis as [T, F]: element (t, f) lives at t * freq_len + f.
        basis_off = offs_t[:, None] * freq_len + offs_f[None, :]
        basis_mask = t_mask[:, None] & f_mask[None, :]
        icos_tf = tl.load(icos + basis_off, mask=basis_mask, other=0.0)
        isin_tf = tl.load(isin + basis_off, mask=basis_mask, other=0.0)
        for block_idx in range(pid, num_blocks, num_core):
            a = block_idx // num_b_blocks
            bb = block_idx - a * num_b_blocks
            b_idx = bb * tile_b + offs_b
            b_mask = b_idx < n_tail
            fb_off = a * freq_len * n_tail + offs_f[:, None] * n_tail + b_idx[None, :]
            fb_mask = f_mask[:, None] & b_mask[None, :]
            xr_fb = tl.load(xr + fb_off, mask=fb_mask, other=0.0)  # [F, B]
            xi_fb = tl.load(xi + fb_off, mask=fb_mask, other=0.0)
            acc = tl.dot(icos_tf, xr_fb) - tl.dot(isin_tf, xi_fb)  # [T, B]
            out_off = a * n * n_tail + offs_t[:, None] * n_tail + b_idx[None, :]
            out_mask = t_mask[:, None] & b_mask[None, :]
            tl.store(out + out_off, acc, mask=out_mask)

    @triton.jit
    def _irfft_axis_backward_kernel(
        grad_y, icos, isin, grad_r, grad_i,
        n_lead: tl.constexpr, n: tl.constexpr, freq_len: tl.constexpr, n_tail: tl.constexpr,
        num_blocks: tl.constexpr,
        tile_b: tl.constexpr, tile_f: tl.constexpr, tile_t: tl.constexpr,
    ):
        # grad_r[a, f, b] =  sum_t icos[t, f] * grad_y[a, t, b]
        # grad_i[a, f, b] = -sum_t isin[t, f] * grad_y[a, t, b]
        # icos/isin are the irfft basis in natural (n, freq_len) layout.
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        num_b_blocks = tl.cdiv(n_tail, tile_b)
        offs_f = tl.arange(0, tile_f)
        offs_t = tl.arange(0, tile_t)
        offs_b = tl.arange(0, tile_b)
        f_mask = offs_f < freq_len
        t_mask = offs_t < n
        # basis as [F, T]: element (f, t) lives at t * freq_len + f.
        basis_off = offs_t[None, :] * freq_len + offs_f[:, None]
        basis_mask = t_mask[None, :] & f_mask[:, None]
        icos_ft = tl.load(icos + basis_off, mask=basis_mask, other=0.0)
        isin_ft = tl.load(isin + basis_off, mask=basis_mask, other=0.0)
        for block_idx in range(pid, num_blocks, num_core):
            a = block_idx // num_b_blocks
            bb = block_idx - a * num_b_blocks
            b_idx = bb * tile_b + offs_b
            b_mask = b_idx < n_tail
            tb_off = a * n * n_tail + offs_t[:, None] * n_tail + b_idx[None, :]
            tb_mask = t_mask[:, None] & b_mask[None, :]
            gy_tb = tl.load(grad_y + tb_off, mask=tb_mask, other=0.0)  # [T, B]
            acc_r = tl.dot(icos_ft, gy_tb)   # [F, B]
            acc_i = -tl.dot(isin_ft, gy_tb)
            out_off = a * freq_len * n_tail + offs_f[:, None] * n_tail + b_idx[None, :]
            out_mask = f_mask[:, None] & b_mask[None, :]
            tl.store(grad_r + out_off, acc_r, mask=out_mask)
            tl.store(grad_i + out_off, acc_i, mask=out_mask)


def _torch_rfft_real(x: torch.Tensor, dim: int, norm: str):
    xk = torch.fft.rfft(x, dim=dim, norm=norm)
    return xk.real, xk.imag


def _torch_irfft_real(xr: torch.Tensor, xi: torch.Tensor, n: int, dim: int, norm: str):
    return torch.fft.irfft(torch.complex(xr, xi), n=n, dim=dim, norm=norm)


def _real_rfft_npu_forward(
    x: torch.Tensor,
    dim: int,
    norm: str,
    block_n: int | None,
    block_m: int,
    block_k: int,
):
    dim = dim % x.ndim
    n = x.shape[dim]
    freq_len = n // 2 + 1

    # Transpose-free path when the FFT axis is not innermost: contract the middle
    # axis of the natural (A, n, B) view directly, no movedim/contiguous copies.
    if dim != x.ndim - 1:
        n_lead = math.prod(x.shape[:dim])
        n_tail = math.prod(x.shape[dim + 1:])
        x_3d = x.contiguous().reshape(n_lead, n, n_tail)
        out_r = torch.empty((n_lead, freq_len, n_tail), device=x.device, dtype=torch.float32)
        out_i = torch.empty_like(out_r)
        rfft_cos, rfft_sin, _, _ = make_basis(n, x.device, norm)
        block_b = block_m or _default_block_m()
        block_f = block_n or _round_up(freq_len)
        block_k = block_k or _round_up(n)
        # 轴上 kernel 一次 tl.dot 覆盖整段 K/F，不能缩小 tile_k/tile_f。
        block_b = _cap_block_m_for_l0c(block_b, block_f, num_acc=2)
        block_b = _cap_block_m_for_cbuf(block_b, block_k, block_f)

        def _launch(bb):
            _rfft_axis_matmul_kernel[(_get_aicore_num(),)](
                x_3d, rfft_cos, rfft_sin, out_r, out_i,
                n_lead, n, freq_len, n_tail, n_lead * triton.cdiv(n_tail, bb),
                tile_b=bb, tile_f=block_f, tile_k=block_k,
            )

        _launch_with_block_m_backoff(
            _launch, block_b, cache_key=("rfft_axis_fwd", n, freq_len, block_f, block_k, str(x.device))
        )
        out_shape = (*x.shape[:dim], freq_len, *x.shape[dim + 1:])
        return out_r.reshape(out_shape), out_i.reshape(out_shape)

    x_last = x.movedim(dim, -1).contiguous()
    rows = x_last.numel() // n
    x_2d = x_last.reshape(rows, n)

    out_r = torch.empty((rows, freq_len), device=x.device, dtype=torch.float32)
    out_i = torch.empty_like(out_r)
    rfft_cos, rfft_sin, _, _ = make_basis(n, x.device, norm)
    # This is a tall-skinny GEMM (rows x n) @ (n x freq_len) with tiny n/freq_len.
    # Cover the whole freq (N) and time (K) axes in a single tl.dot per row-tile so
    # each core runs one large dot instead of hundreds of thousands of 16x16 dots.
    block_n = block_n or _round_up(freq_len)
    block_k = block_k or _round_up(n)
    block_m = block_m or _default_block_m()
    block_m, block_k, block_n = _cap_fft_tiles(block_m, block_k, block_n, num_acc=2)

    def _launch(bm):
        _rfft_matmul_kernel[(_get_aicore_num(),)](
            x_2d,
            rfft_cos,
            rfft_sin,
            out_r,
            out_i,
            rows,
            n,
            freq_len,
            triton.cdiv(rows, bm) * triton.cdiv(freq_len, block_n),
            tile_m=bm,
            tile_n=block_n,
            tile_k=block_k,
        )

    _launch_with_block_m_backoff(
        _launch, block_m, cache_key=("rfft_fwd", n, freq_len, block_n, block_k, str(x.device))
    )

    out_shape = (*x_last.shape[:-1], freq_len)
    return (
        out_r.reshape(out_shape).movedim(-1, dim).contiguous(),
        out_i.reshape(out_shape).movedim(-1, dim).contiguous(),
    )


def _real_rfft_npu_backward(
    grad_r: torch.Tensor,
    grad_i: torch.Tensor,
    n: int,
    dim: int,
    norm: str,
    block_n: int | None,
    block_m: int,
    block_k: int,
):
    dim = dim % grad_r.ndim
    freq_len = n // 2 + 1

    if dim != grad_r.ndim - 1:
        n_lead = math.prod(grad_r.shape[:dim])
        n_tail = math.prod(grad_r.shape[dim + 1:])
        gr_3d = grad_r.contiguous().reshape(n_lead, freq_len, n_tail)
        gi_3d = grad_i.contiguous().reshape(n_lead, freq_len, n_tail)
        grad_x = torch.empty((n_lead, n, n_tail), device=grad_r.device, dtype=torch.float32)
        rfft_cos, rfft_sin, _, _ = make_basis(n, grad_r.device, norm)
        block_b = block_m or _default_block_m()
        block_k = block_n or _round_up(n)
        block_f = _round_up(freq_len)
        block_b = _cap_block_m_for_l0c(block_b, block_k, num_acc=1)
        block_b = _cap_block_m_for_cbuf(block_b, block_f, block_k)

        def _launch(bb):
            _rfft_axis_backward_kernel[(_get_aicore_num(),)](
                gr_3d, gi_3d, rfft_cos, rfft_sin, grad_x,
                n_lead, n, freq_len, n_tail, n_lead * triton.cdiv(n_tail, bb),
                tile_b=bb, tile_k=block_k, tile_f=block_f,
            )

        _launch_with_block_m_backoff(
            _launch, block_b, cache_key=("rfft_axis_bwd", n, freq_len, block_k, block_f, str(grad_r.device))
        )
        out_shape = (*grad_r.shape[:dim], n, *grad_r.shape[dim + 1:])
        return grad_x.reshape(out_shape)

    grad_r_last = grad_r.movedim(dim, -1).contiguous()
    grad_i_last = grad_i.movedim(dim, -1).contiguous()
    rows = grad_r_last.numel() // freq_len
    grad_r_2d = grad_r_last.reshape(rows, freq_len)
    grad_i_2d = grad_i_last.reshape(rows, freq_len)

    grad_x = torch.empty((rows, n), device=grad_r.device, dtype=torch.float32)
    rfft_cos, rfft_sin, _, _ = make_basis(n, grad_r.device, norm)
    # (rows x freq_len) @ (freq_len x n): cover N=n and K=freq_len in one dot/tile.
    block_n = block_n or _round_up(n)
    block_k = block_k or _round_up(freq_len)
    block_m = block_m or _default_block_m()
    block_m, block_k, block_n = _cap_fft_tiles(block_m, block_k, block_n, num_acc=1)
    rfft_cos_t = rfft_cos.transpose(0, 1).contiguous()
    rfft_sin_t = rfft_sin.transpose(0, 1).contiguous()

    def _launch(bm):
        _rfft_backward_matmul_kernel[(_get_aicore_num(),)](
            grad_r_2d,
            grad_i_2d,
            rfft_cos_t,
            rfft_sin_t,
            grad_x,
            rows,
            n,
            freq_len,
            triton.cdiv(rows, bm) * triton.cdiv(n, block_n),
            tile_m=bm,
            tile_n=block_n,
            tile_k=block_k,
        )

    _launch_with_block_m_backoff(
        _launch, block_m, cache_key=("rfft_bwd", n, freq_len, block_n, block_k, str(grad_r.device))
    )

    out_shape = (*grad_r_last.shape[:-1], n)
    return grad_x.reshape(out_shape).movedim(-1, dim).contiguous()


def _real_irfft_npu_forward(
    xr: torch.Tensor,
    xi: torch.Tensor,
    n: int,
    dim: int,
    norm: str,
    block_k: int | None,
    block_m: int,
    block_n: int,
):
    dim = dim % xr.ndim
    freq_len = n // 2 + 1
    if xr.shape[dim] != freq_len:
        raise ValueError(f"frequency dimension must be n//2+1={freq_len}, got {xr.shape[dim]}")

    if dim != xr.ndim - 1:
        n_lead = math.prod(xr.shape[:dim])
        n_tail = math.prod(xr.shape[dim + 1:])
        xr_3d = xr.contiguous().reshape(n_lead, freq_len, n_tail)
        xi_3d = xi.contiguous().reshape(n_lead, freq_len, n_tail)
        out = torch.empty((n_lead, n, n_tail), device=xr.device, dtype=xr.dtype)
        _, _, irfft_cos, irfft_sin = make_basis(n, xr.device, norm)
        block_b = block_m or _default_block_m()
        block_t = block_n or _round_up(n)
        block_f = block_k or _round_up(freq_len)
        block_b = _cap_block_m_for_l0c(block_b, block_t, num_acc=1)
        block_b = _cap_block_m_for_cbuf(block_b, block_f, block_t)

        def _launch(bb):
            _irfft_axis_matmul_kernel[(_get_aicore_num(),)](
                xr_3d, xi_3d, irfft_cos, irfft_sin, out,
                n_lead, n, freq_len, n_tail, n_lead * triton.cdiv(n_tail, bb),
                tile_b=bb, tile_t=block_t, tile_f=block_f,
            )

        _launch_with_block_m_backoff(
            _launch, block_b, cache_key=("irfft_axis_fwd", n, freq_len, block_t, block_f, str(xr.device))
        )
        out_shape = (*xr.shape[:dim], n, *xr.shape[dim + 1:])
        return out.reshape(out_shape)

    xr_last = xr.movedim(dim, -1).contiguous()
    xi_last = xi.movedim(dim, -1).contiguous()
    rows = xr_last.numel() // freq_len
    xr_2d = xr_last.reshape(rows, freq_len)
    xi_2d = xi_last.reshape(rows, freq_len)

    out = torch.empty((rows, n), device=xr.device, dtype=xr.dtype)
    _, _, irfft_cos, irfft_sin = make_basis(n, xr.device, norm)
    # (rows x freq_len) @ (freq_len x n): cover N=n and K=freq_len in one dot/tile.
    block_n = block_n or _round_up(n)
    block_k = block_k or _round_up(freq_len)
    block_m = block_m or _default_block_m()
    block_m, block_k, block_n = _cap_fft_tiles(block_m, block_k, block_n, num_acc=1)
    irfft_cos_t = irfft_cos.transpose(0, 1).contiguous()
    irfft_sin_t = irfft_sin.transpose(0, 1).contiguous()

    def _launch(bm):
        _irfft_matmul_kernel[(_get_aicore_num(),)](
            xr_2d,
            xi_2d,
            irfft_cos_t,
            irfft_sin_t,
            out,
            rows,
            n,
            freq_len,
            triton.cdiv(rows, bm) * triton.cdiv(n, block_n),
            tile_m=bm,
            tile_n=block_n,
            tile_k=block_k,
        )

    _launch_with_block_m_backoff(
        _launch, block_m, cache_key=("irfft_fwd", n, freq_len, block_n, block_k, str(xr.device))
    )

    out_shape = (*xr_last.shape[:-1], n)
    return out.reshape(out_shape).movedim(-1, dim).contiguous()


def _real_irfft_npu_backward(
    grad_y: torch.Tensor,
    n: int,
    dim: int,
    norm: str,
    block_k: int | None,
    block_m: int,
    block_n: int,
):
    dim = dim % grad_y.ndim
    freq_len = n // 2 + 1

    if dim != grad_y.ndim - 1:
        n_lead = math.prod(grad_y.shape[:dim])
        n_tail = math.prod(grad_y.shape[dim + 1:])
        gy_3d = grad_y.contiguous().reshape(n_lead, n, n_tail)
        grad_r = torch.empty((n_lead, freq_len, n_tail), device=grad_y.device, dtype=torch.float32)
        grad_i = torch.empty_like(grad_r)
        _, _, irfft_cos, irfft_sin = make_basis(n, grad_y.device, norm)
        block_b = block_m or _default_block_m()
        block_f = block_n or _round_up(freq_len)
        block_t = block_k or _round_up(n)
        block_b = _cap_block_m_for_l0c(block_b, block_f, num_acc=2)
        block_b = _cap_block_m_for_cbuf(block_b, block_t, block_f)

        def _launch(bb):
            _irfft_axis_backward_kernel[(_get_aicore_num(),)](
                gy_3d, irfft_cos, irfft_sin, grad_r, grad_i,
                n_lead, n, freq_len, n_tail, n_lead * triton.cdiv(n_tail, bb),
                tile_b=bb, tile_f=block_f, tile_t=block_t,
            )

        _launch_with_block_m_backoff(
            _launch, block_b, cache_key=("irfft_axis_bwd", n, freq_len, block_f, block_t, str(grad_y.device))
        )
        out_shape = (*grad_y.shape[:dim], freq_len, *grad_y.shape[dim + 1:])
        return grad_r.reshape(out_shape), grad_i.reshape(out_shape)

    grad_y_last = grad_y.movedim(dim, -1).contiguous()
    rows = grad_y_last.numel() // n
    grad_y_2d = grad_y_last.reshape(rows, n)

    grad_r = torch.empty((rows, freq_len), device=grad_y.device, dtype=torch.float32)
    grad_i = torch.empty_like(grad_r)
    _, _, irfft_cos, irfft_sin = make_basis(n, grad_y.device, norm)
    # (rows x n) @ (n x freq_len): cover N=freq_len and K=n in one dot/tile.
    block_n = block_n or _round_up(freq_len)
    block_k = block_k or _round_up(n)
    block_m = block_m or _default_block_m()
    block_m, block_k, block_n = _cap_fft_tiles(block_m, block_k, block_n, num_acc=2)

    def _launch(bm):
        _irfft_backward_matmul_kernel[(_get_aicore_num(),)](
            grad_y_2d,
            irfft_cos,
            irfft_sin,
            grad_r,
            grad_i,
            rows,
            n,
            freq_len,
            triton.cdiv(rows, bm) * triton.cdiv(freq_len, block_n),
            tile_m=bm,
            tile_n=block_n,
            tile_k=block_k,
        )

    _launch_with_block_m_backoff(
        _launch, block_m, cache_key=("irfft_bwd", n, freq_len, block_n, block_k, str(grad_y.device))
    )

    out_shape = (*grad_y_last.shape[:-1], freq_len)
    return (
        grad_r.reshape(out_shape).movedim(-1, dim).contiguous(),
        grad_i.reshape(out_shape).movedim(-1, dim).contiguous(),
    )


if _has_triton():

    class _RealRFFTFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, dim, norm, block_n, block_m, block_k):
            dim = dim % x.ndim
            ctx.dim = dim
            ctx.n = x.shape[dim]
            ctx.norm = norm
            ctx.block_n = block_n
            ctx.block_m = block_m
            ctx.block_k = block_k
            return _real_rfft_npu_forward(x, dim, norm, block_n, block_m, block_k)

        @staticmethod
        def backward(ctx, grad_r, grad_i):
            grad_x = _real_rfft_npu_backward(
                grad_r,
                grad_i,
                ctx.n,
                ctx.dim,
                ctx.norm,
                ctx.block_n,
                ctx.block_m,
                ctx.block_k,
            )
            # trailing None: dim / norm / block_* are non-tensor arguments
            input_grads = (grad_x, None, None, None, None, None)
            return input_grads


    class _RealIRFFTFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xr, xi, n, dim, norm, block_k, block_m, block_n):
            dim = dim % xr.ndim
            ctx.n = n
            ctx.dim = dim
            ctx.norm = norm
            ctx.block_k = block_k
            ctx.block_m = block_m
            ctx.block_n = block_n
            return _real_irfft_npu_forward(xr, xi, n, dim, norm, block_k, block_m, block_n)

        @staticmethod
        def backward(ctx, grad_y):
            grad_r, grad_i = _real_irfft_npu_backward(
                grad_y,
                ctx.n,
                ctx.dim,
                ctx.norm,
                ctx.block_k,
                ctx.block_m,
                ctx.block_n,
            )
            # trailing None: n / dim / norm / block_* are non-tensor arguments
            input_grads = (grad_r, grad_i, None, None, None, None, None, None)
            return input_grads


def real_rfft(
    x: torch.Tensor,
    dim: int,
    norm: str = "ortho",
    block_n: int | None = None,
    block_m: int | None = None,
    block_k: int | None = None,
):
    """Real-input rFFT returning real tensors `(real, imag)`.

    On Ascend NPU this uses Triton kernels and dense Fourier basis matrices. On
    other devices it falls back to torch.fft so CPU tests can use the same API.
    """
    if norm != "ortho":
        raise ValueError(f"only norm='ortho' is supported, got {norm!r}")
    if not (_is_npu_tensor(x) and _has_triton()):
        return _torch_rfft_real(x, dim=dim, norm=norm)
    return _RealRFFTFunction.apply(x, dim, norm, block_n, block_m, block_k)


def real_irfft(
    xr: torch.Tensor,
    xi: torch.Tensor,
    n: int,
    dim: int,
    norm: str = "ortho",
    block_k: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
):
    """Inverse real FFT from real/imag tensors to a real tensor."""
    if norm != "ortho":
        raise ValueError(f"only norm='ortho' is supported, got {norm!r}")
    if xr.shape != xi.shape:
        raise ValueError(f"xr and xi must have the same shape, got {xr.shape} and {xi.shape}")
    if not (_is_npu_tensor(xr) and _has_triton()):
        return _torch_irfft_real(xr, xi, n=n, dim=dim, norm=norm)
    return _RealIRFFTFunction.apply(xr, xi, n, dim, norm, block_k, block_m, block_n)
