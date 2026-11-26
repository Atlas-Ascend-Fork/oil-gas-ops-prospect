#!/usr/bin/env python3
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

"""ROIAlignRotated / ROIAlignRotatedGrad 的 NPU 精度、autograd 与性能对比。

基准：纯 torch 参考实现 `torch_roi_align_rotated`（逐行对齐 detectron2 CUDA 语义）。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../.."))
PYBIND_DIR = os.path.join(REPO_ROOT, "ascendc", "pybind")
sys.path.insert(0, PYBIND_DIR)

import _aclnn_cache  # noqa: E402,F401  必须在 import torch / ACL 前关缓存
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401  尽早注册 NPU 设备

_spec = importlib.util.spec_from_file_location("local_pkg", os.path.join(REPO_ROOT, "scripts", "local_pkg.py"))
if _spec is not None and _spec.loader is not None:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod.activate(REPO_ROOT)


def build_ext() -> None:
    import subprocess

    env = dict(os.environ)
    env["OIL_GAS_OPS_ASCENDC_OPS"] = "roi_align_rotated"
    subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "ascendc", "pybind_setup.py"), "build_ext", "--inplace", "--force"],
        cwd=PYBIND_DIR,
        check=True,
        env=env,
    )


def rel_err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a64 = a.detach().float().cpu()
    b64 = b.detach().float().cpu()
    max_abs = float((a64 - b64).abs().max())
    den = torch.linalg.norm(a64.reshape(-1)).clamp_min(1e-30)
    rel = float(torch.linalg.norm((a64 - b64).reshape(-1)) / den)
    return max_abs, rel


def load_extension():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("[SKIP] torch_npu 不可用，本测试需要 NPU 环境")
        return None
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("[SKIP] 未检测到 NPU 设备；精度/性能对比必须在真实 910B 上运行")
        return None
    try:
        import roi_align_rotated_npu as ext
    except ImportError as exc:
        print(f"[FAIL] 无法导入 roi_align_rotated_npu 扩展（先跑 --build-ext）: {exc}")
        return None
    return ext


def make_rois(n, c, h, w, k, angles, device, scale=1.0):
    rows = []
    for i in range(k):
        batch = i % max(n, 1)
        cx = (w - 1) * 0.5 + ((i % 3) - 1) * 2.0
        cy = (h - 1) * 0.5 + ((i % 4) - 2) * 2.0
        rw = max(w * 0.6 + (i % 5), 1.0)
        rh = max(h * 0.55 + (i % 3), 1.0)
        rows.append([batch, cx, cy, rw, rh, angles[i % len(angles)]])
    return torch.tensor(rows, device=device, dtype=torch.float32)


def check_precision(ext, rtol: float) -> list[float]:
    from oil_gas_ops_prospect.roi_align_rotated import torch_roi_align_rotated

    cases = [
        ((1, 1, 5, 5), 1, 4, 4, 0, 1.0, [0.0]),
        ((1, 3, 10, 10), 3, 5, 5, 2, 1.0, [0.0, 45.0, 90.0]),
        ((1, 4, 20, 20), 5, 7, 7, 1, 1.0, [30.0, -45.0, 180.0, 270.0, 0.0]),
        ((2, 8, 32, 32), 6, 7, 7, 0, 0.5, [0.0, 15.0, 90.0, 135.0, 270.0, -30.0]),
        ((1, 16, 40, 30), 4, 14, 14, 2, 1.0, [0.0, 22.5, 90.0, 180.0]),
    ]
    errors: list[float] = []
    torch.manual_seed(7)
    for shape, k, ph, pw, sr, scale, angles in cases:
        n, c, h, w = shape
        inp = torch.randn(*shape, device="npu", dtype=torch.float32)
        rois = make_rois(n, c, h, w, k, angles, "npu", scale)

        ref = torch_roi_align_rotated(inp.cpu(), rois.cpu(), (ph, pw), scale, sr)
        got = ext.roi_align_rotated_forward(inp, rois, ph, pw, scale, sr)
        torch.npu.synchronize()
        fwd_max, fwd_rel = rel_err(ref, got)

        # grad_output 必须在 NPU 上；CPU 张量直接进 kernel 会得到非法地址。
        grad_out = torch.randn_like(got)
        got_bwd = ext.roi_align_rotated_backward(grad_out, rois, n, c, h, w, ph, pw, scale, sr)
        torch.npu.synchronize()

        inp_ref = inp.cpu().detach().requires_grad_(True)
        ref_out = torch_roi_align_rotated(inp_ref, rois.cpu(), (ph, pw), scale, sr)
        (ref_bwd,) = torch.autograd.grad(ref_out, inp_ref, grad_out.cpu())
        bwd_max, bwd_rel = rel_err(ref_bwd, got_bwd)

        print(
            f"[precision] {shape} k={k} pool={ph}x{pw} sr={sr} scale={scale} "
            f"fwd(max={fwd_max:.3e},rel={fwd_rel:.3e}) "
            f"bwd(max={bwd_max:.3e},rel={bwd_rel:.3e})"
        )
        errors.extend([fwd_max, fwd_rel, bwd_max, bwd_rel])
        if fwd_max > rtol or bwd_max > rtol:
            raise RuntimeError(f"precision failed: {shape}, fwd={fwd_max}, bwd={bwd_max}, rtol={rtol}")
    return errors


def check_autograd() -> list[float]:
    os.environ["OIL_GAS_OPS_ROI_ALIGN_ROTATED_IMPL"] = "ascendc"
    os.environ["OIL_GAS_OPS_REQUIRE_ASCENDC_ROI_ALIGN_ROTATED"] = "1"
    from oil_gas_ops_prospect.roi_align_rotated import ROIAlignRotated, roi_align_rotated, torch_roi_align_rotated

    inp = torch.randn(1, 3, 12, 12, device="npu", requires_grad=True)
    rois = make_rois(1, 3, 12, 12, 3, [0.0, 30.0, 90.0], "npu")
    out = roi_align_rotated(inp, rois, (5, 5), 1.0, 2)
    grad_out = torch.randn_like(out)
    torch.autograd.backward(out, grad_out)
    torch.npu.synchronize()

    inp_ref = inp.detach().cpu().requires_grad_(True)
    ref_out = torch_roi_align_rotated(inp_ref, rois.cpu(), (5, 5), 1.0, 2)
    (ref_grad,) = torch.autograd.grad(ref_out, inp_ref, grad_out.cpu())
    max_abs, rel = rel_err(ref_grad, inp.grad)
    print(f"[autograd] grad max={max_abs:.3e} rel={rel:.3e}")

    module = ROIAlignRotated((5, 5), 1.0, 2)
    module_out = module(inp.detach(), rois)
    if tuple(module_out.shape) != (3, 3, 5, 5):
        raise RuntimeError(f"module output shape mismatch: {tuple(module_out.shape)}")
    print(f"[module] out shape={tuple(module_out.shape)}")
    return [max_abs, rel]


def check_fp16_and_empty() -> None:
    from oil_gas_ops_prospect.roi_align_rotated import roi_align_rotated

    inp = torch.randn(1, 2, 10, 10, device="npu", dtype=torch.float16)
    rois = make_rois(1, 2, 10, 10, 2, [0.0, 45.0], "npu")
    out = roi_align_rotated(inp, rois, (4, 4), 1.0, 1)
    torch.npu.synchronize()
    if out.dtype != torch.float16 or tuple(out.shape) != (2, 2, 4, 4):
        raise RuntimeError(f"fp16 path mismatch: dtype={out.dtype}, shape={tuple(out.shape)}")
    print(f"[fp16] dtype={out.dtype} shape={tuple(out.shape)}")

    empty = roi_align_rotated(torch.randn(1, 2, 10, 10, device="npu"), torch.zeros(0, 6, device="npu"), (4, 4))
    if tuple(empty.shape) != (0, 2, 4, 4):
        raise RuntimeError(f"empty ROI mismatch: {tuple(empty.shape)}")
    print(f"[empty] shape={tuple(empty.shape)}")


def bench(fn, n_warm=5, n_iter=20):
    for _ in range(n_warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e3


def _vectorized_roi_align_rotated(input, rois, ph, pw, scale, sr):
    """纯 torch 向量化参考（用于性能对拍；语义已与逐 ROI 参考/官方 CPU golden 对齐）。"""
    n, c, h, w = input.shape
    k = rois.shape[0]
    if k == 0 or h == 0 or w == 0:
        return input.new_zeros((k, c, ph, pw))
    r = rois.detach().to(input.device, input.dtype)
    n_idx = r[:, 0].long()
    cx, cy, rw, rh, ang = r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5]
    theta = ang * torch.pi / 180.0
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    center_w = cx * scale - 0.5
    center_h = cy * scale - 0.5
    roi_w = rw * scale
    roi_h = rh * scale
    bin_w = roi_w / pw
    bin_h = roi_h / ph
    if sr > 0:
        grid_h = torch.full((k,), float(sr), device=input.device, dtype=input.dtype)
        grid_w = grid_h.clone()
    else:
        grid_h = torch.ceil(roi_h / ph)
        grid_w = torch.ceil(roi_w / pw)
    max_gh = max(1, int(grid_h.max().item()))
    max_gw = max(1, int(grid_w.max().item()))
    ph_idx = torch.arange(ph, device=input.device, dtype=input.dtype).view(1, ph, 1, 1, 1)
    pw_idx = torch.arange(pw, device=input.device, dtype=input.dtype).view(1, 1, pw, 1, 1)
    iy = torch.arange(max_gh, device=input.device, dtype=input.dtype).view(1, 1, 1, max_gh, 1)
    ix = torch.arange(max_gw, device=input.device, dtype=input.dtype).view(1, 1, 1, 1, max_gw)
    roi_h5 = roi_h.view(k, 1, 1, 1, 1)
    roi_w5 = roi_w.view(k, 1, 1, 1, 1)
    bin_h5 = bin_h.view(k, 1, 1, 1, 1)
    bin_w5 = bin_w.view(k, 1, 1, 1, 1)
    grid_h5 = grid_h.view(k, 1, 1, 1, 1)
    grid_w5 = grid_w.view(k, 1, 1, 1, 1)
    cos5, sin5 = cos_t.view(k, 1, 1, 1, 1), sin_t.view(k, 1, 1, 1, 1)
    center_h5, center_w5 = center_h.view(k, 1, 1, 1, 1), center_w.view(k, 1, 1, 1, 1)
    yy = -roi_h5 / 2.0 + ph_idx * bin_h5 + (iy + 0.5) * bin_h5 / grid_h5
    xx = -roi_w5 / 2.0 + pw_idx * bin_w5 + (ix + 0.5) * bin_w5 / grid_w5
    y = yy * cos5 - xx * sin5 + center_h5
    x = yy * sin5 + xx * cos5 + center_w5
    valid = (
        (iy < grid_h5)
        & (ix < grid_w5)
        & (y >= -1.0)
        & (y <= float(h))
        & (x >= -1.0)
        & (x <= float(w))
    )
    yc = y.clamp(min=0.0, max=float(h - 1))
    xc = x.clamp(min=0.0, max=float(w - 1))
    y_low = torch.floor(yc).long()
    x_low = torch.floor(xc).long()
    y_high = (y_low + 1).clamp(max=h - 1)
    x_high = (x_low + 1).clamp(max=w - 1)
    ly = yc - y_low.to(input.dtype)
    lx = xc - x_low.to(input.dtype)

    flat = input.reshape(n, c, h * w)
    n5 = n_idx.view(k, 1, 1, 1, 1)

    def gather(yy_idx, xx_idx):
        idx = (yy_idx * w + xx_idx).reshape(k, ph, pw, max_gh, max_gw)
        return flat[n5, :, idx]

    v1 = gather(y_low, x_low)
    v2 = gather(y_low, x_high)
    v3 = gather(y_high, x_low)
    v4 = gather(y_high, x_high)
    w1 = ((1.0 - ly) * (1.0 - lx)).unsqueeze(-1)
    w2 = ((1.0 - ly) * lx).unsqueeze(-1)
    w3 = (ly * (1.0 - lx)).unsqueeze(-1)
    w4 = (ly * lx).unsqueeze(-1)
    value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    value = torch.where(valid.unsqueeze(-1), value, torch.zeros_like(value))
    count = (grid_h * grid_w).clamp(min=1.0).view(k, 1, 1, 1, 1)
    out = value.sum(dim=(3, 4)) / count.view(k, 1, 1, 1)
    return out.permute(0, 3, 1, 2)


def report_perf(ext) -> None:
    cases = [
        ((1, 4, 20, 20), 8, 5, 5, 0, 1.0, [0.0, 30.0, 90.0]),
        ((1, 16, 40, 40), 64, 7, 7, 0, 1.0, [0.0, 30.0, 90.0]),
        ((1, 32, 64, 64), 128, 7, 7, 1, 1.0, [30.0, -45.0, 90.0]),
    ]
    torch.manual_seed(11)
    for shape, k, ph, pw, sr, scale, angles in cases:
        n, c, h, w = shape
        inp = torch.randn(*shape, device="npu", dtype=torch.float32)
        rois = make_rois(n, c, h, w, k, angles, "npu", scale)
        grad_out = torch.randn(k, c, ph, pw, device="npu")

        # 先确认 perf shape 上自定义算子输出正确，避免"测得快但没算"。
        ref_vec = _vectorized_roi_align_rotated(inp, rois, ph, pw, scale, sr)
        got = ext.roi_align_rotated_forward(inp, rois, ph, pw, scale, sr)
        torch.npu.synchronize()
        fwd_max, fwd_rel = rel_err(ref_vec, got)
        if fwd_rel > 1e-5:
            raise RuntimeError(f"perf shape correctness failed: {shape}, max_abs={fwd_max}, rel={fwd_rel}")

        ms_asc_fwd = bench(lambda: ext.roi_align_rotated_forward(inp, rois, ph, pw, scale, sr))
        ms_asc_bwd = bench(lambda: ext.roi_align_rotated_backward(grad_out, rois, n, c, h, w, ph, pw, scale, sr))
        inp_ref = inp.detach().clone().requires_grad_(True)
        ms_ref_fwd = bench(
            lambda: _vectorized_roi_align_rotated(inp, rois, ph, pw, scale, sr), n_warm=2, n_iter=5
        )
        ms_ref_bwd = bench(
            lambda: torch.autograd.grad(
                _vectorized_roi_align_rotated(inp_ref, rois, ph, pw, scale, sr), inp_ref, grad_out
            ),
            n_warm=2,
            n_iter=5,
        )
        print(
            f"[perf] {shape} k={k} pool={ph}x{pw} sr={sr} "
            f"npu_vec_ref_fwd={ms_ref_fwd:.3f}ms asc_fwd={ms_asc_fwd:.3f}ms "
            f"speedup_fwd={ms_ref_fwd / ms_asc_fwd:.2f}x "
            f"npu_vec_ref_bwd={ms_ref_bwd:.3f}ms asc_bwd={ms_asc_bwd:.3f}ms "
            f"speedup_bwd={ms_ref_bwd / ms_asc_bwd:.2f}x (fwd_max={fwd_max:.2e},fwd_rel={fwd_rel:.2e})"
        )

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--build-ext", action="store_true", help="先就地编译 roi_align_rotated_npu 扩展")
    p.add_argument("--mode", choices=("all", "precision", "perf"), default="all")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.build_ext:
        build_ext()
    ext = load_extension()
    if ext is None:
        return 1

    torch.npu.set_device(0)
    errors: list[float] = []
    if args.mode in ("all", "precision"):
        errors = check_precision(ext, args.rtol)
        errors.extend(check_autograd())
        check_fp16_and_empty()
    if args.mode in ("all", "perf"):
        report_perf(ext)

    if args.mode != "perf" and errors and max(errors) > args.rtol:
        print(f"FAIL max_err={max(errors):.3e} > rtol={args.rtol}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
