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

"""用 detectron2 的 ROIAlignRotated **CPU C++ 实现** 做独立 golden 校验。

这与仓内 `torch_roi_align_rotated` 是两套独立实现，可避免"同一个理解错误抄两遍"。
需要 detectron2 源码目录（环境变量 ``DETECTRON2_ROOT``）；未设置或目录缺失时 SKIP。

用法：
    DETECTRON2_ROOT=/path/to/detectron2 \
        python3 tests/ascendc/roi_align_rotated/verify_detectron2_cpu_golden.py

默认只在 CPU 上比较"detectron2 CPU golden"与仓内 torch 参考实现；若本机有 NPU 且
roi_align_rotated_npu 扩展已编译，也会顺带比较自定义算子。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
PYBIND_DIR = REPO_ROOT / "ascendc" / "pybind"
sys.path.insert(0, str(PYBIND_DIR))

_spec = importlib.util.spec_from_file_location("local_pkg", REPO_ROOT / "scripts" / "local_pkg.py")
if _spec is not None and _spec.loader is not None:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod.activate(REPO_ROOT)

from oil_gas_ops_prospect.roi_align_rotated import torch_roi_align_rotated  # noqa: E402


def _build_with_gxx(layers_dir: Path, cpu_src: Path, binding: Path):
    """无 ninja 时用 g++ 直接编译（与 distutils 后端等价）。"""
    import importlib.util
    import subprocess
    import sysconfig

    from torch.utils.cpp_extension import include_paths, library_paths

    build_dir = Path(tempfile.gettempdir()) / "og_d2_roi_align_rotated_cpu"
    build_dir.mkdir(parents=True, exist_ok=True)
    name = "detectron2_roi_align_rotated_cpu_golden"
    so_path = build_dir / f"{name}.so"
    py_include = sysconfig.get_paths()["include"]
    py_libdir = sysconfig.get_config_var("LIBDIR")
    py_ldlibrary = sysconfig.get_config_var("LDLIBRARY") or "libpython3.11.so"
    py_lib = py_ldlibrary.split(".")[0]
    if py_lib.startswith("lib"):
        py_lib = py_lib[3:]

    cmd = [
        "g++",
        "-shared",
        "-fPIC",
        "-std=c++17",
        "-O2",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        f"-DTORCH_EXTENSION_NAME={name}",
        f"-I{layers_dir}",
        f"-I{py_include}",
    ]
    for path in include_paths():
        cmd.append(f"-I{path}")
    cmd += [str(binding), str(cpu_src)]
    for path in library_paths():
        cmd += [f"-L{path}", f"-Wl,-rpath,{path}"]
    if py_libdir:
        cmd += [f"-L{py_libdir}", f"-Wl,-rpath,{py_libdir}"]
    cmd += ["-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10", f"-l{py_lib}", "-o", str(so_path)]
    subprocess.run(cmd, check=True)

    spec = importlib.util.spec_from_file_location(name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_detectron2_cpu(detectron_root: Path):
    layers_dir = detectron_root / "detectron2" / "layers" / "csrc" / "ROIAlignRotated"
    cpu_src = layers_dir / "ROIAlignRotated_cpu.cpp"
    binding = THIS_DIR / "detectron2_cpu_binding.cpp"
    if not cpu_src.is_file() or not binding.is_file():
        raise FileNotFoundError(f"missing detectron2 source: {cpu_src}")

    try:
        from torch.utils.cpp_extension import load

        build_dir = Path(tempfile.gettempdir()) / "og_d2_roi_align_rotated_cpu"
        build_dir.mkdir(parents=True, exist_ok=True)
        return load(
            name="detectron2_roi_align_rotated_cpu_golden",
            sources=[str(binding), str(cpu_src)],
            extra_include_paths=[str(layers_dir)],
            extra_cflags=["-O2"],
            build_directory=str(build_dir),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[INFO] torch cpp_extension.load 失败（{exc}），回退 g++ 直编")
        return _build_with_gxx(layers_dir, cpu_src, binding)


def make_rois(n, h, w, k, angles, scale):
    rows = []
    for i in range(k):
        batch = i % max(n, 1)
        cx = (w - 1) * 0.5 + ((i % 3) - 1) * 2.0
        cy = (h - 1) * 0.5 + ((i % 4) - 2) * 2.0
        rw = max(w * 0.6 + (i % 5), 1.0)
        rh = max(h * 0.55 + (i % 3), 1.0)
        rows.append([batch, cx, cy, rw, rh, angles[i % len(angles)]])
    return torch.tensor(rows, dtype=torch.float32)


def stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a64 = a.detach().double().reshape(-1)
    b64 = b.detach().double().reshape(-1)
    max_abs = float((a64 - b64).abs().max())
    rel = float(torch.linalg.norm(a64 - b64) / torch.linalg.norm(a64).clamp_min(1e-30))
    return max_abs, rel


def compare_torch_reference(ext, dtype: torch.dtype, cases, npu_ext=None, check_npu: bool = False) -> list[float]:
    errors = []
    for shape, k, ph, pw, sr, scale, angles in cases:
        n, c, h, w = shape
        torch.manual_seed(17)
        inp = torch.randn(*shape, dtype=dtype)
        rois = make_rois(n, h, w, k, angles, scale).to(dtype)

        golden_fwd = ext.roi_align_rotated_forward_cpu(inp, rois, scale, ph, pw, sr)
        ref_fwd = torch_roi_align_rotated(inp, rois, (ph, pw), scale, sr)
        fwd_max, fwd_rel = stats(golden_fwd, ref_fwd)

        grad_out = torch.randn_like(golden_fwd)
        golden_bwd = ext.roi_align_rotated_backward_cpu(grad_out, rois, scale, ph, pw, n, c, h, w, sr)
        inp_ref = inp.detach().clone().requires_grad_(True)
        ref_out = torch_roi_align_rotated(inp_ref, rois, (ph, pw), scale, sr)
        (ref_bwd,) = torch.autograd.grad(ref_out, inp_ref, grad_out)
        bwd_max, bwd_rel = stats(golden_bwd, ref_bwd)

        print(
            f"[d2-golden] {shape} k={k} pool={ph}x{pw} sr={sr} scale={scale} dtype={dtype} "
            f"torch_ref_fwd(max={fwd_max:.3e},rel={fwd_rel:.3e}) "
            f"torch_ref_bwd(max={bwd_max:.3e},rel={bwd_rel:.3e})"
        )
        errors.extend([fwd_max, fwd_rel, bwd_max, bwd_rel])

        if check_npu and npu_ext is not None:
            inp_npu = inp.npu()
            rois_npu = rois.npu()
            got = npu_ext.roi_align_rotated_forward(inp_npu, rois_npu, ph, pw, scale, sr)
            torch.npu.synchronize()
            nf_max, nf_rel = stats(golden_fwd, got.cpu())
            grad_npu = grad_out.npu()
            got_bwd = npu_ext.roi_align_rotated_backward(grad_npu, rois_npu, n, c, h, w, ph, pw, scale, sr)
            torch.npu.synchronize()
            nb_max, nb_rel = stats(golden_bwd, got_bwd.cpu())
            print(
                f"[d2-golden][npu] {shape} k={k} pool={ph}x{pw} sr={sr} "
                f"npu_fwd(max={nf_max:.3e},rel={nf_rel:.3e}) "
                f"npu_bwd(max={nb_max:.3e},rel={nb_rel:.3e})"
            )
            errors.extend([nf_max, nf_rel, nb_max, nb_rel])
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detectron2-root", default=os.environ.get("DETECTRON2_ROOT", ""))
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--rtol-fp64", type=float, default=1e-10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root_text = args.detectron2_root.strip()
    if not root_text:
        print("[SKIP] DETECTRON2_ROOT 未设置；跳过 detectron2 CPU 独立 golden 校验")
        return 0
    detectron_root = Path(root_text)
    if not (detectron_root / "detectron2" / "layers" / "csrc" / "ROIAlignRotated").is_dir():
        print(f"[SKIP] 未找到 detectron2 源码: {detectron_root}")
        return 0

    try:
        ext = build_detectron2_cpu(detectron_root)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 编译 detectron2 CPU golden 失败: {exc}")
        return 1

    cases = [
        ((1, 1, 5, 5), 1, 4, 4, 0, 1.0, [0.0]),
        ((1, 3, 10, 10), 3, 5, 5, 2, 1.0, [0.0, 45.0, 90.0]),
        ((2, 8, 32, 32), 6, 7, 7, 0, 0.5, [0.0, 15.0, 90.0, 135.0, 270.0, -30.0]),
        ((1, 4, 20, 20), 5, 7, 7, 1, 1.0, [30.0, -45.0, 180.0, 270.0, 0.0]),
    ]
    npu_ext = None
    check_npu = False
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            import roi_align_rotated_npu as npu_ext  # type: ignore

            check_npu = True
    except Exception:  # noqa: BLE001
        npu_ext = None
        check_npu = False

    errors = compare_torch_reference(ext, torch.float32, cases, npu_ext, check_npu)
    errors += compare_torch_reference(ext, torch.float64, cases[:2])
    worst = max(errors) if errors else 0.0
    limit = max(args.rtol, args.rtol_fp64)
    if worst > limit:
        print(f"FAIL worst_err={worst:.3e} > limit={limit:.3e}")
        return 1
    print(f"PASS worst_err={worst:.3e} (detectron2 CPU golden)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
