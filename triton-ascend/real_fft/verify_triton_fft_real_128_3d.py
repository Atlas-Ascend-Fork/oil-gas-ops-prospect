#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, NamedTuple

import torch
import yaml

# Exact shape of the fused rFFT/iRFFT kernels seen in a profiled 3D spectral
# workload: latent grid (41, 41, 41), embed_dim 2048, batch 2, spectral axis 1
# (n=41 -> freq_len=21, rows=6_885_376). Use --run-latent-3d-profiled to check
# precision/perf/memory of the optimized DFT-matmul tiling on this exact workload.
LATENT_3D_PROFILED_SHAPE = (2, 41, 41, 41, 2048)
LATENT_3D_PROFILED_AXES = (1,)

# Exact spectral tensor of a 128^3 channels-first spectral-conv smoke run
# (128^3 volume, batch 1, width 35, modes 7/7/7, real-FFT spectral impl). The model is
# channels-first ``(B, C, D, H, W)`` and its SpectralConv3dReal does a full 3D
# rfftn decomposed into per-axis real rFFTs: dim=-1 (W, the last dim -> 2D
# last-axis kernel, the one that overflowed cbuf), then dim=-2 (H) and dim=-3
# (D) via the transpose-free axis kernels. Benchmarking rFFT along axes 4/3/2 of
# this exact tensor therefore reproduces all three per-axis kernels the model
# launches (n=128 each). Use --run-spectral-3d-smoke.
SPECTRAL_3D_SMOKE_SHAPE = (1, 35, 128, 128, 128)
SPECTRAL_3D_SMOKE_AXES = (4, 3, 2)

TRITON_ASCEND_ROOT = Path(__file__).resolve().parents[1]
if str(TRITON_ASCEND_ROOT) not in sys.path:
    sys.path.insert(0, str(TRITON_ASCEND_ROOT))

import _use_local_ops  # noqa: F401  (import side effect: prefer the in-repo sources)
from _local_ops_package import (
    MODEL_REPO_ENV,
    find_model_repo_root,
    ops_repo_root,
    require_model_repo_root,
)
from oil_gas_ops_prospect.triton.real_fft import real_irfft, real_rfft

OPS_REPO_ROOT = ops_repo_root()

# The downstream model repository, when this operator repo is used alongside one. It is
# only needed by --from-model-config, which infers the FFT shape from a downstream
# model YAML config. Everything else works standalone.
REPO_ROOT = find_model_repo_root()
if REPO_ROOT is not None and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional default for --from-model-config, so downstream projects can point at
# their own config without patching this script.
MODEL_CONFIG_ENV = "OIL_GAS_OPS_MODEL_CONFIG"

# --bench-3d compares against a downstream spectral-conv implementation, given as
# "<module>:<class>". Keeps the downstream module layout out of this repository.
SPECTRAL_CONV_ENV = "OIL_GAS_OPS_SPECTRAL_CONV"
DEFAULT_SPECTRAL_CONV = "src.models.backbones.s2no3d:SpectralConv3dReal"


@dataclass
class AxisRoundResult:
    axis: int
    round_idx: int
    cpu_elapsed_ms: float
    torch_npu_fwd_r_max_abs: float
    torch_npu_fwd_i_max_abs: float
    torch_npu_fwd_r_rel_l2: float
    torch_npu_fwd_i_rel_l2: float
    torch_npu_rfft_bwd_max_abs: float
    torch_npu_rfft_bwd_rel_l2: float
    torch_npu_irfft_fwd_max_abs: float
    torch_npu_irfft_fwd_rel_l2: float
    torch_npu_irfft_bwd_r_max_abs: float
    torch_npu_irfft_bwd_i_max_abs: float
    torch_npu_irfft_bwd_r_rel_l2: float
    torch_npu_irfft_bwd_i_rel_l2: float
    torch_npu_peak_memory_mb: float | None
    torch_npu_elapsed_ms: float
    fwd_r_max_abs: float
    fwd_i_max_abs: float
    fwd_r_rel_l2: float
    fwd_i_rel_l2: float
    rfft_bwd_max_abs: float
    rfft_bwd_rel_l2: float
    irfft_fwd_max_abs: float
    irfft_fwd_rel_l2: float
    irfft_bwd_r_max_abs: float
    irfft_bwd_i_max_abs: float
    irfft_bwd_r_rel_l2: float
    irfft_bwd_i_rel_l2: float
    fused_vs_torch_npu_fwd_r_max_abs: float
    fused_vs_torch_npu_fwd_i_max_abs: float
    fused_vs_torch_npu_fwd_r_rel_l2: float
    fused_vs_torch_npu_fwd_i_rel_l2: float
    fused_vs_torch_npu_rfft_bwd_max_abs: float
    fused_vs_torch_npu_rfft_bwd_rel_l2: float
    fused_vs_torch_npu_irfft_fwd_max_abs: float
    fused_vs_torch_npu_irfft_fwd_rel_l2: float
    fused_vs_torch_npu_irfft_bwd_r_max_abs: float
    fused_vs_torch_npu_irfft_bwd_i_max_abs: float
    fused_vs_torch_npu_irfft_bwd_r_rel_l2: float
    fused_vs_torch_npu_irfft_bwd_i_rel_l2: float
    peak_memory_mb: float | None
    elapsed_ms: float


def _maybe_npu_device(device_id: int = 0) -> torch.device:
    try:
        importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError("torch_npu is required for Triton-Ascend FFT verification") from exc
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise RuntimeError("NPU is not available")
    device = torch.device(f"npu:{device_id}")
    if hasattr(torch.npu, "set_device"):
        torch.npu.set_device(device)
    torch.empty((), device=device)
    return device


def _sync(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()


def _reset_peak_memory(device: torch.device) -> None:
    """Flat early-returns keep the nesting shallow (coding-standard depth limit)."""
    if device.type != "npu" or not hasattr(torch, "npu"):
        return
    if hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()
    if not hasattr(torch.npu, "reset_peak_memory_stats"):
        return
    try:
        torch.npu.reset_peak_memory_stats(device)
    except RuntimeError as exc:
        # Tolerated: the stats only exist once the NPU context is initialised.
        if "did you call init" not in str(exc):
            raise


def _peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "max_memory_allocated"):
        return float(torch.npu.max_memory_allocated(device) / 1024**2)
    return None


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor, eps: float = 1e-30) -> float:
    actual = actual.float()
    expected = expected.float()
    err = torch.linalg.vector_norm(actual - expected)
    ref = torch.linalg.vector_norm(expected).clamp_min(eps)
    return float((err / ref).item())


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _speedup_ms(baseline_ms: float, candidate_ms: float) -> float | None:
    if candidate_ms <= 0:
        return None
    return baseline_ms / candidate_ms


def _memory_delta_mb(baseline_mb: float | None, candidate_mb: float | None) -> float | None:
    if baseline_mb is None or candidate_mb is None:
        return None
    return candidate_mb - baseline_mb


def _shape(batch: int, size: int, channels: int) -> tuple[int, int, int, int, int]:
    return (batch, size, size, size, channels)


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_model_config(model_configs, model_id):
    if isinstance(model_configs, list):
        if isinstance(model_id, int) and 0 <= model_id < len(model_configs):
            return dict(model_configs[model_id])
        if len(model_configs) == 1:
            return dict(model_configs[0])
        raise IndexError(f"model_id={model_id!r} is invalid for {len(model_configs)} model configs")
    return dict(model_configs[str(model_id)])


def _as_3tuple(value) -> tuple[int, int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f"expected 3D size, got {value!r}")
        return tuple(int(v) for v in value)
    v = int(value)
    return (v, v, v)


def _patch_out_size_3d(img_size_3d: tuple[int, int, int], patch_size: int) -> tuple[int, int, int]:
    stride = max(1, patch_size // 2)
    padding = patch_size // 2
    return tuple((n + 2 * padding - patch_size) // stride + 1 for n in img_size_3d)


class SpectralConvRef(NamedTuple):
    """The downstream spectral-conv entry points used by ``--bench-3d``.

    ``rfftn`` / ``irfftn`` are the per-axis decomposition staticmethods. They are
    named with a leading underscore downstream, so they are looked up by name
    here (once, in :func:`_import_spectral_conv`) instead of being spelled as
    protected member accesses all over the benchmark.
    """

    cls: type
    has_real_fft: bool
    rfftn: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    irfftn: Callable[..., torch.Tensor]


def _import_spectral_conv() -> SpectralConvRef:
    """Import the downstream spectral-conv module compared against by --bench-3d.

    Configured as ``<module path>:<class name>`` so this operator repository does
    not hard-code any downstream module layout.
    """
    spec = os.environ.get(SPECTRAL_CONV_ENV, DEFAULT_SPECTRAL_CONV)
    if ":" not in spec:
        raise SystemExit(
            f"{SPECTRAL_CONV_ENV}={spec!r} must be '<module>:<class>', "
            f"e.g. 'my_models.spectral3d:SpectralConv3dReal'"
        )
    module_name, class_name = spec.rsplit(":", 1)

    require_model_repo_root("--bench-3d")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"--bench-3d could not import {module_name!r} ({exc}).\n"
            f"Point {MODEL_REPO_ENV} at the model repository and, if its module path "
            f"differs, set {SPECTRAL_CONV_ENV}=<module>:<class>."
        ) from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise SystemExit(f"{module_name} does not export {class_name!r}") from exc

    missing = [name for name in ("_rfftn", "_irfftn") if not hasattr(cls, name)]
    if missing:
        raise SystemExit(f"{module_name}.{class_name} does not provide {missing}")
    return SpectralConvRef(
        cls=cls,
        has_real_fft=bool(getattr(module, "_HAS_REAL_FFT", True)),
        rfftn=getattr(cls, "_rfftn"),
        irfftn=getattr(cls, "_irfftn"),
    )


def resolve_model_config(config_path: Path, *, option: str) -> Path:
    """Resolve a downstream model YAML path, with an actionable error when missing.

    A relative path is tried against the current directory first, then against
    the downstream model repository (when one is available).
    """

    candidates: list[Path] = []
    if config_path.is_absolute():
        candidates.append(config_path)
    else:
        candidates.append((Path.cwd() / config_path).resolve())
        main_repo = find_model_repo_root()
        if main_repo is not None:
            candidates.append((main_repo / config_path).resolve())

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise SystemExit(
        f"{option}: model config not found.\nTried:\n{tried}\n\n"
        f"Model YAML configs belong to the downstream model repository, not to this\n"
        f"operator repository ({OPS_REPO_ROOT}). Either:\n"
        f"  - point at a checkout: export {MODEL_REPO_ENV}=/path/to/model-repo\n"
        f"  - set a default config:  export {MODEL_CONFIG_ENV}=/abs/path/to/config.yaml\n"
        f"  - pass an absolute path: --from-model-config /abs/path/to/config.yaml\n"
        f"  - or skip config inference and give the shape directly, e.g.\n"
        f"      --shape 1 41 41 41 2048 --axes 1"
    )


def infer_fft_shape_from_model_config(config_path: Path) -> tuple[tuple[int, int, int, int, int], list[int], dict]:
    config = _load_yaml(config_path)
    fwd_config = dict(config.get("fwd_model", {}))
    model_yaml = fwd_config.get("model_yaml")
    if not model_yaml:
        raise ValueError(f"{config_path} does not configure fwd_model.model_yaml")

    model_path = (config_path.parent / model_yaml).resolve()
    if not model_path.exists() and REPO_ROOT is not None:
        model_path = (REPO_ROOT / model_yaml).resolve()
    if not model_path.exists():
        raise SystemExit(
            f"{config_path} references fwd_model.model_yaml={model_yaml!r}, which was not found.\n"
            f"Set {MODEL_REPO_ENV} to the model repository root so relative model YAMLs resolve."
        )
    model_defaults = _select_model_config(_load_yaml(model_path), fwd_config.get("model_id", 0))
    model_config = dict(model_defaults)
    model_config.update(fwd_config)

    seis = config.get("seis_data", {})
    raw_size = (
        int(seis.get("nz", model_config["img_size"])),
        int(seis.get("ny", model_config["img_size"])),
        int(seis.get("nx", model_config["img_size"])),
    )
    pml = int(model_config.get("pml_size", 0))
    padded = tuple(n + 2 * pml for n in raw_size)
    raw_patch = model_config["patch_size"]
    patch_size = int(raw_patch if isinstance(raw_patch, int) else raw_patch[0])
    latent = _patch_out_size_3d(padded, patch_size)
    batch = int(config.get("scheduler", {}).get("batch_size", 1))
    channels = int(model_config["embed_dim"])
    axes = [1, 2, 3] if model_config.get("slim_use_both_axes", False) else [1]
    metadata = {
        "config_path": str(config_path),
        "model_yaml": str(model_path),
        "model_id": fwd_config.get("model_id", 0),
        "raw_size": raw_size,
        "pml_size": pml,
        "padded_size": padded,
        "patch_size": patch_size,
        "latent_size": latent,
        "embed_dim": channels,
        "slim_use_both_axes": bool(model_config.get("slim_use_both_axes", False)),
        "default_axes": axes,
    }
    return (batch, *latent, channels), axes, metadata


class _RoundInputs(NamedTuple):
    """CPU-side random inputs shared by all three rounds of one axis check."""

    x: torch.Tensor
    grad_r: torch.Tensor
    grad_i: torch.Tensor
    grad_y: torch.Tensor
    spec_r: torch.Tensor
    spec_i: torch.Tensor


class _RoundOutputs(NamedTuple):
    """One rFFT+iRFFT forward/backward round, as CPU tensors plus its metrics."""

    spec_r: torch.Tensor
    spec_i: torch.Tensor
    x_grad: torch.Tensor
    irfft: torch.Tensor
    spec_r_grad: torch.Tensor
    spec_i_grad: torch.Tensor
    elapsed_ms: float
    peak_mb: float | None


# The six tensors compared for accuracy, in _RoundOutputs field order.
_METRIC_NAMES = ("fwd_r", "fwd_i", "rfft_bwd", "irfft_fwd", "irfft_bwd_r", "irfft_bwd_i")


def _make_round_inputs(shape, axis: int, seed: int) -> _RoundInputs:
    torch.manual_seed(seed)
    freq_shape = (*shape[:axis], shape[axis] // 2 + 1, *shape[axis + 1:])
    grad_r = torch.randn(freq_shape, dtype=torch.float32)
    return _RoundInputs(
        x=torch.randn(shape, dtype=torch.float32),
        grad_r=grad_r,
        grad_i=torch.randn_like(grad_r),
        grad_y=torch.randn(shape, dtype=torch.float32),
        spec_r=torch.randn_like(grad_r),
        spec_i=torch.randn_like(grad_r),
    )


def _cpu_reference_round(inp: _RoundInputs, axis: int, shape) -> _RoundOutputs:
    """``torch.fft`` on CPU: the ground truth both device rounds are scored against."""
    started = time.perf_counter()
    x_ref = inp.x.detach().clone().requires_grad_(True)
    spec = torch.fft.rfft(x_ref, dim=axis, norm="ortho")
    (spec.real * inp.grad_r + spec.imag * inp.grad_i).sum().backward()

    spec_r_ref = inp.spec_r.detach().clone().requires_grad_(True)
    spec_i_ref = inp.spec_i.detach().clone().requires_grad_(True)
    inverse = torch.fft.irfft(
        torch.complex(spec_r_ref, spec_i_ref), n=shape[axis], dim=axis, norm="ortho"
    )
    (inverse * inp.grad_y).sum().backward()
    return _RoundOutputs(
        spec_r=spec.real.detach(),
        spec_i=spec.imag.detach(),
        x_grad=x_ref.grad.detach(),
        irfft=inverse.detach(),
        spec_r_grad=spec_r_ref.grad.detach(),
        spec_i_grad=spec_i_ref.grad.detach(),
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        peak_mb=None,
    )


def _device_round(inp: _RoundInputs, axis: int, shape, device, *, fused: bool) -> _RoundOutputs:
    """One rFFT+iRFFT forward/backward round on ``device``.

    ``fused`` selects the Triton operators, otherwise native ``torch.fft``
    (torch_npu on NPU). Both paths are timed and measured identically so the
    numbers are directly comparable; the device tensors are released on return so
    each round's peak-memory reading is independent.
    """
    _reset_peak_memory(device)
    _sync(device)
    started = time.perf_counter()

    x = inp.x.to(device).requires_grad_(True)
    grad_r = inp.grad_r.to(device)
    grad_i = inp.grad_i.to(device)
    if fused:
        spec_r, spec_i = real_rfft(x, dim=axis, norm="ortho")
    else:
        spec = torch.fft.rfft(x, dim=axis, norm="ortho")
        spec_r, spec_i = spec.real, spec.imag
    (spec_r * grad_r + spec_i * grad_i).sum().backward()

    src_r = inp.spec_r.to(device).requires_grad_(True)
    src_i = inp.spec_i.to(device).requires_grad_(True)
    grad_y = inp.grad_y.to(device)
    if fused:
        inverse = real_irfft(src_r, src_i, n=shape[axis], dim=axis, norm="ortho")
    else:
        inverse = torch.fft.irfft(
            torch.complex(src_r, src_i), n=shape[axis], dim=axis, norm="ortho"
        )
    (inverse * grad_y).sum().backward()

    _sync(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return _RoundOutputs(
        spec_r=spec_r.detach().cpu(),
        spec_i=spec_i.detach().cpu(),
        x_grad=x.grad.detach().cpu(),
        irfft=inverse.detach().cpu(),
        spec_r_grad=src_r.grad.detach().cpu(),
        spec_i_grad=src_i.grad.detach().cpu(),
        elapsed_ms=elapsed_ms,
        peak_mb=_peak_memory_mb(device),
    )


def _accuracy_metrics(prefix: str, got: _RoundOutputs, ref: _RoundOutputs) -> dict:
    """``{prefix}{name}_max_abs`` / ``_rel_l2`` for the six compared tensors."""
    metrics = {}
    for name, actual, expected in zip(_METRIC_NAMES, got[:6], ref[:6]):
        metrics[f"{prefix}{name}_max_abs"] = _max_abs(actual, expected)
        metrics[f"{prefix}{name}_rel_l2"] = _rel_l2(actual, expected)
    return metrics


def run_axis_round(
    *,
    axis: int,
    round_idx: int,
    shape: tuple[int, int, int, int, int],
    device: torch.device,
    seed: int,
) -> AxisRoundResult:
    inp = _make_round_inputs(shape, axis, seed)
    ref = _cpu_reference_round(inp, axis, shape)
    native = _device_round(inp, axis, shape, device, fused=False)
    fused = _device_round(inp, axis, shape, device, fused=True)
    return AxisRoundResult(
        axis=axis,
        round_idx=round_idx,
        cpu_elapsed_ms=ref.elapsed_ms,
        torch_npu_peak_memory_mb=native.peak_mb,
        torch_npu_elapsed_ms=native.elapsed_ms,
        peak_memory_mb=fused.peak_mb,
        elapsed_ms=fused.elapsed_ms,
        **_accuracy_metrics("torch_npu_", native, ref),
        **_accuracy_metrics("", fused, ref),
        **_accuracy_metrics("fused_vs_torch_npu_", fused, native),
    )


def _assert_thresholds(results: list[AxisRoundResult], atol: float, rtol: float) -> None:
    failures: list[str] = []
    for r in results:
        checks = {
            "torch_npu_fwd_r": (r.torch_npu_fwd_r_max_abs, r.torch_npu_fwd_r_rel_l2),
            "torch_npu_fwd_i": (r.torch_npu_fwd_i_max_abs, r.torch_npu_fwd_i_rel_l2),
            "torch_npu_rfft_bwd": (r.torch_npu_rfft_bwd_max_abs, r.torch_npu_rfft_bwd_rel_l2),
            "torch_npu_irfft_fwd": (r.torch_npu_irfft_fwd_max_abs, r.torch_npu_irfft_fwd_rel_l2),
            "torch_npu_irfft_bwd_r": (r.torch_npu_irfft_bwd_r_max_abs, r.torch_npu_irfft_bwd_r_rel_l2),
            "torch_npu_irfft_bwd_i": (r.torch_npu_irfft_bwd_i_max_abs, r.torch_npu_irfft_bwd_i_rel_l2),
            "fused_fwd_r": (r.fwd_r_max_abs, r.fwd_r_rel_l2),
            "fused_fwd_i": (r.fwd_i_max_abs, r.fwd_i_rel_l2),
            "fused_rfft_bwd": (r.rfft_bwd_max_abs, r.rfft_bwd_rel_l2),
            "fused_irfft_fwd": (r.irfft_fwd_max_abs, r.irfft_fwd_rel_l2),
            "fused_irfft_bwd_r": (r.irfft_bwd_r_max_abs, r.irfft_bwd_r_rel_l2),
            "fused_irfft_bwd_i": (r.irfft_bwd_i_max_abs, r.irfft_bwd_i_rel_l2),
        }
        for name, (max_abs, rel_l2) in checks.items():
            if max_abs > atol or rel_l2 > rtol:
                failures.append(
                    f"axis={r.axis} round={r.round_idx} {name} "
                    f"max_abs={max_abs:.3e} rel_l2={rel_l2:.3e}"
                )
    if failures:
        raise AssertionError("\n".join(failures))


def _fmt_optional(value: float | None, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _print_round_result(result: AxisRoundResult) -> None:
    speedup = _speedup_ms(result.torch_npu_elapsed_ms, result.elapsed_ms)
    memory_delta = _memory_delta_mb(result.torch_npu_peak_memory_mb, result.peak_memory_mb)
    print(f"axis={result.axis} round={result.round_idx}")
    print(f"  torch_cpu: elapsed_ms={result.cpu_elapsed_ms:.2f}")
    print(
        "  torch_npu vs torch_cpu: "
        f"rfft_fwd_max=({result.torch_npu_fwd_r_max_abs:.3e},{result.torch_npu_fwd_i_max_abs:.3e}) "
        f"rfft_bwd_max={result.torch_npu_rfft_bwd_max_abs:.3e} "
        f"irfft_max=({result.torch_npu_irfft_fwd_max_abs:.3e},"
        f"{result.torch_npu_irfft_bwd_r_max_abs:.3e},"
        f"{result.torch_npu_irfft_bwd_i_max_abs:.3e}) "
        f"peak_mb={_fmt_optional(result.torch_npu_peak_memory_mb)} "
        f"elapsed_ms={result.torch_npu_elapsed_ms:.2f}"
    )
    print(
        "  fused_npu vs torch_cpu: "
        f"rfft_fwd_max=({result.fwd_r_max_abs:.3e},{result.fwd_i_max_abs:.3e}) "
        f"rfft_bwd_max={result.rfft_bwd_max_abs:.3e} "
        f"irfft_max=({result.irfft_fwd_max_abs:.3e},"
        f"{result.irfft_bwd_r_max_abs:.3e},"
        f"{result.irfft_bwd_i_max_abs:.3e}) "
        f"peak_mb={_fmt_optional(result.peak_memory_mb)} "
        f"elapsed_ms={result.elapsed_ms:.2f}"
    )
    print(
        "  fused_npu vs torch_npu: "
        f"rfft_fwd_max=({result.fused_vs_torch_npu_fwd_r_max_abs:.3e},"
        f"{result.fused_vs_torch_npu_fwd_i_max_abs:.3e}) "
        f"rfft_bwd_max={result.fused_vs_torch_npu_rfft_bwd_max_abs:.3e} "
        f"irfft_max=({result.fused_vs_torch_npu_irfft_fwd_max_abs:.3e},"
        f"{result.fused_vs_torch_npu_irfft_bwd_r_max_abs:.3e},"
        f"{result.fused_vs_torch_npu_irfft_bwd_i_max_abs:.3e}) "
        f"speedup={_fmt_optional(speedup, 'x')} "
        f"memory_delta_mb={_fmt_optional(memory_delta)}"
    )


def _bench_ms(fn, iters: int, device: torch.device) -> float:
    """Average wall time (ms) of ``fn`` over ``iters`` calls, device-synced."""
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / max(1, iters)


def run_kernel_benchmark(
    *,
    shape: tuple[int, int, int, int, int],
    axis: int,
    device: torch.device,
    iters: int,
    warmup: int,
    seed: int = 0,
) -> None:
    """Pure on-device kernel timing (no host<->device copies in the timed region).

    This isolates the fused rFFT/iRFFT kernel cost so the per-call number is
    directly comparable to the Ascend profiler's kernel Duration, unlike the
    end-to-end ``run_axis_round`` wall time which is dominated by ~GB .to(device)
    transfers and the CPU reference. Reports forward and forward+backward for the
    fused kernels and native ``torch.fft`` for reference, plus peak memory.
    """
    torch.manual_seed(seed)
    n = shape[axis]
    freq_shape = (*shape[:axis], n // 2 + 1, *shape[axis + 1:])
    x = torch.randn(shape, device=device, dtype=torch.float32)
    xr0 = torch.randn(freq_shape, device=device, dtype=torch.float32)
    xi0 = torch.randn(freq_shape, device=device, dtype=torch.float32)
    x_grad = x.detach().clone().requires_grad_(True)

    def fused_rfft_fwd():
        with torch.no_grad():
            real_rfft(x, dim=axis, norm="ortho")

    def torch_rfft_fwd():
        with torch.no_grad():
            torch.fft.rfft(x, dim=axis, norm="ortho")

    def fused_irfft_fwd():
        with torch.no_grad():
            real_irfft(xr0, xi0, n=n, dim=axis, norm="ortho")

    def torch_irfft_fwd():
        with torch.no_grad():
            torch.fft.irfft(torch.complex(xr0, xi0), n=n, dim=axis, norm="ortho")

    def fused_rfft_fwdbwd():
        xr, xi = real_rfft(x_grad, dim=axis, norm="ortho")
        torch.autograd.grad(xr.sum() + xi.sum(), x_grad)

    def torch_rfft_fwdbwd():
        spec = torch.fft.rfft(x_grad, dim=axis, norm="ortho")
        torch.autograd.grad(spec.real.sum() + spec.imag.sum(), x_grad)

    cases = [
        ("rfft_fwd    fused", fused_rfft_fwd),
        ("rfft_fwd    torch", torch_rfft_fwd),
        ("irfft_fwd   fused", fused_irfft_fwd),
        ("irfft_fwd   torch", torch_irfft_fwd),
        ("rfft_fwdbwd fused", fused_rfft_fwdbwd),
        ("rfft_fwdbwd torch", torch_rfft_fwdbwd),
    ]

    print(f"[kernel-bench] shape={shape} axis={axis} n={n} freq_len={n // 2 + 1} "
          f"block_m={os.environ.get('OIL_GAS_OPS_FFT_BLOCK_M', 'default(512)')} "
          f"warmup={warmup} iters={iters}")
    for label, fn in cases:
        for _ in range(warmup):
            fn()
        _reset_peak_memory(device)
        ms = _bench_ms(fn, iters, device)
        peak = _peak_memory_mb(device)
        print(f"  {label}: {ms:8.3f} ms/call  peak_mb={_fmt_optional(peak)}")


def _report_3d_precision(ref, x, ref_ft, out_re, out_im, sizes, *, atol: float, rtol: float) -> None:
    """Decomposed-vs-native precision of the 3D transform, plus the round trip."""
    grid = {"s": sizes, "dim": (-3, -2, -1), "norm": "ortho"}
    with torch.no_grad():
        tri_re, tri_im = ref.rfftn(x, *sizes)
        tri_inv = ref.irfftn(out_re, out_im, *sizes)
        nat_inv = torch.fft.irfftn(torch.complex(out_re, out_im), **grid)
        tri_round = ref.irfftn(tri_re, tri_im, *sizes)
        nat_round = torch.fft.irfftn(ref_ft, **grid)

    checks = (
        ("rfftn_fwd_r", tri_re, ref_ft.real),
        ("rfftn_fwd_i", tri_im, ref_ft.imag),
        ("irfftn_fwd", tri_inv, nat_inv),
        ("roundtrip", tri_round, x),
    )
    errs = {name: (_max_abs(a, b), _rel_l2(a, b)) for name, a, b in checks}
    fwd_r, fwd_i = errs["rfftn_fwd_r"], errs["rfftn_fwd_i"]
    inv, rnd = errs["irfftn_fwd"], errs["roundtrip"]
    print("  precision (decomposed vs native, ortho):")
    print(f"    rfftn_fwd    max_abs=({fwd_r[0]:.3e},{fwd_i[0]:.3e}) "
          f"rel_l2=({fwd_r[1]:.3e},{fwd_i[1]:.3e})")
    print(f"    irfftn_fwd   max_abs={inv[0]:.3e} rel_l2={inv[1]:.3e}")
    print(f"    roundtrip    decomposed_vs_x max_abs={rnd[0]:.3e} rel_l2={rnd[1]:.3e} "
          f"| native_vs_x max_abs={_max_abs(nat_round, x):.3e}")

    failures = [
        f"{name} max_abs={mx:.3e} rel_l2={rl:.3e}"
        for name, (mx, rl) in errs.items()
        if mx > atol or rl > rtol
    ]
    if failures:
        raise AssertionError("bench-3d precision exceeded thresholds:\n" + "\n".join(failures))


def run_3d_bench(
    *,
    shape: tuple[int, int, int, int, int],
    device: torch.device,
    iters: int,
    warmup: int,
    atol: float,
    rtol: float,
    seed: int = 0,
) -> None:
    """Full-3D decomposed (Triton per-axis) vs native multi-axis ``torch.fft``.

    Mirrors exactly what ``SpectralConv3dReal`` does in the model: its ``_rfftn``
    /``_irfftn`` decompose a 3D rfftn/irfftn into per-axis Triton real FFTs,
    whereas native does the whole transform in a single
    ``torch.fft.rfftn(x, dim=(-3,-2,-1))`` / ``torch.fft.irfftn(out_ft,
    s=(D,H,W), dim=(-3,-2,-1))`` call. ``shape`` is interpreted channels-first
    ``(B, C, D, H, W)`` (the last three dims are the transformed grid), so the
    smoke tensor ``(1, 35, 128, 128, 128)`` maps directly. Both paths use
    ``norm="ortho"`` (the norm the decomposition is built on) for an
    element-wise-fair precision comparison; the round-trip reconstruction error
    is norm-independent. Reports latency, peak memory and precision.

    The spectral-conv module under test lives in the downstream model repository;
    opt in with ``OIL_GAS_OPS_MODEL_REPO`` and point ``OIL_GAS_OPS_SPECTRAL_CONV``
    at its import path (``<module>:<class>``).
    """
    ref = _import_spectral_conv()

    torch.manual_seed(seed)
    size_d, size_h, size_w = int(shape[-3]), int(shape[-2]), int(shape[-1])
    x = torch.randn(shape, device=device, dtype=torch.float32)
    x_grad = x.detach().clone().requires_grad_(True)

    def tri_rfftn_fwd():
        with torch.no_grad():
            ref.rfftn(x, size_d, size_h, size_w)

    def torch_rfftn_fwd():
        with torch.no_grad():
            torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")

    # Common frequency-domain input for the inverse comparison (native forward,
    # ortho), split into the real/imag pair the decomposition consumes.
    with torch.no_grad():
        ref_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
    out_re = ref_ft.real.contiguous()
    out_im = ref_ft.imag.contiguous()
    out_re_grad = out_re.detach().clone().requires_grad_(True)
    out_im_grad = out_im.detach().clone().requires_grad_(True)

    def tri_irfftn_fwd():
        with torch.no_grad():
            ref.irfftn(out_re, out_im, size_d, size_h, size_w)

    def torch_irfftn_fwd():
        with torch.no_grad():
            torch.fft.irfftn(
                torch.complex(out_re, out_im), s=(size_d, size_h, size_w), dim=(-3, -2, -1), norm="ortho"
            )

    def tri_rfftn_fwdbwd():
        re, im = ref.rfftn(x_grad, size_d, size_h, size_w)
        torch.autograd.grad(re.sum() + im.sum(), x_grad)

    def torch_rfftn_fwdbwd():
        ft = torch.fft.rfftn(x_grad, dim=(-3, -2, -1), norm="ortho")
        torch.autograd.grad(ft.real.sum() + ft.imag.sum(), x_grad)

    def tri_irfftn_fwdbwd():
        y = ref.irfftn(out_re_grad, out_im_grad, size_d, size_h, size_w)
        torch.autograd.grad(y.sum(), (out_re_grad, out_im_grad))

    def torch_irfftn_fwdbwd():
        y = torch.fft.irfftn(
            torch.complex(out_re_grad, out_im_grad), s=(size_d, size_h, size_w), dim=(-3, -2, -1), norm="ortho"
        )
        torch.autograd.grad(y.sum(), (out_re_grad, out_im_grad))

    cases = [
        ("rfftn_fwd     decomposed(triton)", tri_rfftn_fwd),
        ("rfftn_fwd     native(torch.fft) ", torch_rfftn_fwd),
        ("irfftn_fwd    decomposed(triton)", tri_irfftn_fwd),
        ("irfftn_fwd    native(torch.fft) ", torch_irfftn_fwd),
        ("rfftn_fwdbwd  decomposed(triton)", tri_rfftn_fwdbwd),
        ("rfftn_fwdbwd  native(torch.fft) ", torch_rfftn_fwdbwd),
        ("irfftn_fwdbwd decomposed(triton)", tri_irfftn_fwdbwd),
        ("irfftn_fwdbwd native(torch.fft) ", torch_irfftn_fwdbwd),
    ]

    print(
        f"[bench-3d] shape={shape} (B,C,D,H,W) grid=({size_d},{size_h},{size_w}) "
        f"has_real_fft={ref.has_real_fft} norm=ortho "
        f"block_m={os.environ.get('OIL_GAS_OPS_FFT_BLOCK_M', 'default(512)')} "
        f"warmup={warmup} iters={iters}"
    )
    timings: dict[str, float] = {}
    for label, fn in cases:
        for _ in range(warmup):
            fn()
        _reset_peak_memory(device)
        ms = _bench_ms(fn, iters, device)
        peak = _peak_memory_mb(device)
        timings[label.strip()] = ms
        print(f"  {label}: {ms:9.3f} ms/call  peak_mb={_fmt_optional(peak)}")

    def _pair_speedup(kind: str) -> None:
        tri = timings.get(f"{kind} decomposed(triton)")
        nat = timings.get(f"{kind} native(torch.fft)")
        if tri and nat:
            print(f"  -> {kind:13s} decomposed/native = {tri / nat:.2f}x (native/decomposed = {nat / tri:.2f}x)")

    for kind in ("rfftn_fwd", "irfftn_fwd", "rfftn_fwdbwd", "irfftn_fwdbwd"):
        _pair_speedup(kind)

    _report_3d_precision(ref, x, ref_ft, out_re, out_im,
                         (size_d, size_h, size_w), atol=atol, rtol=rtol)


def _add_shape_args(parser: argparse.ArgumentParser) -> None:
    """Explicit / derived input shape."""
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--shape", type=int, nargs=5, metavar=("B", "D", "H", "W", "C"), default=None)
    parser.add_argument("--axes", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--from-model-config",
        type=Path,
        default=None,
        help=(
            "Infer the spectral latent shape from a downstream model YAML config. "
            f"Defaults to ${MODEL_CONFIG_ENV} when set."
        ),
    )


def _add_shortcut_args(parser: argparse.ArgumentParser) -> None:
    """Preset shapes for the two shapes we profile most."""
    parser.add_argument(
        "--run-latent-3d-profiled",
        action="store_true",
        help=(
            "Shortcut for the exact profiled 3D spectral latent tensor "
            f"{LATENT_3D_PROFILED_SHAPE} on axis {LATENT_3D_PROFILED_AXES[0]} "
            "(the shape of the slow rFFT/iRFFT kernels)."
        ),
    )
    parser.add_argument(
        "--run-spectral-3d-smoke",
        action="store_true",
        help=(
            "Shortcut for the exact 128^3 smoke spectral tensor "
            f"{SPECTRAL_3D_SMOKE_SHAPE} (channels-first B,C,D,H,W) on axes "
            f"{list(SPECTRAL_3D_SMOKE_AXES)} -> reproduces the three per-axis rFFT "
            "kernels SpectralConv3dReal launches (W=last-axis 2D kernel, H/D "
            "transpose-free axis kernels), all n=128."
        ),
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Tiling override, repetition count, device, tolerances and output."""
    parser.add_argument(
        "--block-m",
        type=int,
        default=None,
        help="Override OIL_GAS_OPS_FFT_BLOCK_M (DFT-matmul row tile) for this run; sweep e.g. 128/256/512.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device-id", type=int, default=0,
                        help="NPU device id used by native torch-npu and fused Triton tests.")
    parser.add_argument("--atol", type=float, default=3e-3)
    parser.add_argument("--rtol", type=float, default=3e-3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--print-shape-only", action="store_true")


def _add_bench_args(parser: argparse.ArgumentParser) -> None:
    """The two benchmark-only modes."""
    parser.add_argument(
        "--bench-only",
        action="store_true",
        help="Pure on-device kernel timing (no host<->device copies, no CPU reference).",
    )
    parser.add_argument(
        "--bench-3d",
        action="store_true",
        help=(
            "Full-3D benchmark: SpectralConv3dReal decomposed per-axis rfftn/irfftn "
            "(Triton) vs native multi-axis torch.fft.rfftn/irfftn(dim=(-3,-2,-1)). "
            "shape is read channels-first (B,C,D,H,W); pair with --run-spectral-3d-smoke."
        ),
    )
    parser.add_argument("--bench-iters", type=int, default=20, help="Timed iterations per case in --bench-only mode.")
    parser.add_argument("--bench-warmup", type=int, default=3,
                        help="Warmup iterations (compile + caches) in --bench-only mode.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Triton-Ascend real FFT on 3D channels-last inputs.")
    _add_shape_args(parser)
    _add_shortcut_args(parser)
    _add_run_args(parser)
    _add_bench_args(parser)
    return parser


def _apply_shortcuts(args) -> None:
    """Resolve --block-m / $OIL_GAS_OPS_MODEL_CONFIG and the two shape shortcuts."""
    if args.block_m is not None:
        os.environ["OIL_GAS_OPS_FFT_BLOCK_M"] = str(args.block_m)
    if args.from_model_config is None and os.environ.get(MODEL_CONFIG_ENV):
        args.from_model_config = Path(os.environ[MODEL_CONFIG_ENV])
    for enabled, shape, axes in (
        (args.run_latent_3d_profiled, LATENT_3D_PROFILED_SHAPE, LATENT_3D_PROFILED_AXES),
        (args.run_spectral_3d_smoke, SPECTRAL_3D_SMOKE_SHAPE, SPECTRAL_3D_SMOKE_AXES),
    ):
        if not enabled:
            continue
        args.shape = list(shape)
        if args.axes == [1, 2, 3]:
            args.axes = list(axes)


def _resolve_shape(args):
    """``(shape, inferred_metadata)``; may also fill in ``args.axes``."""
    if args.from_model_config is not None:
        config_path = resolve_model_config(args.from_model_config, option="--from-model-config")
        shape, inferred_axes, metadata = infer_fft_shape_from_model_config(config_path)
        if args.axes == [1, 2, 3]:
            args.axes = inferred_axes
        return shape, metadata
    if args.shape is not None:
        return tuple(args.shape), None
    return _shape(args.batch, args.size, args.channels), None


def _run_axis_rounds(args, shape, device, inferred_metadata) -> None:
    results: list[AxisRoundResult] = []
    for round_idx in range(args.rounds):
        for axis in args.axes:
            result = run_axis_round(
                axis=axis,
                round_idx=round_idx,
                shape=shape,
                device=device,
                seed=args.seed + round_idx * 100 + axis,
            )
            results.append(result)
            _print_round_result(result)

    _assert_thresholds(results, atol=args.atol, rtol=args.rtol)
    if args.output is None:
        return
    payload = {
        "shape": shape,
        "rounds": args.rounds,
        "axes": args.axes,
        "atol": args.atol,
        "rtol": args.rtol,
        "inferred_metadata": inferred_metadata,
        "results": [asdict(r) for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    _apply_shortcuts(args)
    shape, inferred_metadata = _resolve_shape(args)

    block_m = os.environ.get("OIL_GAS_OPS_FFT_BLOCK_M", "default(512)")
    print(f"shape={shape} axes={args.axes} block_m={block_m}")
    if inferred_metadata is not None:
        print(f"inferred={json.dumps(inferred_metadata, ensure_ascii=False)}")
    if args.print_shape_only:
        return

    device = _maybe_npu_device(args.device_id)
    if args.bench_3d:
        run_3d_bench(shape=shape, device=device, iters=args.bench_iters,
                     warmup=args.bench_warmup, atol=args.atol, rtol=args.rtol, seed=args.seed)
    elif args.bench_only:
        for axis in args.axes:
            run_kernel_benchmark(shape=shape, axis=axis, device=device, iters=args.bench_iters,
                                 warmup=args.bench_warmup, seed=args.seed + axis)
    else:
        _run_axis_rounds(args, shape, device, inferred_metadata)


if __name__ == "__main__":
    main()
