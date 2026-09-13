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

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# 未装 whl 时用仓内源码（见 scripts/local_pkg.py / tests/conftest.py）。
REPO_ROOT = Path(__file__).resolve().parents[3]
for _extra in (REPO_ROOT / "tests" / "triton",):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from _model_repo import MODEL_REPO_ENV, find_model_repo_root  # noqa: E402
from oil_gas_ops_prospect.real_fft import make_basis, real_irfft, real_rfft  # noqa: E402

# The downstream model repository, when this operator repo is checked out next to one.
# Only the two cross-check cases below need it; the rest of this module is
# standalone. Keep it off the module-level import path so a standalone checkout
# can still collect and run the CPU coverage.
MODEL_REPO_ROOT = find_model_repo_root()
if MODEL_REPO_ROOT is not None and str(MODEL_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_REPO_ROOT))

MIN_FFT_TOL = 1e-5

# Downstream backbone used by the optional fused-FFT cross-check, as
# "<module>:<class>". Keeps the downstream module layout out of this repository.
FFT_BACKBONE_ENV = "OIL_GAS_OPS_FFT_BACKBONE"
DEFAULT_FFT_BACKBONE = "src.models.backbones.fdpot:SlimHybridAFNOxD"

# Shapes taken from a profiled 3D spectral workload: the latent grid is
# (B, 41, 41, 41, embed) and the fused rFFT/iRFFT runs along a length-41 spectral
# axis (n=41 -> freq_len=21). The heavy kernels in the profiler used
# (2, 41, 41, 41, 2048) -> rows=6_885_376. The light variants below keep the
# n=41/freq_len=21 tiling (padded to BLOCK_K=48 / BLOCK_N=32) while staying cheap
# enough for the CPU-fallback and quick NPU correctness checks.
LATENT_3D_LIGHT_SHAPES = [
    ((1, 41, 41, 41, 2), 1),
    ((1, 41, 41, 41, 2), 2),
    ((1, 41, 41, 41, 2), 3),
]
LATENT_3D_FULL_SHAPE = (2, 41, 41, 41, 2048)


def _maybe_npu_device():
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        return None
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    return None


def _require_model_repo(what: str):
    """Skip instead of failing when the downstream model repository is absent."""

    if MODEL_REPO_ROOT is None:
        pytest.skip(
            f"{what} lives in the downstream model repository, which was not found next to this "
            f"operator repository; set {MODEL_REPO_ENV}=/path/to/model-repo to enable"
        )
    return MODEL_REPO_ROOT


@pytest.mark.parametrize(
    "shape,dim",
    [((2, 8, 3), 1), ((2, 5, 7, 4), 2), ((1, 4, 6, 8, 3), 3)] + LATENT_3D_LIGHT_SHAPES,
)
def test_real_fft_cpu_fallback_matches_torch_fft(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)

    xr, xi = real_rfft(x, dim=dim, norm="ortho")
    ref = torch.fft.rfft(x, dim=dim, norm="ortho")
    assert torch.allclose(xr, ref.real, rtol=MIN_FFT_TOL, atol=MIN_FFT_TOL)
    assert torch.allclose(xi, ref.imag, rtol=MIN_FFT_TOL, atol=MIN_FFT_TOL)

    y = real_irfft(xr, xi, n=x.shape[dim], dim=dim, norm="ortho")
    ref_y = torch.fft.irfft(ref, n=x.shape[dim], dim=dim, norm="ortho")
    assert torch.allclose(y, ref_y, rtol=MIN_FFT_TOL, atol=MIN_FFT_TOL)


@pytest.mark.parametrize("n", [7, 8, 16, 41, 192, 320])
def test_internal_fourier_basis_matches_torch_fft(n):
    torch.manual_seed(n)
    x = torch.randn((3, n), dtype=torch.float32)

    rfft_cos, rfft_sin, irfft_cos, irfft_sin = make_basis(n, torch.device("cpu"), "ortho")
    xr = x @ rfft_cos
    xi = x @ rfft_sin
    ref = torch.fft.rfft(x, dim=-1, norm="ortho")
    assert torch.allclose(xr, ref.real, rtol=MIN_FFT_TOL, atol=MIN_FFT_TOL)
    assert torch.allclose(xi, ref.imag, rtol=MIN_FFT_TOL, atol=MIN_FFT_TOL)

    y = xr @ irfft_cos.T - xi @ irfft_sin.T
    ref_y = torch.fft.irfft(ref, n=n, dim=-1, norm="ortho")
    assert torch.allclose(y, ref_y, rtol=MIN_FFT_TOL, atol=MIN_FFT_TOL)
    # n≥192 is where unfolded float32 2πkt/n sets a ~3e-5 floor; folding must beat that.
    if n >= 192:
        assert max((xr - ref.real).abs().max().item(), (xi - ref.imag).abs().max().item()) < 3e-6


def test_model_fused_fft_config_switch_matches_reference_path():
    """Cross-check a downstream backbone's fused-FFT switch against its reference path.

    Opt-in: the backbone lives in the downstream model repository. Enable with
    ``OIL_GAS_OPS_MODEL_REPO``, and override the import path with
    ``OIL_GAS_OPS_FFT_BACKBONE=<module>:<class>`` if it differs.
    """
    _require_model_repo("the downstream fused-FFT backbone")

    spec = os.environ.get(FFT_BACKBONE_ENV, DEFAULT_FFT_BACKBONE)
    if ":" not in spec:
        pytest.skip(f"{FFT_BACKBONE_ENV}={spec!r} must be '<module>:<class>'")
    module_name, class_name = spec.rsplit(":", 1)

    module = pytest.importorskip(
        module_name,
        reason=f"{module_name} is not importable from the model repository",
    )
    backbone_cls = getattr(module, class_name, None)
    if backbone_cls is None:
        pytest.skip(f"{module.__file__} does not export {class_name}")

    torch.manual_seed(3)
    device = _maybe_npu_device() or torch.device("cpu")
    fused = backbone_cls(
        width=16,
        num_blocks=4,
        spatial_dims=3,
        channel_first=False,
        modes_1d=4,
        rank=2,
        use_both_axes=False,
        use_fused_fft=True,
    )
    reference = backbone_cls(
        width=16,
        num_blocks=4,
        spatial_dims=3,
        channel_first=False,
        modes_1d=4,
        rank=2,
        use_both_axes=False,
        use_fused_fft=False,
    )
    reference.load_state_dict(fused.state_dict())
    fused = fused.to(device)
    reference = reference.to(device)

    assert fused.use_fused_fft is True
    assert fused.shared_head.use_fused_fft is True
    assert reference.use_fused_fft is False
    assert reference.shared_head.use_fused_fft is False

    x = torch.randn((1, 4, 5, 6, 16), dtype=torch.float32, device=device)
    y_fused = fused(x, axis_mode="W")
    y_ref = reference(x, axis_mode="W")
    assert torch.allclose(y_fused.cpu(), y_ref.cpu(), rtol=MIN_FFT_TOL, atol=MIN_FFT_TOL)


@pytest.mark.parametrize(
    "shape,dim",
    [((2, 8, 3), 1), ((1, 4, 6, 8, 3), 3)] + LATENT_3D_LIGHT_SHAPES,
)
def test_real_fft_npu_triton_matches_cpu_torch_fft(shape, dim):
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(1)
    x_cpu = torch.randn(shape, dtype=torch.float32)
    x_npu = x_cpu.to(device)

    xr, xi = real_rfft(x_npu, dim=dim, norm="ortho")
    ref = torch.fft.rfft(x_cpu, dim=dim, norm="ortho")
    assert torch.allclose(xr.cpu(), ref.real, rtol=2e-4, atol=2e-4)
    assert torch.allclose(xi.cpu(), ref.imag, rtol=2e-4, atol=2e-4)

    y = real_irfft(xr, xi, n=shape[dim], dim=dim, norm="ortho")
    ref_y = torch.fft.irfft(ref, n=shape[dim], dim=dim, norm="ortho")
    assert torch.allclose(y.cpu(), ref_y, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize(
    "shape,dim",
    [((2, 8, 3), 1), ((1, 4, 6, 8, 3), 3)] + LATENT_3D_LIGHT_SHAPES,
)
def test_real_fft_npu_triton_backward_matches_cpu_torch_fft(shape, dim):
    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(2)
    x_cpu = torch.randn(shape, dtype=torch.float32)
    freq_shape = (*shape[:dim], shape[dim] // 2 + 1, *shape[dim + 1:])
    grad_r_cpu = torch.randn(freq_shape, dtype=torch.float32)
    grad_i_cpu = torch.randn(freq_shape, dtype=torch.float32)

    x_ref = x_cpu.detach().clone().requires_grad_(True)
    ref = torch.fft.rfft(x_ref, dim=dim, norm="ortho")
    (ref.real * grad_r_cpu + ref.imag * grad_i_cpu).sum().backward()

    x_npu = x_cpu.to(device).requires_grad_(True)
    xr, xi = real_rfft(x_npu, dim=dim, norm="ortho")
    (xr * grad_r_cpu.to(device) + xi * grad_i_cpu.to(device)).sum().backward()
    assert torch.allclose(x_npu.grad.cpu(), x_ref.grad, rtol=2e-4, atol=2e-4)

    spec_r_cpu = torch.randn(freq_shape, dtype=torch.float32)
    spec_i_cpu = torch.randn(freq_shape, dtype=torch.float32)
    grad_y_cpu = torch.randn(shape, dtype=torch.float32)

    spec_r_ref = spec_r_cpu.detach().clone().requires_grad_(True)
    spec_i_ref = spec_i_cpu.detach().clone().requires_grad_(True)
    ref_y = torch.fft.irfft(torch.complex(spec_r_ref, spec_i_ref), n=shape[dim], dim=dim, norm="ortho")
    (ref_y * grad_y_cpu).sum().backward()

    spec_r_npu = spec_r_cpu.to(device).requires_grad_(True)
    spec_i_npu = spec_i_cpu.to(device).requires_grad_(True)
    y = real_irfft(spec_r_npu, spec_i_npu, n=shape[dim], dim=dim, norm="ortho")
    (y * grad_y_cpu.to(device)).sum().backward()
    assert torch.allclose(spec_r_npu.grad.cpu(), spec_r_ref.grad, rtol=2e-4, atol=2e-4)
    assert torch.allclose(spec_i_npu.grad.cpu(), spec_i_ref.grad, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_real_fft_npu_latent_3d_full_shape(dim, capsys):
    """Exact profiled fwi-3d-model-128 shape (2, 41, 41, 41, 2048), rows=6_885_376.

    Validates the optimized DFT-matmul tiling (n=41 -> BLOCK_K=48,
    freq_len=21 -> BLOCK_N=32, big BLOCK_M) on the real workload. Gated behind an
    env flag because it allocates ~1.1GB fp32 tensors on the NPU. Correctness is
    checked two ways to keep host memory bounded: an ortho rFFT/iRFFT round-trip
    at full scale, and an absolute comparison against torch.fft on a channel slice.
    """
    if os.environ.get("OIL_GAS_OPS_RUN_LATENT3D_FFT_TEST") != "1":
        pytest.skip("set OIL_GAS_OPS_RUN_LATENT3D_FFT_TEST=1 to run the heavy (2,41,41,41,2048) NPU FFT test")

    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    torch.manual_seed(41)
    shape = LATENT_3D_FULL_SHAPE
    n = shape[dim]
    x_npu = torch.randn(shape, dtype=torch.float32, device=device)

    xr, xi = real_rfft(x_npu, dim=dim, norm="ortho")
    assert xr.shape[dim] == n // 2 + 1

    # (1) full-scale ortho round-trip: irfft(rfft(x)) == x
    x_round = real_irfft(xr, xi, n=n, dim=dim, norm="ortho")
    assert torch.allclose(x_round.cpu(), x_npu.cpu(), rtol=2e-4, atol=2e-4)

    # (2) absolute check against torch.fft on a bounded channel slice.
    sl = [slice(None)] * x_npu.ndim
    sl[-1] = slice(0, 4)
    x_slice = x_npu[tuple(sl)].contiguous()
    ref = torch.fft.rfft(x_slice.cpu(), dim=dim, norm="ortho")
    xr_s, xi_s = real_rfft(x_slice, dim=dim, norm="ortho")
    assert torch.allclose(xr_s.cpu(), ref.real, rtol=2e-4, atol=2e-4)
    assert torch.allclose(xi_s.cpu(), ref.imag, rtol=2e-4, atol=2e-4)


def test_real_fft_npu_128_3d_backward_and_memory_opt_in(capsys):
    if os.environ.get("OIL_GAS_OPS_RUN_128_FFT_TEST") != "1":
        pytest.skip("set OIL_GAS_OPS_RUN_128_FFT_TEST=1 to run the heavy 128^3 NPU FFT test")
    from verify_triton_fft_real_128_3d import _assert_thresholds, _print_round_result, run_axis_round

    device = _maybe_npu_device()
    if device is None:
        pytest.skip("NPU is not available")

    shape = (1, 128, 128, 128, 1)
    results = []
    with capsys.disabled():
        for round_idx in range(2):
            for axis in (1, 2, 3):
                result = run_axis_round(
                    axis=axis,
                    round_idx=round_idx,
                    shape=shape,
                    device=device,
                    seed=2026 + round_idx * 10 + axis,
                )
                results.append(result)
                _print_round_result(result)
    _assert_thresholds(results, atol=3e-3, rtol=3e-3)
    assert all(r.peak_memory_mb is not None and r.peak_memory_mb > 0 for r in results)


def test_model_config_fft_shape_inference():
    """Shape inference from a downstream model YAML.

    Opt-in: the config belongs to the downstream project, so point at one with
    ``OIL_GAS_OPS_MODEL_CONFIG=/abs/path/to/config.yaml``. Assertions are
    structural (the exact dimensions are config-specific).
    """
    from verify_triton_fft_real_128_3d import MODEL_CONFIG_ENV, infer_fft_shape_from_model_config

    raw = os.environ.get(MODEL_CONFIG_ENV)
    if not raw:
        pytest.skip(f"set {MODEL_CONFIG_ENV}=/abs/path/to/config.yaml to check shape inference")
    config_path = Path(raw).expanduser()
    if not config_path.is_file():
        pytest.skip(f"{MODEL_CONFIG_ENV}={raw} is not a file")

    shape, axes, metadata = infer_fft_shape_from_model_config(config_path)

    assert len(shape) == 5 and all(isinstance(d, int) and d > 0 for d in shape)
    assert axes and all(1 <= a <= 3 for a in axes)
    for key in ("raw_size", "padded_size", "latent_size", "embed_dim"):
        assert key in metadata, f"missing inferred metadata key: {key}"
    # latent grid must be the spatial part of the inferred tensor
    assert tuple(shape[1:4]) == tuple(metadata["latent_size"])
    assert shape[4] == metadata["embed_dim"]
