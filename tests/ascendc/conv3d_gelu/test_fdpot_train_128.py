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
"""在 128×256×256 的 FDPOT3D 整网上做一步训练，对比融合 Conv3d+GELU。

空间尺寸与 mid40 网格一致：(nz, ny, nx) = (128, 256, 256)，pml=16。
网络是 FDPOT3DModel（encoder Conv3d+GELU、DPOT blocks、out_layer Conv3d+GELU）。
深度取 4，避免 30 层 latent 激活把 64GB 卡撑满，仍然走完整的前向、反传和 Adam 更新。
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ACLNN_CACHE_LIMIT", "0")

import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AI4FWI = os.environ.get("AI4FWI_ROOT", "/root/work/workspace/mcg/ai4fwi-main")
sys.path.insert(0, os.path.join(REPO, "ascendc"))
sys.path.insert(0, os.path.join(REPO, "ascendc", "pybind"))
sys.path.insert(0, AI4FWI)
sys.path.insert(0, os.path.join(AI4FWI, "train"))

import torch_npu  # noqa: E402,F401
from conv3d_gelu import conv3d_gelu  # noqa: E402
from src.models.model import FDPOT3DModel  # noqa: E402


def _is_gelu(module: nn.Module) -> bool:
    name = module.__class__.__name__.lower()
    return name in ("gelu", "customgelu")


def _is_conv3d(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.Tensor) and weight.ndim == 5 and hasattr(module, "kernel_size")


def _kernel_tuple(module: nn.Module) -> tuple[int, int, int]:
    k = module.kernel_size
    if isinstance(k, int):
        return (k, k, k)
    return tuple(int(v) for v in k)


def _fuse_sequential(seq: nn.Sequential, fused: bool) -> nn.Sequential:
    items = list(seq._modules.items())
    out = []
    i = 0
    while i < len(items):
        key, mod = items[i]
        nxt = items[i + 1][1] if i + 1 < len(items) else None
        if (
            _is_conv3d(mod)
            and nxt is not None
            and _is_gelu(nxt)
            and int(getattr(mod, "groups", 1)) == 1
            and all(k in (1, 3) for k in _kernel_tuple(mod))
        ):
            out.append((key, _Conv3dThenGelu(mod, nxt, fused=fused)))
            i += 2
            continue
        if isinstance(mod, nn.Sequential):
            out.append((key, _fuse_sequential(mod, fused)))
        else:
            out.append((key, mod))
        i += 1
    rebuilt = nn.Sequential()
    for key, mod in out:
        rebuilt.add_module(key, mod)
    return rebuilt


class _Conv3dThenGelu(nn.Module):
    def __init__(self, conv: nn.Module, act: nn.Module, fused: bool):
        super().__init__()
        self.conv = conv
        self.act = act
        self.fused = fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused and x.dim() == 5 and x.dtype == torch.float32:
            return conv3d_gelu(
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
                padding_mode=self.conv.padding_mode,
            )
        import torch.nn.functional as F
        return F.gelu(self.conv(x), approximate="tanh")


def _replace(module: nn.Module, fused: bool) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Sequential):
            setattr(module, name, _fuse_sequential(child, fused))
            _replace(getattr(module, name), fused)
        else:
            _replace(child, fused)


def _build(fused: bool, device: torch.device) -> FDPOT3DModel:
    config = {
        "pml_size": 16,
        "patch_size": 8,
        "in_channels": 4,
        "out_channels": 2,
        "in_timesteps": 1,
        "out_timesteps": 1,
        "embed_dim": 256,
        "out_layer_dim": 64,
        "n_blocks": 4,
        "depth": 4,
        "modes": 16,
        "mlp_ratio": 2.0,
        "normalize": False,
        "act": "gelu",
        "time_agg": "exp_mlp",
        "n_cls": 1,
        "slim_rank": 4,
        "slim_num_bands": 2,
        "slim_use_lora": False,
        "slim_use_both_axes": False,
        "spatial_dims": 3,
        "encoder_channels": 32,
        "zero_init_last_conv": False,
    }
    model = FDPOT3DModel(config, resolutions=(128, 256, 256))
    _replace(model, fused=fused)
    return model.to(device)


def _step(model: nn.Module, inputs: torch.Tensor):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.npu.empty_cache()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    opt.zero_grad(set_to_none=True)
    before = torch.npu.memory_allocated()
    pred, _ = model(inputs)
    torch.npu.synchronize()
    held = (torch.npu.memory_allocated() - before) / (1024 ** 2)
    peak_fwd = torch.npu.max_memory_allocated() / (1024 ** 2)
    loss = (pred - pred.detach().mean()).float().square().mean()
    loss.backward()
    opt.step()
    torch.npu.synchronize()
    peak_step = torch.npu.max_memory_allocated() / (1024 ** 2)
    return pred.detach(), float(loss.detach().item()), held, peak_fwd, peak_step


def main() -> int:
    os.environ.setdefault("OIL_GAS_OPS_CONV3D_GELU_IMPL", "ascendc")
    os.environ.setdefault("OIL_GAS_OPS_REQUIRE_ASCENDC_CONV3D_GELU", "1")
    if not torch.npu.is_available():
        print("NPU 不可用")
        return 1
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch_npu.npu.conv.allow_hf32 = False
    torch.manual_seed(0)
    inputs = torch.randn(1, 2, 128, 256, 256, device=device, dtype=torch.float32)

    def _watch(model):
        box = {}

        def hook(_m, _i, out):
            box["y"] = out.detach()

        handle = model.encoder_c.register_forward_hook(hook)
        return box, handle

    torch.manual_seed(1)
    base = _build(False, device)
    n_fused = sum(1 for m in base.modules() if isinstance(m, _Conv3dThenGelu))
    print(f"fused_pairs={n_fused}", flush=True)
    state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    base_box, base_handle = _watch(base)
    base_pred, base_loss, base_held, base_pf, base_ps = _step(base, inputs)
    base_handle.remove()
    base_enc = base_box["y"].float().cpu()
    del base, base_box
    torch.npu.empty_cache()
    torch.npu.synchronize()

    fused = _build(True, device)
    fused.load_state_dict(state)
    fused_box, fused_handle = _watch(fused)
    fused_pred, fused_loss, fused_held, fused_pf, fused_ps = _step(fused, inputs)
    fused_handle.remove()
    enc_diff = (fused_box["y"].float().cpu() - base_enc).abs()
    enc_rel = enc_diff.max().item() / max(base_enc.abs().max().item(), 1e-8)
    held_saved = base_held - fused_held
    peak_saved = base_ps - fused_ps
    print(f"baseline loss={base_loss:.6e} held={base_held:.1f} peak_fwd={base_pf:.1f} peak_step={base_ps:.1f} MB")
    print(f"fused    loss={fused_loss:.6e} held={fused_held:.1f} peak_fwd={fused_pf:.1f} peak_step={fused_ps:.1f} MB")
    print(f"encoder_rel={enc_rel:.3e} held_saved={held_saved:.1f} peak_saved={peak_saved:.1f} MB")
    ok = (
        n_fused > 0
        and enc_rel < 1e-5
        and held_saved > 512.0
        and fused_pf < base_pf
        and fused_ps < base_ps
    )
    print(f"network_ok={ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
