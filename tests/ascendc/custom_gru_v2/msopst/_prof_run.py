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

"""Helper script for msprof profiling of CustomGRUV2."""
import argparse
import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
torch.npu.config.allow_internal_format = False


def run_single():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    x = torch.randn(1, 1000, 3, device="npu:0")
    wi = torch.randn(72, 3, device="npu:0") * 0.1
    wh = torch.randn(72, 24, device="npu:0") * 0.1
    bi = torch.randn(72, device="npu:0") * 0.1
    bh = torch.randn(72, device="npu:0") * 0.1
    for _ in range(5):
        y, h = custom_gru_v2(x, wi, wh, bi, bh, 24, False)
    torch.npu.synchronize()
    for _ in range(20):
        y, h = custom_gru_v2(x, wi, wh, bi, bh, 24, False)
    torch.npu.synchronize()
    print(f"single: y.shape={y.shape}")


def run_batch():
    from oil_gas_ops_prospect.custom_gru_v2 import custom_gru_v2
    x = torch.randn(32, 100, 3, device="npu:0")
    wi = torch.randn(72, 3, device="npu:0") * 0.1
    wh = torch.randn(72, 24, device="npu:0") * 0.1
    bi = torch.randn(72, device="npu:0") * 0.1
    bh = torch.randn(72, device="npu:0") * 0.1
    for _ in range(5):
        y, h = custom_gru_v2(x, wi, wh, bi, bh, 24, False)
    torch.npu.synchronize()
    for _ in range(20):
        y, h = custom_gru_v2(x, wi, wh, bi, bh, 24, False)
    torch.npu.synchronize()
    print(f"batch: y.shape={y.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["single", "batch"], required=True)
    args = p.parse_args()
    if args.case == "single":
        run_single()
    else:
        run_batch()
