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

"""TDD tests for Round 4 review fixes.

P1-1: npu_only() must check NPU hardware availability, not just import success.
P1-2: CPU precision tests must skip when no compiled kernel (self-proving).

Each test pair demonstrates RED→GREEN for the corresponding fix.
"""
import unittest.mock as mock
import torch


# --- P1-1: npu_only() hardware check ---

def test_npu_only_uses_hardware_check():
    """npu_only() should skip when torch_npu imports but no NPU hardware."""
    import sys
    conftest_mod = sys.modules.get('conftest')
    if conftest_mod is None:
        import importlib
        conftest_mod = importlib.import_module('conftest')

    with mock.patch.object(conftest_mod, 'HAS_TORCH_NPU', True):
        with mock.patch.object(torch.npu, 'is_available', return_value=False):
            mark = conftest_mod.npu_only()
            skip_condition = mark.args[0] if mark.args else mark.kwargs.get('condition', None)
            assert skip_condition is True, (
                "npu_only() should skip when torch_npu imports but no NPU hardware. "
                f"Got condition={skip_condition}"
            )


def test_npu_hardware_only_consistent_with_npu_only():
    """npu_only() and npu_hardware_only() should have same skip behavior."""
    import sys
    conftest_mod = sys.modules.get('conftest')
    if conftest_mod is None:
        import importlib
        conftest_mod = importlib.import_module('conftest')

    with mock.patch.object(conftest_mod, 'HAS_TORCH_NPU', True):
        with mock.patch.object(torch.npu, 'is_available', return_value=False):
            mark_npu = conftest_mod.npu_only()
            mark_hw = conftest_mod.npu_hardware_only()
            cond_npu = mark_npu.args[0] if mark_npu.args else None
            cond_hw = mark_hw.args[0] if mark_hw.args else None
            assert cond_npu == cond_hw, (
                f"npu_only() condition ({cond_npu}) != npu_hardware_only() condition ({cond_hw}). "
                "Both should check hardware availability."
            )


# --- P1-2: requires_compiled_kernel marker ---

def test_has_compiled_kernel_helper_exists():
    """conftest must expose _has_compiled_kernel() for kernel availability check."""
    import sys
    conftest_mod = sys.modules.get('conftest')
    if conftest_mod is None:
        import importlib
        conftest_mod = importlib.import_module('conftest')
    assert hasattr(conftest_mod, '_has_compiled_kernel'), (
        "conftest must define _has_compiled_kernel() — P1-2 requires detecting "
        "whether the compiled NPU extension is available to avoid self-proving tests."
    )


def test_requires_compiled_kernel_marker_skips_without_kernel():
    """requires_compiled_kernel() should skip when no compiled NPU kernel."""
    import sys
    conftest_mod = sys.modules.get('conftest')
    if conftest_mod is None:
        import importlib
        conftest_mod = importlib.import_module('conftest')
    assert hasattr(conftest_mod, 'requires_compiled_kernel'), (
        "conftest must define requires_compiled_kernel() marker."
    )
    with mock.patch.object(conftest_mod, '_has_compiled_kernel', return_value=False):
        mark = conftest_mod.requires_compiled_kernel()
        skip_condition = mark.args[0] if mark.args else None
        assert skip_condition is True, (
            "requires_compiled_kernel() should skip when _has_compiled_kernel()=False. "
            f"Got condition={skip_condition}"
        )
