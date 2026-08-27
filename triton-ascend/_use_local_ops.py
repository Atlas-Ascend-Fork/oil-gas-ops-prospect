# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Import-time selection of the in-repo Triton sources.

``_local_ops_package.use_local_triton_sources()`` has to run *before* anything
imports ``oil_gas_ops_prospect.triton.*``, otherwise the installed wheel wins.
Calling it as a bare statement in between the imports of a test/verify script
puts a non-import statement inside the import block, which the coding standard
forbids (imports must be one contiguous block after the module docstring).

Importing this module performs the selection as an import side effect, so the
callers keep a single import block::

    import _use_local_ops  # noqa: F401  (side effect: prefer in-repo sources)
    from oil_gas_ops_prospect.triton.real_fft import real_rfft

Set ``OIL_GAS_OPS_TRITON_SOURCE=whl`` to test the installed package instead;
then this module is a no-op and ``USING_LOCAL_SOURCES`` is False.
"""

from __future__ import annotations

from _local_ops_package import use_local_triton_sources

USING_LOCAL_SOURCES = use_local_triton_sources()
