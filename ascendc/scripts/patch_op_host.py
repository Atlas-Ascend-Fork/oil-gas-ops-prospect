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

"""为 msopgen 生成的 op_host/CMakeLists.txt 补充额外 include 目录。

部分移植算子（如 conv3d）的 host tiling 库依赖 CANN 的 op_common 头（例如
`log/log.h`），而 msopgen 默认仅向 opbuild 与 host 库传入 `-I${CANN}/include`，
缺少 `aarch64-linux/pkg_inc/op_common`。此 patch 在不改动 msopgen 模板的前提下，
向 opbuild（npu_op_code_gen 的 COMPILE_OPTIONS）与 host 库（include_directories）
注入这些目录。对不需要它们的算子也无害。
"""

from __future__ import annotations

from pathlib import Path

from scripts.config import OPERATORS_DIR

_EXTRA_CMAKE_MARKER = "# [oil_gas_ops_prospect] per-op extra.cmake appended"

# 相对 ${ASCEND_CANN_PACKAGE_PATH} 的额外 include 目录。
# 默认留空：移植算子（如 conv3d）的 host tiling 库已通过本地 shim 头消除对
# op_common/log（dlog）等的依赖；如后续算子需要额外 CANN 头，可在此追加。
_EXTRA_INCLUDE_SUBDIRS: tuple[str, ...] = ()

_MARKER = "# [oil_gas_ops_prospect] extra include dirs"


def _extra_include_cmake_paths() -> list[str]:
    return [f"${{ASCEND_CANN_PACKAGE_PATH}}/{sub}" for sub in _EXTRA_INCLUDE_SUBDIRS]


def patch_op_host_includes(custom_op: Path) -> None:
    cmake_path = custom_op / "op_host" / "CMakeLists.txt"
    if not cmake_path.is_file():
        raise FileNotFoundError(f"op_host/CMakeLists.txt 不存在: {cmake_path}")

    if not _EXTRA_INCLUDE_SUBDIRS:
        return  # 无额外 include 需求

    text = cmake_path.read_text(encoding="utf-8")
    if _MARKER in text:
        return  # 幂等

    inc_paths = _extra_include_cmake_paths()
    inc_block_lines = "\n".join(f"    {p}" for p in inc_paths)
    inject = (
        f"\n{_MARKER}\n"
        f"set(_OILGAS_EXTRA_INC\n{inc_block_lines}\n)\n"
        f"include_directories(${{_OILGAS_EXTRA_INC}})\n"
    )

    # 1) 在第一个 aux_source_directory(...) 之后插入 include_directories（影响 host 库目标）。
    anchor = "aux_source_directory("
    idx = text.find(anchor)
    if idx != -1:
        line_end = text.find(")", idx)
        line_end = text.find("\n", line_end) + 1
        text = text[:line_end] + inject + text[line_end:]
    else:
        text = inject + text

    # 2) 给 npu_op_code_gen 增加 COMPILE_OPTIONS（opbuild 自动生成步骤的编译命令）。
    compile_opts = " ".join(f"-I{p}" for p in inc_paths)
    needle = "npu_op_code_gen("
    pos = text.find(needle)
    if pos != -1:
        close = text.find(")", pos)
        if close != -1 and "COMPILE_OPTIONS" not in text[pos:close]:
            insertion = f"    COMPILE_OPTIONS {compile_opts}\n"
            text = text[:close] + insertion + text[close:]

    cmake_path.write_text(text, encoding="utf-8")
    print(f"[patch] op_host include dirs 注入完成: {cmake_path}")


def append_op_host_extra_cmake(custom_op: Path, selected_ops: list[str]) -> None:
    """将算子自带的 op_host/extra.cmake 追加到生成的 op_host/CMakeLists.txt 末尾。

    用于需要原位编译仓库 host tiling（如 conv3d 反向算子）并补充 include/链接的场景。
    extra.cmake 在所有 host 库目标（cust_optiling/cust_op_proto/cust_opapi）定义之后执行，
    因此可安全使用 target_sources/target_link_libraries。
    """
    cmake_path = custom_op / "op_host" / "CMakeLists.txt"
    if not cmake_path.is_file():
        raise FileNotFoundError(f"op_host/CMakeLists.txt 不存在: {cmake_path}")
    text = cmake_path.read_text(encoding="utf-8")

    appended = []
    for op in selected_ops:
        extra = OPERATORS_DIR / op / "op_host" / "extra.cmake"
        if not extra.is_file():
            continue
        block_marker = f"{_EXTRA_CMAKE_MARKER}: {op}"
        if block_marker in text:
            continue
        text += f"\n{block_marker}\n" + extra.read_text(encoding="utf-8") + "\n"
        appended.append(op)

    if appended:
        cmake_path.write_text(text, encoding="utf-8")
        print(f"[patch] op_host extra.cmake 已追加: {', '.join(appended)}")
