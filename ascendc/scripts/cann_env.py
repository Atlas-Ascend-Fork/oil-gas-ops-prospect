# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CANN 环境探测与在子 shell 中执行命令。"""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
import tempfile
from pathlib import Path


def resolve_cann_root(install_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if install_path:
        candidates.append(Path(install_path))
    if env := os.environ.get("ASCEND_HOME_PATH"):
        candidates.append(Path(env))
    candidates.extend([
        Path.home() / "Ascend" / "ascend-toolkit" / "latest",
        Path("/usr/local/Ascend/cann-9.0.0"),
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
        Path("/usr/local/Ascend/cann"),
    ])
    for root in candidates:
        if _has_set_env_script(root):
            return root
    raise FileNotFoundError(
        "找不到 CANN 安装目录。请 source set_env.sh 或使用 -i/--install-path 指定"
    )


def _has_set_env_script(root: Path) -> bool:
    return any(
        (root / name).is_file()
        for name in ("set_env.sh", "bin/setenv.bash", "cann/set_env.sh")
    )


def find_set_env_script(root: Path) -> Path:
    for name in ("set_env.sh", "bin/setenv.bash"):
        p = root / name
        if p.is_file():
            return p
    p = root / "cann" / "set_env.sh"
    if p.is_file():
        return p
    raise FileNotFoundError(f"在 {root} 下找不到 set_env.sh / setenv.bash")


def _child_env(extra_env: dict[str, str] | None) -> dict[str, str]:
    """把额外变量并进当前环境，通过 env= 传给子进程而不是拼进脚本文本。

    这样调用方给的值永远不会被 shell 重新解析，杜绝命令注入；变量名限制为合法
    标识符，避免 `FOO=1; rm -rf /` 之类的键被当成 shell 语句。
    """
    env = dict(os.environ)
    for key, value in (extra_env or {}).items():
        if not key.isidentifier():
            raise ValueError(f"非法环境变量名: {key!r}")
        env[key] = str(value)
    return env


def run_in_cann_shell(
    command: str,
    *,
    cwd: Path | None = None,
    install_path: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    """在 source 过 CANN set_env.sh 的环境里执行 command。

    CANN 只以**可 source 的脚本**（set_env.sh）形式提供环境，而 msopgen / CustomOp
    的 build.sh 依赖它导出的变量，所以这里需要一层 shell。做法是把 source + 命令写成
    一个带 shebang 的私有脚本再直接执行它，而不是把脚本正文当参数传给 ``bash -c``：

    * ``shell=False`` + argv 只有脚本自身路径，父进程侧不做任何二次解析，也不把
      命令正文塞进命令行；
    * 脚本落在 ``mkdtemp`` 建出的 0700 私有目录里（无 /tmp 抢占竞争），退出即删；
    * 唯一插进脚本正文的路径经过 ``shlex.quote``；
    * 调用方变量走 ``env=`` 传递，不拼进脚本正文（见 :func:`_child_env`）。

    ``command`` 本身是本仓构造的固定构建命令（msopgen / cmake / ctest），不接收
    外部输入。
    """
    root = resolve_cann_root(install_path)
    set_env = find_set_env_script(root)
    print(f"ASCEND_HOME_PATH={root}")
    script_text = (
        "#!/bin/bash\n"
        "set -e\n"
        f"source {shlex.quote(str(set_env))}\n"
        f"{command}\n"
    )
    with tempfile.TemporaryDirectory(prefix="oil_gas_ops_cann_") as tmp_dir:
        script = Path(tmp_dir) / "run_in_cann_env.sh"
        script.write_text(script_text, encoding="utf-8")
        script.chmod(stat.S_IRWXU)  # 0700，仅当前用户可读写执行
        subprocess.run(
            [str(script)],
            cwd=cwd,
            check=True,
            shell=False,
            env=_child_env(extra_env),
        )
