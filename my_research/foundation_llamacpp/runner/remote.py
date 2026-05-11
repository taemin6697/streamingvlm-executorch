from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def run(cmd: list[str], *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture_output)


def adb_cmd(serial: str | None) -> list[str]:
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    return cmd


def push(adb: list[str], local: Path, remote_dir: str) -> None:
    run(adb + ["push", str(local), f"{remote_dir}/{local.name}"])


def remote_exists(adb: list[str], remote_path: str) -> bool:
    result = subprocess.run(
        adb + ["shell", f"test -f {shlex.quote(remote_path)}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def pull_if_exists(adb: list[str], remote: str, local: Path) -> None:
    result = subprocess.run(adb + ["shell", f"test -f {shlex.quote(remote)}"], check=False)
    if result.returncode == 0:
        local.parent.mkdir(parents=True, exist_ok=True)
        run(adb + ["pull", remote, str(local)])


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)

