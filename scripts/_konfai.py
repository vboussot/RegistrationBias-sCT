"""Shared helpers for small, transparent KonfAI command wrappers."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def run(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command))
    if dry_run:
        return
    if shutil.which(command[0]) is None:
        raise SystemExit("The `konfai` executable is missing. Install requirements first.")
    env = os.environ.copy()
    source = str(ROOT / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def device_args(gpu: list[int] | None, cpu: int | None) -> list[str]:
    if gpu and cpu:
        raise SystemExit("Choose either --gpu or --cpu, not both.")
    if gpu:
        return ["--gpu", *[str(item) for item in gpu]]
    return ["--cpu", str(cpu or 1)]
