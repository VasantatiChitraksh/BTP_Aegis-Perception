from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record(config: dict[str, Any] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        record.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
        )
    except ImportError:
        record["torch"] = None
    if config is not None:
        record["config"] = config
    return record


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_path)
