#!/usr/bin/env python3
"""Run the shared train/evaluate CLI with explicit LiteWeather v2 support."""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_perception.liteweather_runtime import activate


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "evaluate"}:
        raise SystemExit("Usage: liteweather_entry.py {train|evaluate} [CLI options]")
    script = ROOT / "scripts/restoration" / f"{sys.argv[1]}.py"
    sys.argv = [str(script), *sys.argv[2:]]
    activate()
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
