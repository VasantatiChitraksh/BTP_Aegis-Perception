#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a configured restoration model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Resume an epoch-boundary checkpoint")
    parser.add_argument(
        "--stop-after-epochs", type=int, help="Stop early without changing the schedule"
    )
    args = parser.parse_args()
    from aegis_perception.training import train_restoration

    train_restoration(
        load_config(args.config), resume=args.resume, stop_after_epochs=args.stop_after_epochs
    )


if __name__ == "__main__":
    main()
