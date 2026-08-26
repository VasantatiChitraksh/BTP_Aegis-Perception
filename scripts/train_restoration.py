#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.config import load_config
from aegis_perception.training import train_restoration


def main() -> None:
    parser = argparse.ArgumentParser(description="Train attention/vanilla Pix2Pix restoration")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train_restoration(load_config(args.config))


if __name__ == "__main__":
    main()
