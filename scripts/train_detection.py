#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an Ultralytics detector")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", type=Path, default=Path("artifacts/detection/training"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install the detection extra: pip install -e '.[detection]'") from exc

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
        plots=True,
    )


if __name__ == "__main__":
    main()
