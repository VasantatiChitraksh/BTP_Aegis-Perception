#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.detection import evaluate_yolo
from aegis_perception.reproducibility import environment_record, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one YOLO dataset view")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/detection/evaluation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_yolo(
        model_path=args.model,
        data=args.data,
        image_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        split=args.split,
        project=args.output_dir,
        name=args.condition,
    )
    payload = {
        "environment": environment_record(),
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "condition": args.condition,
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "split": args.split,
        "metrics": metrics,
    }
    output = args.output_dir / f"{args.condition}.json"
    write_json(output, payload)
    print(metrics)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
