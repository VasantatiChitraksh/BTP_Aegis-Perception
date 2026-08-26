#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.detection import evaluate_yolo
from aegis_perception.reproducibility import environment_record, write_json


def view(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("views must be CONDITION=DATASET_YAML")
    condition, path = value.split("=", 1)
    if not condition or not path:
        raise argparse.ArgumentTypeError("views must be CONDITION=DATASET_YAML")
    return condition, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clean/degraded/restored YOLO views")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--view", type=view, action="append", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/detection/matrix"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.view) < 2:
        raise SystemExit("Provide at least two condition views")
    names = [name for name, _ in args.view]
    if len(names) != len(set(names)):
        raise SystemExit("Condition names must be unique")

    results = {}
    for condition, data in args.view:
        results[condition] = evaluate_yolo(
            model_path=args.model,
            data=data,
            image_size=args.imgsz,
            batch_size=args.batch,
            device=args.device,
            split=args.split,
            project=args.output_dir / "ultralytics",
            name=condition,
        )
    payload = {
        "environment": environment_record(),
        "model": str(args.model.resolve()),
        "views": {name: str(path.resolve()) for name, path in args.view},
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "split": args.split,
        "results": results,
    }
    output = args.output_dir / "matrix.json"
    write_json(output, payload)
    print(results)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
