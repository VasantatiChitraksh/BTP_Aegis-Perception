#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.weather import degrade

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sample_seed(base_seed: int, relative_path: Path) -> int:
    digest = hashlib.sha256(f"{base_seed}:{relative_path.as_posix()}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible adverse-weather images")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", choices=("fog", "rain", "snow", "smoke"), required=True)
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend", choices=("procedural", "diffusion"), default="procedural"
    )
    parser.add_argument("--diffusion-model")
    parser.add_argument("--diffusion-revision")
    parser.add_argument("--diffusion-device", default="cuda")
    parser.add_argument("--diffusion-strength", type=float, default=0.25)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--diffusion-guidance", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [
        path
        for path in sorted(args.input_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise SystemExit(f"No images found below {args.input_root}")
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == input_root or output_root.is_relative_to(input_root):
        raise SystemExit("Output root must be outside the input tree")

    diffusion = None
    if args.backend == "diffusion":
        if not args.diffusion_model:
            raise SystemExit("--diffusion-model is required for the diffusion backend")
        from aegis_perception.weather.diffusion import (
            DiffusionSettings,
            DiffusionWeatherGenerator,
        )

        diffusion = DiffusionWeatherGenerator(
            DiffusionSettings(
                model_id=args.diffusion_model,
                revision=args.diffusion_revision,
                device=args.diffusion_device,
                strength=args.diffusion_strength,
                guidance_scale=args.diffusion_guidance,
                num_inference_steps=args.diffusion_steps,
            )
        )

    records = []
    for input_path in paths:
        relative = input_path.relative_to(args.input_root)
        output_path = args.output_root / relative.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seed = sample_seed(args.seed, relative)
        with Image.open(input_path) as handle:
            image = handle.convert("RGB")
        if diffusion is None:
            generated = degrade(image, args.condition, severity=args.severity, seed=seed)
        else:
            generated = diffusion.generate(image, condition=args.condition, seed=seed)
        generated.save(output_path, format="PNG")
        records.append(
            {
                "input": str(input_path.resolve()),
                "output": str(output_path.resolve()),
                "condition": args.condition,
                "severity": args.severity if diffusion is None else None,
                "seed": seed,
                "backend": args.backend,
                "model": args.diffusion_model,
                "model_revision": args.diffusion_revision,
                "diffusion_strength": args.diffusion_strength if diffusion else None,
                "diffusion_steps": args.diffusion_steps if diffusion else None,
                "diffusion_guidance": args.diffusion_guidance if diffusion else None,
            }
        )

    manifest_path = args.output_root.parent / f"{args.output_root.name}_generation.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"generated {len(records)} images; manifest={manifest_path}")


if __name__ == "__main__":
    main()
