#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from aegis_perception.checkpoints import load_generator
from aegis_perception.config import load_config
from aegis_perception.data.paired import normalized_tensor_to_pil, pil_to_normalized_tensor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize restored images for detector evaluation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--control-output-root",
        type=Path,
        help="Also write a bicubic down/up-sampled control for fair detector comparison",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == input_root or output_root.is_relative_to(input_root):
        raise SystemExit("Output root must be outside the input tree")
    if args.control_output_root:
        control_root = args.control_output_root.resolve()
        if control_root == input_root or control_root.is_relative_to(input_root):
            raise SystemExit("Control output root must be outside the input tree")
        if control_root == output_root or control_root.is_relative_to(output_root):
            raise SystemExit("Control and restored output trees must not overlap")
        if output_root.is_relative_to(control_root):
            raise SystemExit("Control and restored output trees must not overlap")
    config = load_config(args.config)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    generator, _ = load_generator(args.checkpoint, device=device)
    image_size = tuple(config["data"]["image_size"])
    paths = [
        path
        for path in sorted(args.input_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise SystemExit(f"No images found below {args.input_root}")
    with torch.inference_mode():
        for path in paths:
            relative = path.relative_to(args.input_root)
            output = args.output_root / relative.with_suffix(".png")
            output.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as handle:
                original = handle.convert("RGB")
            original_size = original.size
            model_input = original.resize(
                (image_size[1], image_size[0]), Image.Resampling.BICUBIC
            )
            tensor = pil_to_normalized_tensor(model_input).unsqueeze(0).to(device)
            restored = generator(tensor)[0]
            restored_image = normalized_tensor_to_pil(restored).resize(
                original_size, Image.Resampling.BICUBIC
            )
            restored_image.save(output, format="PNG")
            if args.control_output_root:
                control_output = args.control_output_root / relative.with_suffix(".png")
                control_output.parent.mkdir(parents=True, exist_ok=True)
                model_input.resize(original_size, Image.Resampling.BICUBIC).save(
                    control_output, format="PNG"
                )
    print(f"restored {len(paths)} images to {args.output_root}")


if __name__ == "__main__":
    main()
