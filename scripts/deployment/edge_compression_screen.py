#!/usr/bin/env python3
"""Screen Restormer pruning on four existing validation fixtures, without retraining.

This is a rejection screen, not a deployment accuracy benchmark. No checkpoint
is replaced, no test data is used, and CPU timings are not Jetson measurements.
"""

import argparse
import csv
import importlib.util
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn

from aegis_perception.checkpoints import load_generator
from aegis_perception.edge.common import digest, environment, model_info, write_json
from aegis_perception.edge.quality import quality


def prune_two_four(model):
    """Keep two largest magnitudes per four input channels, at each kernel position.

    Conv weights are K,C,R,S. Depthwise/grouped convolutions and input-channel
    counts not divisible by four remain dense. Eligibility does not prove that
    TensorRT will select sparse tactics on a particular device.
    """
    names = []
    with torch.no_grad():
        for name, layer in model.named_modules():
            if not isinstance(layer, nn.Conv2d) or layer.groups != 1:
                continue
            k, c, r, s = layer.weight.shape
            if c % 4:
                continue
            blocks = (
                layer.weight.permute(0, 2, 3, 1).contiguous().clone().reshape(k, r, s, c // 4, 4)
            )
            indices = blocks.abs().argsort(dim=-1, stable=True)[..., :2]
            blocks.scatter_(-1, indices, 0)
            layer.weight.copy_(blocks.reshape(k, r, s, c).permute(0, 3, 1, 2))
            names.append(name)
    return names


def tensor_stats(model):
    weights = [m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    total = sum(w.numel() for w in weights)
    zeros = sum(int((w == 0).sum()) for w in weights)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return {
        "convolution_weights": total,
        "zero_convolution_weights": zeros,
        "weighted_convolution_sparsity": zeros / total,
        "dense_state_dict_bytes": buffer.tell(),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def write_table(path, report):
    rows = []
    for variant in report["variants"]:
        for sample in variant["rows"]:
            rows.append(
                {
                    "variant": variant["name"],
                    **sample,
                    "dense_state_dict_bytes": variant["statistics"]["dense_state_dict_bytes"],
                    "weighted_convolution_sparsity": variant["statistics"][
                        "weighted_convolution_sparsity"
                    ],
                    "scope": "four-image validation screen; NOT full test or Jetson benchmark",
                }
            )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/edge_v1/compression_screen"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--resume", action="store_true", help="Resume completed variants of a screen"
    )
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        parser.error("Output already exists; choose a new directory to preserve results")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    checkpoint, _ = model_info("restormer")
    model, _ = load_generator(checkpoint, device="cpu")
    original = {key: value.clone() for key, value in model.state_dict().items()}
    # Execute exactly the Linear-only conversion used by the existing script.
    dynamic = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    dynamic_audit = {
        "linear_modules_before": sum(isinstance(m, nn.Linear) for m in model.modules()),
        "quantized_linear_modules_after": sum(
            isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in dynamic.modules()
        ),
        "all_state_tensors_identical": list(original) == list(dynamic.state_dict())
        and all(torch.equal(value, dynamic.state_dict()[key]) for key, value in original.items()),
        "interpretation": "Zero converted layers is a no-op, not a successful INT8 experiment",
    }
    del dynamic
    legacy_path = ROOT / "quantization/scripts/prune_model.py"
    spec = importlib.util.spec_from_file_location("legacy_pruning", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    fixture_dir = ROOT / "artifacts/edge_v1/exports/256x448/restormer"
    with np.load(fixture_dir / "fixtures.npz", allow_pickle=False) as archive:
        fixtures = dict(archive)
    import json

    provenance = json.loads((fixture_dir / "fixture_provenance.json").read_text())
    if provenance["sha256"] != digest(fixture_dir / "fixtures.npz") or any(
        row["split"] != "val" for row in provenance["samples"]
    ):
        raise ValueError("Invalid validation fixture provenance")
    report = {
        "scope": "Four validation images, one per weather; rejection screen ONLY",
        "geometry": [1, 3, 256, 448],
        "quality_space": "Clipped float RGB [0,1], resized clean targets, padding excluded",
        "checkpoint_sha256": digest(checkpoint),
        "pruning_source_sha256": digest(legacy_path),
        "script_sha256": digest(__file__),
        "fixture_provenance": provenance,
        "dynamic_int8_audit": dynamic_audit,
        "environment": environment(),
        "screen_gate": {
            "minimum_psnr_delta_db_each_image": -0.1,
            "minimum_ssim_delta_each_image": -0.001,
        },
        "jetson_speed_memory_power": "NOT MEASURED",
        "variants": [],
    }
    baseline = {}
    completed = set()
    if args.resume:
        previous = json.loads((args.output / "results.json").read_text())
        for key in (
            "checkpoint_sha256",
            "pruning_source_sha256",
            "fixture_provenance",
            "screen_gate",
        ):
            if previous[key] != report[key]:
                raise ValueError(f"Resume contract changed: {key}")
        if previous.get("completed"):
            raise ValueError("Screen already completed")
        report["variants"] = previous["variants"]
        report["prior_script_sha256"] = previous["script_sha256"]
        completed = {v["name"] for v in report["variants"]}
        for variant in report["variants"]:
            if variant["name"] == "dense_fp32":
                baseline = {row["weather"]: row for row in variant["rows"]}
    write_json(args.output / "results.json", report)
    for variant in ("dense_fp32", "global_10", "global_30", "global_50", "two_four"):
        if variant in completed:
            continue
        model.load_state_dict(original, strict=True)
        selected_layers = []
        if variant.startswith("global_"):
            legacy.apply_global_pruning(model, int(variant.split("_")[1]) / 100)
            legacy.make_pruning_permanent(model)
        elif variant == "two_four":
            selected_layers = prune_two_four(model)
        result = {
            "name": variant,
            "statistics": tensor_stats(model),
            "script_sha256": report["script_sha256"],
            "rows": [],
        }
        if selected_layers:
            result["two_four_layers"] = selected_layers
        for weather in ("fog", "rain", "smoke", "snow"):
            x = fixtures["input_" + weather]
            with torch.inference_mode():
                output = model(torch.from_numpy(x)).numpy()
            if not np.isfinite(output).all():
                raise ValueError(f"Nonfinite output: {variant}/{weather}")
            h, w = fixtures["valid_hw_" + weather]

            def rgb(a, h=h, w=w):
                return ((np.clip(a[0, :, :h, :w], -1, 1) + 1) / 2).transpose(1, 2, 0)

            scores = quality(rgb(fixtures["target_" + weather]), rgb(output))
            if variant == "dense_fp32":
                np.testing.assert_allclose(
                    output, fixtures["expected_" + weather], atol=5e-4, rtol=1e-3
                )
                baseline[weather] = scores
            row = {"weather": weather, **scores}
            row["psnr_delta_db"] = scores["psnr_db"] - baseline[weather]["psnr_db"]
            row["ssim_delta"] = scores["ssim"] - baseline[weather]["ssim"]
            row["screen_passed"] = row["psnr_delta_db"] >= -0.1 and row["ssim_delta"] >= -0.001
            result["rows"].append(row)
            print(
                f"{variant}/{weather}: PSNR={scores['psnr_db']:.3f} "
                f"SSIM={scores['ssim']:.5f} delta={row['psnr_delta_db']:+.3f}dB",
                flush=True,
            )
        result["screen_passed"] = all(row["screen_passed"] for row in result["rows"])
        result["mean_psnr_delta_db"] = float(np.mean([r["psnr_delta_db"] for r in result["rows"]]))
        result["mean_ssim_delta"] = float(np.mean([r["ssim_delta"] for r in result["rows"]]))
        report["variants"].append(result)
        write_json(args.output / "results.json", report)
    if digest(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("Source checkpoint changed")
    report["completed"] = True
    write_json(args.output / "results.json", report)
    write_table(args.output / "results.csv", report)
    print(f"Completed: {args.output / 'results.json'}; original checkpoint unchanged", flush=True)


if __name__ == "__main__":
    main()
