#!/usr/bin/env python3
"""prune_model.py — Apply magnitude-based unstructured pruning to any Aegis generator.

Usage
-----
python quantization/scripts/prune_model.py \\
    --checkpoint checkpoints/liteweatherformer_rain.pt \\
    --sparsity 0.5 \\
    --method global \\
    --image data/raw/rain/realrain1k/rain/0001.jpg \\
    --output-dir artifacts/quantization/pruning

Description
-----------
1. Loads the generator from checkpoint.
2. Records FP32 baseline inference latency and output (used as PSNR reference).
3. Applies L1-unstructured pruning (global or local) across all Conv2d and
   ConvTranspose2d weight tensors.
4. Logs per-layer sparsity to a table.
5. Runs inference on the pruned model, measures PSNR degradation and latency.
6. Makes pruning permanent (prune.remove), saves the pruned model .pt file.
7. Writes a JSON report.

Notes
-----
* For GPU inference, CUDA synchronise is called before stopping each timer to
  avoid measuring asynchronous GPU-side work on the CPU clock.
* Unstructured pruning zeroes individual weights; it does NOT reduce the number
  of operations on GPU (dense kernels execute on all weights including zeros).
  Real speed-up from unstructured pruning requires sparse-capable runtimes
  (e.g., NVIDIA Ampere 2:4 sparsity, or TorchSparse on CPU).
* The PSNR reference is the FP32 un-pruned output on the provided image.
  Compare against clean GT with scripts/restoration/evaluate.py.
* Point --image to data/raw/rain/realrain1k/rain/*.jpg once datasets are ready.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# ---------------------------------------------------------------------------
# Repo root → sys.path  (quantization/scripts/ → parents[2] = repo root)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from aegis_perception.checkpoints import load_generator  # noqa: E402
from aegis_perception.official_runtime import activate  # noqa: E402
from aegis_perception.data.paired import pil_to_normalized_tensor  # noqa: E402
from aegis_perception.metrics import psnr  # noqa: E402
from PIL import Image  # noqa: E402


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sync_cuda(device: str) -> None:
    """Synchronise CUDA stream before reading the wall-clock timer."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _to_np_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, C, H, W) float tensor → (H, W, C) float32 numpy in [0, 1]."""
    return tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).float().cpu().numpy()


def _model_name_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    """Extract the model ``name`` from the checkpoint config dict."""
    return checkpoint["config"]["model"]["name"].lower()


def _time_inference(
    model: nn.Module,
    inp: torch.Tensor,
    device: str,
    n_warmup: int = 10,
    n_timed: int = 50,
) -> tuple[float, torch.Tensor]:
    """Measure mean inference latency in milliseconds.

    Returns
    -------
    latency_ms:
        Mean over n_timed runs (excluding warm-up).
    last_output:
        The model output from the final timed run.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            out = model(inp)
            _sync_cuda(device)

        times: list[float] = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            out = model(inp)
            _sync_cuda(device)
            times.append((time.perf_counter() - t0) * 1_000)

    return float(np.mean(times)), out.detach()


def _collect_conv_layers(
    model: nn.Module,
) -> list[tuple[str, nn.Module]]:
    """Return (name, module) pairs for all Conv2d and ConvTranspose2d layers."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            layers.append((name, module))
    return layers


def _layer_sparsity(module: nn.Module) -> float:
    """Compute the fraction of zero weights in a module's weight tensor."""
    w = module.weight
    if hasattr(module, "weight_mask"):
        # Reparameterised by prune API — count zeros in effective weight
        w = w  # effective weight already = weight_orig * weight_mask
    total = w.numel()
    zeros = float((w == 0).sum().item())
    return zeros / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Pruning logic
# ---------------------------------------------------------------------------

def apply_global_pruning(model: nn.Module, sparsity: float) -> None:
    """Apply global L1-unstructured pruning across all Conv2d / ConvTranspose2d.

    The pruning threshold is computed jointly across all selected parameters,
    so insensitive (low-norm) layers receive more sparsity.
    """
    parameters_to_prune = [
        (module, "weight")
        for _, module in _collect_conv_layers(model)
    ]
    if not parameters_to_prune:
        raise RuntimeError("No Conv2d / ConvTranspose2d layers found in model.")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )


def apply_local_pruning(model: nn.Module, sparsity: float) -> None:
    """Apply local L1-unstructured pruning: each layer independently to ``sparsity``."""
    for _, module in _collect_conv_layers(model):
        prune.l1_unstructured(module, name="weight", amount=sparsity)


def make_pruning_permanent(model: nn.Module) -> None:
    """Call prune.remove() on all pruned layers to bake the mask into weights."""
    for _, module in _collect_conv_layers(model):
        try:
            prune.remove(module, "weight")
        except ValueError:
            # Layer was not pruned (e.g., skipped); ignore.
            pass


# ---------------------------------------------------------------------------
# Per-layer sparsity table
# ---------------------------------------------------------------------------

def _print_sparsity_table(
    layer_stats: list[dict[str, Any]],
    model_name: str,
    method: str,
) -> None:
    """Print a formatted per-layer sparsity table to stdout."""
    col = [50, 14, 12, 10]
    sep = "+" + "+".join("-" * w for w in col) + "+"

    print(f"\n{'='*90}")
    print(f"  Per-Layer Sparsity — {model_name.upper()} ({method} pruning)")
    print(f"{'='*90}")
    print(sep)
    headers = ["Layer", "Shape", "Params", "Sparsity"]
    header_row = "|" + "|".join(
        f" {h:<{col[i]-2}} " for i, h in enumerate(headers)
    ) + "|"
    print(header_row)
    print(sep)

    for stat in sorted(layer_stats, key=lambda s: s["sparsity"], reverse=True):
        row = "|" + "|".join([
            f" {stat['name'][:col[0]-2]:<{col[0]-2}} ",
            f" {stat['shape']:<{col[1]-2}} ",
            f" {stat['params']:>{col[2]-2},} ",
            f" {stat['sparsity']:>{col[3]-3}.1%} ",
        ]) + "|"
        print(row)
    print(sep)
    print()


def _collect_layer_stats(model: nn.Module) -> list[dict[str, Any]]:
    """Gather per-layer sparsity statistics after pruning."""
    stats = []
    for name, module in _collect_conv_layers(model):
        w = module.weight
        stats.append({
            "name": name,
            "shape": str(tuple(w.shape)),
            "params": w.numel(),
            "sparsity": _layer_sparsity(module),
            "l1_norm_mean": float(w.detach().abs().mean().item()),
        })
    return stats


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------

def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data atomically (tmp file then rename)."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply magnitude-based pruning to an Aegis Perception generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to .pt checkpoint (written by save_checkpoint).",
    )
    parser.add_argument(
        "--sparsity", required=True, type=float,
        help="Target sparsity fraction [0, 1).  E.g. 0.5 zeroes 50%% of weights.",
    )
    parser.add_argument(
        "--method", choices=["global", "local"], default="global",
        help="global: joint threshold across all layers; local: per-layer threshold.",
    )
    parser.add_argument(
        "--image", required=True, type=Path,
        help=(
            "Degraded input image for PSNR measurement.  "
            "Point to data/raw/rain/realrain1k/rain/*.jpg once datasets are ready."
        ),
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to save pruned model .pt and JSON report.",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU (Raspberry Pi 5 simulation).",
    )
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-timed", type=int, default=50)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full pruning experiment: baseline → prune → measure → save."""
    args = parse_args()

    if not (0.0 <= args.sparsity < 1.0):
        raise ValueError(f"--sparsity must be in [0, 1).  Got {args.sparsity}")

    # --- Device ---
    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
        print("[INFO] Running on CPU.")
    else:
        device = "cuda"
        print(f"[INFO] Running on CUDA: {torch.cuda.get_device_name(0)}")

    # --- Load checkpoint ---
    activate()
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    generator, checkpoint = load_generator(str(args.checkpoint), device=device)
    model_name = _model_name_from_checkpoint(checkpoint)
    print(f"[INFO] Model: {model_name} | Sparsity target: {args.sparsity:.0%} | Method: {args.method}")

    # --- Load image ---
    pil_img = Image.open(args.image).convert("RGB")
    inp: torch.Tensor = pil_to_normalized_tensor(pil_img).unsqueeze(0).to(device)

    # --- FP32 baseline inference ---
    print("[INFO] Measuring FP32 baseline ...")
    baseline_latency_ms, baseline_out = _time_inference(
        generator, inp, device, args.n_warmup, args.n_timed
    )
    baseline_np = _to_np_hwc(baseline_out)
    # PSNR of the un-pruned output vs itself is ∞; we report it as None
    # (the pruned PSNR will be vs this reference).
    print(f"[INFO] Baseline latency: {baseline_latency_ms:.2f} ms")

    # --- Apply pruning ---
    print(f"[INFO] Applying {args.method} L1-unstructured pruning @ sparsity={args.sparsity:.0%} ...")
    if args.method == "global":
        apply_global_pruning(generator, args.sparsity)
    else:
        apply_local_pruning(generator, args.sparsity)

    # --- Collect per-layer stats (before making permanent, masks still explicit) ---
    layer_stats = _collect_layer_stats(generator)
    actual_sparsity = float(np.mean([s["sparsity"] for s in layer_stats]))

    # --- Print per-layer table ---
    _print_sparsity_table(layer_stats, model_name, args.method)

    # --- Pruned inference ---
    print("[INFO] Measuring pruned model inference ...")
    pruned_latency_ms, pruned_out = _time_inference(
        generator, inp, device, args.n_warmup, args.n_timed
    )
    pruned_np = _to_np_hwc(pruned_out)
    pruned_psnr = psnr(baseline_np, pruned_np)
    psnr_drop = 0.0 - (pruned_psnr - 0.0)  # drop vs ∞ baseline isn't meaningful;
    # the meaningful quantity is drop vs a gt-calibrated baseline from evaluate.py.
    # Here we report PSNR of pruned-vs-fp32-ref.

    print(f"[INFO] Pruned latency:  {pruned_latency_ms:.2f} ms")
    print(f"[INFO] PSNR (pruned vs FP32 ref): {pruned_psnr:.4f} dB")
    print(f"[INFO] Speedup: {baseline_latency_ms / pruned_latency_ms:.3f}×")
    print(
        "[NOTE] PSNR vs FP32 ref measures quantization/pruning error. "
        "Run scripts/restoration/evaluate.py for PSNR vs clean GT."
    )

    # --- Make pruning permanent ---
    print("[INFO] Making pruning permanent (prune.remove) ...")
    make_pruning_permanent(generator)

    # --- Save pruned model ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sparsity_tag = f"{int(args.sparsity * 100):03d}"
    model_save_path = args.output_dir / f"{model_name}_pruned_s{sparsity_tag}.pt"
    torch.save(generator.state_dict(), model_save_path)
    print(f"[INFO] Pruned model saved to: {model_save_path}")

    # --- JSON report ---
    report: dict[str, Any] = {
        "model_name": model_name,
        "checkpoint": str(args.checkpoint),
        "image": str(args.image),
        "device": device,
        "method": args.method,
        "target_sparsity": args.sparsity,
        "actual_sparsity": round(actual_sparsity, 6),
        "baseline_latency_ms": round(baseline_latency_ms, 3),
        "pruned_latency_ms": round(pruned_latency_ms, 3),
        "speedup_ratio": round(baseline_latency_ms / max(pruned_latency_ms, 1e-6), 4),
        "baseline_psnr_vs_fp32": None,  # by definition (baseline IS the fp32 reference)
        "pruned_psnr_vs_fp32": round(pruned_psnr, 4),
        "psnr_drop_db": round(pruned_psnr, 4),  # lower = more degradation
        "pruned_model_path": str(model_save_path),
        "per_layer_sparsity": layer_stats,
        "note": (
            "PSNR is vs FP32 un-pruned output on the same image (not vs clean GT). "
            "Speedup from unstructured pruning on GPU is typically minimal; "
            "real speedup requires sparse-capable runtimes."
        ),
    }
    report_path = args.output_dir / f"{model_name}_pruned_s{sparsity_tag}_report.json"
    _write_json_atomic(report_path, report)
    print(f"[INFO] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
