#!/usr/bin/env python3
"""edge_quantize.py — Execute and evaluate edge quantization for Restormer and restoration models.

Usage:
------
python edge_pipeline/quantization_scripts/edge_quantize.py \\
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \\
    --image data/test_rain_sample.jpg \\
    --output-dir edge_pipeline/checkpoints/restormer/quantized \\
    --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from aegis_perception.checkpoints import load_generator
from aegis_perception.metrics import psnr, ssim
from edge_pipeline.quantization_scripts.custom_quantizer import DevisedEdgeQuantizer


def preprocess_image(image_path: Path, height: int = 256, width: int = 448) -> Tuple[torch.Tensor, np.ndarray]:
    """Load, resize, and normalize input image to [-1, 1] tensor."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 127.5) - 1.0
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, rgb


def tensor_to_rgb(tensor: torch.Tensor, orig_w: int, orig_h: int) -> np.ndarray:
    """Convert [-1, 1] tensor back to uint8 RGB image."""
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    rgb = np.clip((arr + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
    return cv2.resize(rgb, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def benchmark_model(
    model: nn.Module,
    inp: torch.Tensor,
    n_warmup: int = 2,
    n_timed: int = 5,
) -> Tuple[float, torch.Tensor]:
    """Benchmark inference latency (mean ms) and return output tensor."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(inp)
        
        times = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            out = model(inp)
            times.append((time.perf_counter() - t0) * 1000.0)
    
    return float(np.mean(times)), out


def quantize_and_evaluate(
    checkpoint_path: Path,
    image_path: Path,
    output_dir: Path,
    device: str = "cpu",
    bits: int = 8,
    clipping_percentile: float = 99.95,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading model checkpoint: {checkpoint_path}")
    
    ckpt_raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" in ckpt_raw and "state_dict" in ckpt_raw:
        from aegis_perception.models import build_generator
        model = build_generator(ckpt_raw["model_config"]).to(device)
        model.load_state_dict(ckpt_raw["state_dict"])
    else:
        model, _ = load_generator(checkpoint_path, device=device)

    model.eval()

    # Preprocess image
    h, w = 256, 448
    inp_tensor, orig_rgb = preprocess_image(image_path, height=h, width=w)
    inp_tensor = inp_tensor.to(device)

    # 1. Baseline FP32 Performance
    print("[INFO] Evaluating Baseline FP32 performance...")
    fp32_latency, fp32_out = benchmark_model(model, inp_tensor)
    fp32_rgb = tensor_to_rgb(fp32_out, orig_rgb.shape[1], orig_rgb.shape[0])
    
    # 2. Apply Devised Edge Quantizer
    print(f"[INFO] Initializing Devised Edge Quantizer ({bits}-bit, percentile={clipping_percentile}%)...")
    quantizer = DevisedEdgeQuantizer(
        num_bits=bits,
        symmetric=True,
        per_channel=True,
        clipping_percentile=clipping_percentile,
    )
    
    # Calibration pass
    quantizer.calibrate(model, [inp_tensor])

    # Apply quantization
    print("[INFO] Quantizing model weights using Devised Algorithm...")
    quant_model, quant_stats = quantizer.apply_to_model(model, preserve_sensitive_layers=True)
    
    # 3. Quantized Model Performance
    print("[INFO] Evaluating Quantized Model performance...")
    quant_latency, quant_out = benchmark_model(quant_model, inp_tensor)
    quant_rgb = tensor_to_rgb(quant_out, orig_rgb.shape[1], orig_rgb.shape[0])

    # 4. Metric Computation (Relative to FP32 reference)
    psnr_ref = psnr(fp32_rgb.astype(np.float32) / 255.0, quant_rgb.astype(np.float32) / 255.0)
    ssim_ref = ssim(fp32_rgb.astype(np.float32) / 255.0, quant_rgb.astype(np.float32) / 255.0)

    # File size metrics
    fp32_mb = quant_stats["total_weight_bytes_fp32"] / (1024 * 1024)
    quant_mb = quant_stats["simulated_weight_bytes_int8"] / (1024 * 1024)
    compression = quant_stats["compression_ratio"]

    report = {
        "model": "Restormer",
        "algorithm": "DevisedEdgeQuantizer",
        "bits": bits,
        "clipping_percentile": clipping_percentile,
        "fp32_latency_ms": round(fp32_latency, 2),
        "quantized_latency_ms": round(quant_latency, 2),
        "fp32_weight_mb": round(fp32_mb, 2),
        "quantized_weight_mb": round(quant_mb, 2),
        "compression_ratio": round(compression, 2),
        "relative_psnr_to_fp32_db": round(psnr_ref, 4),
        "relative_ssim_to_fp32": round(ssim_ref, 6),
        "quantized_layers_count": len(quant_stats["quantized_layers"]),
        "preserved_layers_count": len(quant_stats["preserved_layers"]),
    }

    # Print summary table
    print("\n" + "=" * 70)
    print(">>> DEVISED EDGE QUANTIZATION RESULTS <<<")
    print("=" * 70)
    print(f"FP32 Model Size:             {fp32_mb:.2f} MB")
    print(f"Quantized Model Size:        {quant_mb:.2f} MB ({compression:.2f}x compression)")
    print(f"FP32 Inference Latency:      {fp32_latency:.2f} ms")
    print(f"Quantized Inference Latency: {quant_latency:.2f} ms")
    print(f"PSNR Retention vs FP32:      {psnr_ref:.2f} dB")
    print(f"SSIM Retention vs FP32:      {ssim_ref:.4f}")
    print(f"Quantized Layers:            {len(quant_stats['quantized_layers'])} layers")
    print(f"Preserved Sensitive Layers:  {len(quant_stats['preserved_layers'])} layers")
    print("=" * 70 + "\n")

    # Save artifacts
    report_path = output_dir / "edge_quantization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    calib_path = output_dir / "quantization_scales.json"
    quantizer.export_calibration_table(calib_path)

    # Save restored visual comparisons
    cv2.imwrite(str(output_dir / "restored_fp32.jpg"), cv2.cvtColor(fp32_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "restored_quantized.jpg"), cv2.cvtColor(quant_rgb, cv2.COLOR_RGB2BGR))

    # Save quantized model weights
    quant_model_path = output_dir / "restormer_quantized_sim.pt"
    torch.save(quant_model.state_dict(), quant_model_path)
    print(f"[SUCCESS] Quantized model saved to: {quant_model_path}")
    print(f"[SUCCESS] Quantization report saved to: {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run devised edge quantization on Restormer model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to imported Restormer checkpoint")
    parser.add_argument("--image", type=Path, required=True, help="Path to input test image")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "edge_pipeline" / "checkpoints" / "restormer" / "quantized",
        help="Directory to save quantized model and report",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device")
    parser.add_argument("--bits", type=int, default=8, help="Quantization bit-width (8 or 4)")
    parser.add_argument("--percentile", type=float, default=99.95, help="Outlier clipping percentile")
    args = parser.parse_args()

    quantize_and_evaluate(
        args.checkpoint,
        args.image,
        args.output_dir,
        device=args.device,
        bits=args.bits,
        clipping_percentile=args.percentile,
    )


if __name__ == "__main__":
    main()
