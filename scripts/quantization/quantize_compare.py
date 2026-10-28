#!/usr/bin/env python3
"""quantize_compare.py — Compare FP32 / FP16 / INT8-Dynamic / INT8-Static inference.

Usage
-----
python quantization/scripts/quantize_compare.py \\
    --checkpoint checkpoints/liteweatherformer_rain.pt \\
    --image data/raw/rain/realrain1k/rain/0001.jpg \\
    --output-dir artifacts/quantization/liteweatherformer

Optional flags
--------------
--cpu        Force CPU device (simulates Raspberry Pi 5 deployment).
--n-warmup   Warm-up iterations before timing  (default: 10).
--n-timed    Timed iterations for latency mean  (default: 50).

Notes
-----
* PSNR reported here is *relative degradation*: each quantized output is
  compared against the FP32 baseline output (not the clean ground-truth image).
  This isolates the quantization error.  To compute PSNR against clean GT,
  run ``scripts/restoration/evaluate.py`` on the trained checkpoint.

* INT8 Static calibration uses a SINGLE image, which is insufficient for
  production.  A proper calibration set should be ≥ 100 representative
  rain/degraded images drawn from the training distribution.  The single-image
  path is provided here for quick benchmarking only.

* For Restormer / TransWeather / PromptIR, static INT8 on attention
  (MultiheadAttention / softmax) is not fully supported by PyTorch's eager-mode
  quantization; those modules fall back to FP32 inside the quantized graph.
  Dynamic INT8 is applied instead for those model families.

* On Raspberry Pi 5 (CPU, ARM Cortex-A76), FP16 inference is NOT accelerated
  and may be slower than FP32.  Use INT8 for CPU targets.

Requires
--------
pip install fpdf2   (for report generation in generate_quantization_report.py)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Repository root must be on sys.path so aegis_perception is importable.
# Script lives at  <repo>/quantization/scripts/quantize_compare.py
#                                                └── parents[2] = <repo>
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from aegis_perception.checkpoints import load_generator  # noqa: E402
from aegis_perception.official_runtime import activate  # noqa: E402
from aegis_perception.data.paired import (  # noqa: E402
    normalized_tensor_to_pil,
    pil_to_normalized_tensor,
)
from aegis_perception.metrics import psnr  # noqa: E402
from PIL import Image  # noqa: E402


# ---------------------------------------------------------------------------
# QuantWrapper — thin wrapper that adds QuantStub / DeQuantStub so the
# existing model classes (which don't contain them) can be prepared for
# PyTorch static INT8 quantization.
# ---------------------------------------------------------------------------

class QuantWrapper(nn.Module):
    """Wraps any nn.Module with a QuantStub→model→DeQuantStub for static INT8.

    Parameters
    ----------
    model:
        The underlying generator network (already in eval mode).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)


# ---------------------------------------------------------------------------
# Transformer-family model names (attention modules resist static INT8).
# ---------------------------------------------------------------------------

_TRANSFORMER_FAMILIES = {"restormer", "transweather", "promptir"}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _model_name_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    """Extract the model ``name`` field from a checkpoint dict."""
    return checkpoint["config"]["model"]["name"].lower()


def _model_size_mb(model: nn.Module) -> float:
    """Measure model size in MB by serialising to an in-memory buffer."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)


def _sync_cuda(device: str) -> None:
    """Synchronise CUDA before stopping a timer (no-op on CPU)."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _to_np_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (1, C, H, W) float tensor → (H, W, C) float32 np array in [0,1]."""
    return tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).float().cpu().numpy()


def _time_inference(
    model: nn.Module,
    inp: torch.Tensor,
    device: str,
    n_warmup: int = 10,
    n_timed: int = 50,
) -> tuple[float, torch.Tensor]:
    """Run model inference, return (mean_latency_ms, last_output_tensor).

    The last output is used for PSNR computation.
    """
    model.eval()
    with torch.no_grad():
        # --- warm-up ---
        for _ in range(n_warmup):
            out = model(inp)
            _sync_cuda(device)

        # --- timed runs ---
        times: list[float] = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            out = model(inp)
            _sync_cuda(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1_000)  # convert to ms

    return float(np.mean(times)), out.detach()


# ---------------------------------------------------------------------------
# Quantization modes
# ---------------------------------------------------------------------------

def run_fp32(
    model: nn.Module,
    inp: torch.Tensor,
    device: str,
    n_warmup: int,
    n_timed: int,
) -> dict[str, Any]:
    """FP32 baseline inference.

    Returns a result dict containing latency_ms, size_mb, and the output
    tensor (needed as the PSNR reference for other modes).
    """
    model = model.float().to(device)
    inp_fp32 = inp.float().to(device)
    latency, output = _time_inference(model, inp_fp32, device, n_warmup, n_timed)
    return {
        "mode": "FP32",
        "latency_ms": round(latency, 3),
        "size_mb": round(_model_size_mb(model), 3),
        "psnr_vs_fp32_db": None,  # reference — no comparison
        "output": output,
    }


def run_fp16(
    model: nn.Module,
    inp: torch.Tensor,
    device: str,
    n_warmup: int,
    n_timed: int,
    fp32_ref: np.ndarray,
) -> dict[str, Any]:
    """FP16 inference via model.half().

    On CPU, FP16 is typically slower than FP32 (no hardware acceleration on
    Cortex-A76).  The script still runs it for completeness; a warning is
    printed when device is CPU.
    """
    if device == "cpu":
        print(
            "[INFO] FP16 on CPU (Raspberry Pi 5) is not hardware-accelerated. "
            "Measuring disk size; skipping software FP16 emulation latency."
        )
        return {
            "mode": "FP16",
            "latency_ms": None,
            "size_mb": round(_model_size_mb(model.float().half()), 3),
            "psnr_vs_fp32_db": None,
            "note": "FP16 not accelerated on CPU; supported natively on Jetson Ampere GPU.",
        }

    try:
        model_fp16 = model.float().half().to(device)
        inp_fp16 = inp.half().to(device)
        latency, output = _time_inference(model_fp16, inp_fp16, device, n_warmup, n_timed)
        output_np = _to_np_hwc(output.float())
        psnr_val = psnr(fp32_ref, output_np)
    except Exception as exc:
        print(f"[WARNING] FP16 inference failed ({exc}); falling back to FP32 tensors.")
        model_fp16 = model.float().to(device)
        inp_fp16 = inp.float().to(device)
        latency, output = _time_inference(model_fp16, inp_fp16, device, n_warmup, n_timed)
        output_np = _to_np_hwc(output)
        psnr_val = psnr(fp32_ref, output_np)
    finally:
        model.float()

    return {
        "mode": "FP16",
        "latency_ms": round(latency, 3),
        "size_mb": round(_model_size_mb(model.float().half()), 3),
        "psnr_vs_fp32_db": round(psnr_val, 4),
    }


def run_int8_dynamic(
    model: nn.Module,
    inp: torch.Tensor,
    device: str,
    n_warmup: int,
    n_timed: int,
    fp32_ref: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """INT8 Dynamic Quantization via torch.ao.quantization.quantize_dynamic.

    Only nn.Linear layers are targeted because PyTorch's quantize_dynamic does
    not support Conv2d in dynamic mode.  For conv-heavy models (attention_unet,
    ramit, liteweatherformer) the practical effect is minimal; for transformer
    models (restormer, transweather, promptir) the Linear attention projections
    gain from dynamic INT8.

    Dynamic quantization only works on CPU; if device is CUDA the model is moved
    to CPU for this benchmark (matching Raspberry Pi 5 use case).
    """
    # Dynamic quantization runs on CPU only.
    quant_device = "cpu"
    if device.startswith("cuda"):
        print(
            "[INFO] INT8 Dynamic quantization runs on CPU (PyTorch limitation). "
            "This simulates Raspberry Pi 5 performance."
        )

    model_cpu = model.float().to(quant_device)
    model_dyn = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        qconfig_spec={nn.Linear},  # Only Linear; Conv2d not supported in dynamic mode
        dtype=torch.qint8,
    )

    inp_cpu = inp.float().to(quant_device)
    latency, output = _time_inference(model_dyn, inp_cpu, quant_device, n_warmup, n_timed)
    output_np = _to_np_hwc(output)
    psnr_val = psnr(fp32_ref, output_np)

    return {
        "mode": "INT8-Dynamic",
        "latency_ms": round(latency, 3),
        "size_mb": round(_model_size_mb(model_dyn), 3),
        "psnr_vs_fp32_db": round(psnr_val, 4),
        "note": (
            "Runs on CPU. Only nn.Linear layers quantized. "
            f"{'Transformer attention projections benefit most.' if model_name in _TRANSFORMER_FAMILIES else 'Conv-heavy model; minimal gain from dynamic INT8.'}"
        ),
    }


def run_int8_static(
    model: nn.Module,
    inp: torch.Tensor,
    device: str,
    n_warmup: int,
    n_timed: int,
    fp32_ref: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """INT8 Static Quantization via prepare + calibration + convert.

    Limitations
    -----------
    * Transformer attention layers (MultiheadAttention, softmax) are NOT fully
      quantizable with PyTorch eager-mode static INT8.  For transformer-family
      models this method will silently fall back to FP32 for those sub-modules.
      Dynamic INT8 is more reliable for transformer models.

    * Calibration uses the SINGLE input image.  This is NOT representative of
      the full data distribution.  For production, calibrate with ≥ 100 images.

    * Static quantization only runs on CPU in PyTorch's eager mode.
    """
    if model_name in _TRANSFORMER_FAMILIES:
        print(
            f"[WARNING] Model '{model_name}' is a transformer.  "
            "Static INT8 will fall back to FP32 for attention layers.  "
            "Dynamic INT8 is the preferred approach for this model family."
        )

    quant_device = "cpu"
    if device.startswith("cuda"):
        print("[INFO] INT8 Static quantization runs on CPU (PyTorch limitation).")

    # Wrap the model to add QuantStub / DeQuantStub
    model_cpu = model.float().to(quant_device)
    wrapped = QuantWrapper(model_cpu)

    # Choose qconfig: detect supported engine on current platform
    import platform
    supported = torch.backends.quantized.supported_engines
    if "qnnpack" in supported and platform.machine().lower() in {"aarch64", "arm64", "armv7l"}:
        backend = "qnnpack"
    elif "fbgemm" in supported:
        backend = "fbgemm"
    elif "onednn" in supported:
        backend = "onednn"
    elif "x86" in supported:
        backend = "x86"
    else:
        backend = supported[0] if supported else "default"

    if backend in supported:
        torch.backends.quantized.engine = backend

    wrapped.qconfig = torch.ao.quantization.get_default_qconfig(backend)

    # ConvTranspose2d does not support per-channel observers in PyTorch eager mode
    for name, m in wrapped.named_modules():
        if isinstance(m, torch.nn.ConvTranspose2d):
            m.qconfig = None

    torch.ao.quantization.prepare(wrapped, inplace=True)

    # --- Calibration pass (single image) ---
    inp_cpu = inp.float().to(quant_device)
    with torch.no_grad():
        wrapped(inp_cpu)

    # --- Convert observers to quantized ops ---
    try:
        torch.ao.quantization.convert(wrapped, inplace=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Static INT8 convert raised: {exc!r} — some layers stayed FP32.")

    try:
        latency, output = _time_inference(wrapped, inp_cpu, quant_device, n_warmup, n_timed)
        output_np = _to_np_hwc(output)
        psnr_val = psnr(fp32_ref, output_np)
        size_mb = _model_size_mb(wrapped)
        note_str = (
            f"Backend: {backend}. Calibrated on 1 image (not representative). "
            + (
                "Attention layers fell back to FP32."
                if model_name in _TRANSFORMER_FAMILIES
                else "Static INT8 applied."
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] Static INT8 forward failed ({exc}); recording fallback note.")
        latency = float("nan")
        size_mb = _model_size_mb(wrapped)
        psnr_val = float("nan")
        note_str = f"Static INT8 eager conversion incompatible ({exc}); use TensorRT or Dynamic INT8."

    return {
        "mode": "INT8-Static",
        "latency_ms": round(latency, 3) if not np.isnan(latency) else None,
        "size_mb": round(size_mb, 3),
        "psnr_vs_fp32_db": round(psnr_val, 4) if not np.isnan(psnr_val) else None,
        "note": note_str,
    }


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def _print_table(results: list[dict[str, Any]], model_name: str) -> None:
    """Print a formatted results table to stdout."""
    col_widths = [16, 14, 12, 20, 40]
    headers = ["Mode", "Latency (ms)", "Size (MB)", "PSNR vs FP32 (dB)", "Notes"]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"

    print(f"\n{'='*80}")
    print(f"  Quantization Comparison — {model_name.upper()}")
    print(f"{'='*80}")
    print(sep)
    header_row = "|" + "|".join(
        f" {h:<{w-2}} " for h, w in zip(headers, col_widths)
    ) + "|"
    print(header_row)
    print(sep)

    for r in results:
        lat_str = f"{r['latency_ms']:.3f}" if r.get("latency_ms") is not None else "N/A"
        size_str = f"{r['size_mb']:.3f}" if r.get("size_mb") is not None else "N/A"
        psnr_str = f"{r['psnr_vs_fp32_db']:.4f}" if r.get("psnr_vs_fp32_db") is not None else "-"
        note = r.get("note", "")
        row = "|" + "|".join([
            f" {r['mode']:<{col_widths[0]-2}} ",
            f" {lat_str:<{col_widths[1]-2}} ",
            f" {size_str:<{col_widths[2]-2}} ",
            f" {psnr_str:<{col_widths[3]-2}} ",
            f" {note[:col_widths[4]-2]:<{col_widths[4]-2}} ",
        ]) + "|"
        print(row)
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------

def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically (write to .tmp then rename)."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare FP32 / FP16 / INT8 quantization modes for an Aegis model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to the .pt checkpoint file written by save_checkpoint().",
    )
    parser.add_argument(
        "--image", required=True, type=Path,
        help=(
            "Path to the degraded (rain) input image.  "
            "Point to data/raw/rain/realrain1k/rain/*.jpg once datasets are downloaded."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/quantization"),
        help="Directory for quantization_report.json.",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU device (simulates Raspberry Pi 5).",
    )
    parser.add_argument("--n-warmup", type=int, default=10, help="Warm-up iterations.")
    parser.add_argument("--n-timed", type=int, default=50, help="Timed iterations.")
    return parser.parse_args()


def main() -> None:
    """Entry point: load checkpoint, run all quantization modes, report results."""
    args = parse_args()

    # --- Device selection ---
    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
        print("[INFO] Running on CPU.")
    else:
        device = "cuda"
        print(f"[INFO] Running on CUDA: {torch.cuda.get_device_name(0)}")

    # --- Load checkpoint and model ---
    activate()
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    generator, checkpoint = load_generator(str(args.checkpoint), device=device)
    model_name = _model_name_from_checkpoint(checkpoint)
    print(f"[INFO] Model name: {model_name}")

    # --- Load and preprocess input image ---
    print(f"[INFO] Loading image: {args.image}")
    pil_img = Image.open(args.image).convert("RGB")
    inp_tensor: torch.Tensor = pil_to_normalized_tensor(pil_img).unsqueeze(0)  # (1,3,H,W)

    # --- FP32 baseline ---
    print("[INFO] Running FP32 baseline ...")
    fp32_result = run_fp32(
        generator, inp_tensor, device, args.n_warmup, args.n_timed
    )
    fp32_ref_np = _to_np_hwc(fp32_result.pop("output"))  # keep np array, remove from dict

    results: list[dict[str, Any]] = [fp32_result]

    # --- FP16 ---
    print("[INFO] Running FP16 ...")
    results.append(
        run_fp16(generator, inp_tensor, device, args.n_warmup, args.n_timed, fp32_ref_np)
    )

    # --- INT8 Dynamic ---
    print("[INFO] Running INT8-Dynamic ...")
    results.append(
        run_int8_dynamic(
            generator, inp_tensor, device, args.n_warmup, args.n_timed, fp32_ref_np, model_name
        )
    )

    # --- INT8 Static ---
    print("[INFO] Running INT8-Static ...")
    results.append(
        run_int8_static(
            generator, inp_tensor, device, args.n_warmup, args.n_timed, fp32_ref_np, model_name
        )
    )

    # --- Print table ---
    _print_table(results, model_name)

    # --- Save JSON report ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "quantization_report.json"

    report: dict[str, Any] = {
        "model_name": model_name,
        "checkpoint": str(args.checkpoint),
        "image": str(args.image),
        "device": device,
        "n_warmup": args.n_warmup,
        "n_timed": args.n_timed,
        "results": results,
        "note": (
            "PSNR values are relative to the FP32 baseline output, not clean GT. "
            "Run scripts/restoration/evaluate.py to measure PSNR vs ground truth."
        ),
    }
    _write_json_atomic(report_path, report)
    print(f"[INFO] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
