#!/usr/bin/env python3
"""run_inference.py — CLI application to run the final perception pipeline on edge targets.

Supports:
- Single image evaluation (`--image`)
- Directory batch inference (`--input-dir`)
- Video file processing (`--video`)
- Live camera stream (`--camera <index>`)
- Side-by-side comparison visualization and telemetry export

Usage:
------
# 1. Run on single image:
python edge_pipeline/final_pipeline/run_inference.py \\
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \\
    --image data/test_rain_sample.jpg \\
    --output-dir artifacts/edge_v1/pipeline_outputs

# 2. Run live camera on JetAuto Pro / Jetson:
python edge_pipeline/final_pipeline/run_inference.py \\
    --checkpoint edge_pipeline/checkpoints/restormer/restormer_imported.pt \\
    --camera 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from edge_pipeline.final_pipeline.end_to_end_pipeline import AegisEdgePipeline


def process_single_image(
    pipeline: AegisEdgePipeline,
    image_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print(f"\n[INFO] Processing image: {image_path}")
    result = pipeline.process_frame(img_bgr, evaluate_raw=True)

    lat = result["latency"]
    print(f"[SUCCESS] Restoration Time: {lat['restoration_ms']} ms")
    print(f"[SUCCESS] Detection Time:   {lat['detection_ms']} ms")
    print(f"[SUCCESS] Total Latency:    {lat['total_pipeline_ms']} ms ({lat['fps']} FPS)")
    print(f"[SUCCESS] Detections on Restored: {len(result['restored_detections']['boxes'])} objects "
          f"({result['restored_detections']['class_names']})")

    # Render and save visual comparison
    comparison = pipeline.render_side_by_side(img_bgr, result)
    out_img_path = output_dir / f"perception_{image_path.stem}.jpg"
    cv2.imwrite(str(out_img_path), comparison)
    print(f"[SUCCESS] Saved side-by-side perception image to: {out_img_path}")

    # Save telemetry JSON
    meta = {
        "source_image": str(image_path),
        "latency": lat,
        "restored_detections": {
            "count": len(result["restored_detections"]["boxes"]),
            "boxes": result["restored_detections"]["boxes"].tolist(),
            "scores": [round(float(s), 3) for s in result["restored_detections"]["scores"]],
            "class_names": result["restored_detections"]["class_names"],
        },
        "raw_detections": {
            "count": len(result["raw_detections"]["boxes"]) if result["raw_detections"] else 0,
            "class_names": result["raw_detections"]["class_names"] if result["raw_detections"] else [],
        },
    }
    out_json_path = output_dir / f"perception_{image_path.stem}.json"
    with open(out_json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[SUCCESS] Saved telemetry report to: {out_json_path}\n")


def process_camera_stream(
    pipeline: AegisEdgePipeline,
    camera_id: int,
    output_video_path: Optional[Path] = None,
) -> None:
    print(f"[INFO] Opening camera stream {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot access camera index {camera_id}")

    writer = None
    print("[INFO] Live camera inference running. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = pipeline.process_frame(frame, evaluate_raw=False)
            rendered = pipeline.render_side_by_side(frame, result)

            if output_video_path and writer is None:
                h, w = rendered.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_video_path), fourcc, 20.0, (w, h))

            if writer:
                writer.write(rendered)

            cv2.imshow("Aegis Edge Perception - Live Stream", rendered)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end de-weathering and YOLOv8 perception pipeline")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to Restormer checkpoint (imported or quantized)",
    )
    parser.add_argument("--image", type=Path, default=None, help="Path to single input image")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of test images")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index (e.g. 0)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "edge_v1" / "pipeline_outputs",
        help="Output directory for visual comparisons and reports",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    args = parser.parse_args()

    pipeline = AegisEdgePipeline(
        restormer_checkpoint=args.checkpoint,
        device=args.device,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    )

    if args.image:
        process_single_image(pipeline, args.image, args.output_dir)
    elif args.input_dir:
        image_files = sorted(args.input_dir.glob("*.jpg")) + sorted(args.input_dir.glob("*.png"))
        print(f"[INFO] Found {len(image_files)} images in {args.input_dir}")
        for img_p in image_files:
            process_single_image(pipeline, img_p, args.output_dir)
    elif args.camera is not None:
        process_camera_stream(pipeline, args.camera)
    else:
        print("[ERROR] Please provide --image, --input-dir, or --camera.")


if __name__ == "__main__":
    main()
