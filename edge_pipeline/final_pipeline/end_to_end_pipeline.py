#!/usr/bin/env python3
"""end_to_end_pipeline.py — Chained extreme-weather restoration and road object detection.

Pipeline Flow:
Degraded Image (Rain/Fog/Snow/Smoke)
       │
       ▼  Aspect-Preserving Resize & Normalization [-1, 1]
[Quantized Restormer Engine] (Accelerated via Devised Quantization Algorithm)
       │
       ▼  Denormalization [0, 255] RGB
[Restored Road Image]
       │
       ▼  Letterbox (640x640) & Road-Class Filtering [0, 1, 2, 3, 5, 7]
[YOLOv8 Detection Engine]
       │
       ▼  Bounding Boxes & Classes (Person, Bicycle, Car, Motorcycle, Bus, Truck)
Final Annotated Perception Output + Real-Time Telemetry
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from aegis_perception.checkpoints import load_generator
from edge_pipeline.yolov8_loader.load_yolov8 import YOLOv8Detector


class AegisEdgePipeline:
    """End-to-end edge perception pipeline combining quantized Restormer and YOLOv8."""

    def __init__(
        self,
        restormer_checkpoint: Union[str, Path],
        yolo_weights: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        restoration_geometry: Tuple[int, int] = (256, 448),
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ):
        self.device = device
        self.geometry = restoration_geometry
        self.restormer = None
        self.detector = None

        self._load_restormer(Path(restormer_checkpoint))
        self._load_detector(yolo_weights, conf_thresh, iou_thresh)

    def _load_restormer(self, checkpoint_path: Path) -> None:
        """Load and configure Restormer restoration model."""
        print(f"[PIPELINE] Loading Restormer model: {checkpoint_path}")
        ckpt_raw = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if isinstance(ckpt_raw, dict) and "model_config" in ckpt_raw and "state_dict" in ckpt_raw:
            from aegis_perception.models import build_generator
            self.restormer = build_generator(ckpt_raw["model_config"]).to(self.device)
            self.restormer.load_state_dict(ckpt_raw["state_dict"])
        elif isinstance(ckpt_raw, dict) and "generator" in ckpt_raw:
            self.restormer, _ = load_generator(checkpoint_path, device=self.device)
        elif isinstance(ckpt_raw, dict) and all(isinstance(v, torch.Tensor) for v in ckpt_raw.values()):
            # Raw state dict
            from edge_pipeline.import_restormer.import_restormer import RESTORMER_DEFAULT_CONFIG
            from aegis_perception.models import build_generator
            self.restormer = build_generator(RESTORMER_DEFAULT_CONFIG).to(self.device)
            self.restormer.load_state_dict(ckpt_raw)
        else:
            self.restormer, _ = load_generator(checkpoint_path, device=self.device)

        self.restormer.eval()
        print("[PIPELINE] Restormer loaded and initialized in eval mode.")

    def _load_detector(
        self,
        yolo_weights: Optional[Union[str, Path]],
        conf_thresh: float,
        iou_thresh: float,
    ) -> None:
        """Load and configure YOLOv8 road object detector."""
        print("[PIPELINE] Loading YOLOv8 road detector...")
        self.detector = YOLOv8Detector(
            weights_path=yolo_weights,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            device=self.device,
        )
        print("[PIPELINE] YOLOv8 detector loaded and ready.")

    def preprocess_restoration(
        self, img_bgr: np.ndarray
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess BGR image to normalized [-1, 1] tensor for Restormer."""
        orig_h, orig_w = img_bgr.shape[:2]
        target_h, target_w = self.geometry
        
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        norm = (resized.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, (orig_w, orig_h)

    def postprocess_restoration(
        self, out_tensor: torch.Tensor, orig_wh: Tuple[int, int]
    ) -> np.ndarray:
        """Postprocess Restormer output tensor back to full-resolution RGB image."""
        orig_w, orig_h = orig_wh
        arr = out_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        rgb = np.clip((arr + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
        restored_rgb = cv2.resize(rgb, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return restored_rgb

    def process_frame(
        self, img_bgr: np.ndarray, evaluate_raw: bool = True
    ) -> Dict[str, Any]:
        """Execute end-to-end perception: restoration followed by detection."""
        t_total_start = time.perf_counter()

        # 1. Restoration Stage
        t_rest_start = time.perf_counter()
        inp_tensor, orig_wh = self.preprocess_restoration(img_bgr)
        
        with torch.no_grad():
            out_tensor = self.restormer(inp_tensor)
        
        restored_rgb = self.postprocess_restoration(out_tensor, orig_wh)
        restoration_time_ms = (time.perf_counter() - t_rest_start) * 1000.0

        # 2. Detection Stage (on Restored Image)
        t_det_start = time.perf_counter()
        restored_detections = self.detector.detect(restored_rgb)
        detection_time_ms = (time.perf_counter() - t_det_start) * 1000.0

        # Optional: Baseline Detection (on Unrestored Raw Image) for A/B comparison
        raw_detections = None
        if evaluate_raw:
            raw_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            raw_detections = self.detector.detect(raw_rgb)

        total_time_ms = (time.perf_counter() - t_total_start) * 1000.0
        fps = 1000.0 / max(total_time_ms, 1e-4)

        return {
            "restored_rgb": restored_rgb,
            "restored_detections": restored_detections,
            "raw_detections": raw_detections,
            "latency": {
                "restoration_ms": round(restoration_time_ms, 2),
                "detection_ms": round(detection_time_ms, 2),
                "total_pipeline_ms": round(total_time_ms, 2),
                "fps": round(fps, 1),
            },
        }

    def render_side_by_side(
        self,
        img_bgr: np.ndarray,
        result: Dict[str, Any],
    ) -> np.ndarray:
        """Create split-screen visual comparison: Raw vs Restored with Detections."""
        h, w = img_bgr.shape[:2]
        
        # Left panel: Raw input with raw detections
        left = img_bgr.copy()
        if result["raw_detections"]:
            left = self.detector.draw_detections(
                left, result["raw_detections"], color=(0, 140, 255)  # Orange for raw
            )
        cv2.putText(
            left, "Raw Degraded Input", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2
        )

        # Right panel: Restored input with restored detections
        right = cv2.cvtColor(result["restored_rgb"], cv2.COLOR_RGB2BGR)
        right = self.detector.draw_detections(
            right, result["restored_detections"], color=(0, 255, 0)  # Green for restored
        )
        telemetry = (
            f"Restored + YOLOv8 | Latency: {result['latency']['total_pipeline_ms']}ms "
            f"({result['latency']['fps']} FPS)"
        )
        cv2.putText(
            right, telemetry, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

        # Concatenate horizontally
        side_by_side = np.hstack([left, right])
        return side_by_side
