#!/usr/bin/env python3
"""test_edge_pipeline.py — End-to-end integration test for the edge pipeline suite."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from edge_pipeline.final_pipeline.end_to_end_pipeline import AegisEdgePipeline
from edge_pipeline.import_restormer.import_restormer import (
    RESTORMER_DEFAULT_CONFIG,
    create_template_checkpoint,
    import_restormer_checkpoint,
)
from edge_pipeline.quantization_scripts.custom_quantizer import DevisedEdgeQuantizer
from edge_pipeline.yolov8_loader.load_yolov8 import YOLOv8Detector, letterbox


def test_import_restormer_template(tmp_path):
    """Verify that Restormer template creation and validation work without errors."""
    template_ckpt = create_template_checkpoint(tmp_path)
    assert template_ckpt.exists()

    model, imported_path = import_restormer_checkpoint(
        template_ckpt,
        tmp_path,
        device="cpu",
        test_geometry=(128, 128),
    )
    assert imported_path.exists()
    assert isinstance(model, torch.nn.Module)


def test_yolov8_detector_components():
    """Verify letterboxing geometry and detector initialization."""
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, geom = letterbox(dummy_img, target_size=640)
    assert tensor.shape == (1, 3, 640, 640)
    assert geom["original_wh"] == (640, 480)

    detector = YOLOv8Detector(device="cpu")
    dets = detector.detect(dummy_img)
    assert "boxes" in dets
    assert "scores" in dets
    assert "class_names" in dets


def test_devised_edge_quantizer(tmp_path):
    """Verify DevisedEdgeQuantizer scale computation and model quantization."""
    quantizer = DevisedEdgeQuantizer(num_bits=8, symmetric=True, per_channel=True)
    
    # Test tensor quantization
    sample_tensor = torch.randn(16, 32, 3, 3)
    scale, zp = quantizer.compute_scale_zero_point(sample_tensor, ch_axis=0)
    assert scale.shape == (16, 1, 1, 1)
    assert zp.shape == (16, 1, 1, 1)

    q = quantizer.quantize(sample_tensor, scale, zp)
    dq = quantizer.dequantize(q, scale, zp)
    assert dq.shape == sample_tensor.shape
    assert (dq - sample_tensor).abs().mean() < 0.1

    # Test small model quantization
    conv = torch.nn.Conv2d(3, 16, 3, padding=1)
    q_conv, stats = quantizer.apply_to_model(conv, preserve_sensitive_layers=False)
    assert len(stats["quantized_layers"]) == 1
    assert stats["compression_ratio"] > 1.0


def test_end_to_end_pipeline(tmp_path):
    """Verify the chained perception pipeline from degraded input to detection."""
    template_ckpt = create_template_checkpoint(tmp_path)
    pipeline = AegisEdgePipeline(
        restormer_checkpoint=template_ckpt,
        device="cpu",
        restoration_geometry=(128, 128),
    )

    dummy_bgr = np.full((240, 320, 3), 120, dtype=np.uint8)
    res = pipeline.process_frame(dummy_bgr, evaluate_raw=True)

    assert "restored_rgb" in res
    assert "restored_detections" in res
    assert "latency" in res
    assert res["restored_rgb"].shape == dummy_bgr.shape
    assert res["latency"]["fps"] > 0
