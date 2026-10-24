#!/usr/bin/env python3
"""Export frozen official YOLOv8n and check backend + NMS parity on real images."""

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ["YOLO_AUTOINSTALL"] = "false"
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO, settings
from ultralytics.utils.ops import non_max_suppression, scale_boxes

from aegis_perception.edge.common import (
    OUT,
    ROAD_CLASSES,
    digest,
    environment,
    ort_session,
    write_json,
)
from aegis_perception.edge.detector import decode, letterbox

settings.update({"sync": False, "wandb": False})
torch.set_num_threads(2)
weights = ROOT / "artifacts/pretrained/detection/yolov8n.pt"
directory = OUT / "exports/yolov8n"
directory.mkdir(parents=True, exist_ok=True)
# Keep exporter outputs isolated from the original downloaded file.
shutil.copy2(weights, directory / "yolov8n.pt")
model = YOLO(str(directory / "yolov8n.pt"))
exported = model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    simplify=False,
    opset=17,
    half=False,
    device="cpu",
)
session = ort_session(exported, threads=2)
checks, fixtures = [], {}
for weather in ("fog", "rain", "snow"):
    path = sorted((ROOT / "data/raw/dawn" / weather).rglob("*.jpg"))[0]
    image = np.asarray(Image.open(path).convert("RGB"))
    x, geometry = letterbox(image)
    with torch.inference_mode():
        expected = model.model(torch.from_numpy(x))
        if isinstance(expected, tuple):
            expected = expected[0]
        expected = expected.numpy()
    actual = session.run(None, {"images": x})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=2e-3)
    ours = decode(actual, geometry)
    theirs = non_max_suppression(torch.from_numpy(actual.copy()), 0.25, 0.7, classes=ROAD_CLASSES)[
        0
    ]
    theirs[:, :4] = scale_boxes((640, 640), theirs[:, :4], image.shape[:2])
    np.testing.assert_allclose(ours, theirs.numpy(), atol=0.02, rtol=1e-4)
    fixtures["input_" + weather] = x
    fixtures["expected_" + weather] = actual
    checks.append(
        {
            "weather": weather,
            "detections": len(ours),
            "max_abs_error": float(np.abs(actual - expected).max()),
            "decoder_matches_ultralytics": True,
        }
    )
np.savez_compressed(directory / "fixtures.npz", **fixtures)
write_json(
    directory / "validation.json",
    {
        "passed": True,
        "weights_sha256": digest(weights),
        "onnx_sha256": digest(exported),
        "source": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "license": "Ultralytics AGPL-3.0; review terms before distribution/commercial use",
        "shape": [1, 3, 640, 640],
        "output": [1, 84, 8400],
        "checks": checks,
        "environment": environment(),
    },
)
print(json.dumps(checks, indent=2))
