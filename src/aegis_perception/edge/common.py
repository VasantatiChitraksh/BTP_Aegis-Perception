"""Shared, explicit image geometry and artifact contracts for edge inference."""

import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/edge_v1"
MODELS = (
    "attention_pix2pix",
    "restormer",
    "liteweatherformer",
    "transweather",
    "promptir",
    "ramit",
)
HEIGHT, WIDTH = 256, 448
ROAD_CLASSES = [0, 1, 2, 3, 5, 7]
COCO_ROAD = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment():
    import importlib.metadata

    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    for package in ("torch", "onnx", "onnxruntime", "ultralytics", "numpy", "Pillow", "tensorrt"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    for key, path in (
        ("jetson_model", "/proc/device-tree/model"),
        ("jetpack_l4t", "/etc/nv_tegra_release"),
    ):
        result[key] = Path(path).read_text().strip("\x00\n") if Path(path).is_file() else None
    return result


def preprocess(image, height=HEIGHT, width=WIDTH):
    """RGB, aspect-preserving bilinear resize, bottom/right edge padding, NCHW [-1,1]."""
    image = image.convert("RGB")
    original_w, original_h = image.size
    scale = min(width / original_w, height / original_h)
    resized_w = max(1, min(width, round(original_w * scale)))
    resized_h = max(1, min(height, round(original_h * scale)))
    array = np.asarray(
        image.resize((resized_w, resized_h), Image.Resampling.BILINEAR), dtype=np.float32
    )
    array = np.pad(array, ((0, height - resized_h), (0, width - resized_w), (0, 0)), mode="edge")
    tensor = np.ascontiguousarray((array / 127.5 - 1).transpose(2, 0, 1)[None])
    return tensor, {"original_wh": [original_w, original_h], "resized_wh": [resized_w, resized_h]}


def postprocess(output, geometry):
    """Remove padding, clip, round RGB uint8, and restore original image geometry."""
    output = np.asarray(output)
    if output.ndim != 4 or output.shape[:2] != (1, 3) or not np.isfinite(output).all():
        raise ValueError("Invalid/nonfinite restoration output")
    width, height = geometry["resized_wh"]
    rgb = (
        ((np.clip(output[0, :, :height, :width], -1, 1).transpose(1, 2, 0) + 1) * 127.5)
        .round()
        .astype(np.uint8)
    )
    return Image.fromarray(rgb).resize(tuple(geometry["original_wh"]), Image.Resampling.BILINEAR)


def model_info(name):
    if name not in MODELS:
        raise ValueError(name)
    info = json.loads((ROOT / "artifacts/models/index.json").read_text())["models"][name]
    checkpoint = ROOT / info["library"] / "inference.pt"
    if digest(checkpoint) != info["inference_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {name}")
    return checkpoint, info


def ort_session(path, threads=4):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def select_panel(split, per_weather):
    """Fixed score-blind selection: rotate independent backgrounds before reusing any."""
    from collections import defaultdict

    from aegis_perception.data.manifest import load_manifest

    groups = defaultdict(lambda: defaultdict(list))
    for record in load_manifest(ROOT / "data/manifests/weather_v3/active.csv", split):
        groups[record.weather][record.group_id].append(record)

    def key(value):
        return hashlib.sha256(str(value).encode()).hexdigest()

    selected = []
    for weather in sorted(groups):
        ordered = [
            sorted(rows, key=lambda r: key(r.sample_id))
            for _, rows in sorted(groups[weather].items(), key=lambda item: key(item[0]))
        ]
        offset, chosen = 0, []
        while len(chosen) < per_weather:
            available = [rows[offset] for rows in ordered if offset < len(rows)]
            if not available:
                raise ValueError(f"Insufficient {split}/{weather} records")
            chosen.extend(available[: per_weather - len(chosen)])
            offset += 1
        selected.extend(chosen)
    return selected
