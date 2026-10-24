#!/usr/bin/env python3
"""Audit DAWN labels and evaluate frozen YOLOv8n on paired deployment views."""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ["YOLO_AUTOINSTALL"] = "false"

import numpy as np
import yaml
from PIL import Image

from aegis_perception.edge.common import (
    COCO_ROAD,
    OUT,
    ROAD_CLASSES,
    digest,
    environment,
    ort_session,
    postprocess,
    preprocess,
    write_json,
)

DATA = OUT / "dawn_pilot72"
PER_WEATHER = 24
WEIGHTS = ROOT / "artifacts/pretrained/detection/yolov8n.pt"
GRAPH = OUT / "exports/256x448/restormer/restoration.onnx"
VIEWS = ("raw", "resize_control", "restormer")


def annotations(image_path):
    """Validate source TXT against named VOC boxes, then normalize COCO IDs."""
    xmls = list(image_path.parent.rglob(image_path.stem + ".xml"))
    texts = list(image_path.parent.rglob(image_path.stem + ".txt"))
    if len(xmls) != 1 or len(texts) != 1:
        raise ValueError(f"Ambiguous/missing annotations: {image_path}")
    root = ET.parse(xmls[0]).getroot()
    with Image.open(image_path) as image:
        width, height = image.size
    if (int(root.findtext("size/width")), int(root.findtext("size/height"))) != (width, height):
        raise ValueError(f"Wrong annotation dimensions: {image_path}")
    lines = [line.split() for line in texts[0].read_text().splitlines() if line.strip()]
    objects = root.findall("object")
    if len(lines) != len(objects):
        raise ValueError("VOC/TXT counts differ")
    labels, counts, excluded = [], Counter(), Counter()
    mapping = {**COCO_ROAD, "train": 6}
    for obj, line in zip(objects, lines, strict=False):
        name = obj.findtext("name").strip().lower()
        class_id = mapping[name]
        if len(line) != 5 or int(line[0]) != class_id + 1:
            raise ValueError(f"Unexpected source class mapping: {image_path}: {line}")
        box = obj.find("bndbox")
        xmin, ymin, xmax, ymax = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        expected = np.array(
            [
                (xmin + xmax) / (2 * width),
                (ymin + ymax) / (2 * height),
                (xmax - xmin) / width,
                (ymax - ymin) / height,
            ]
        )
        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            raise ValueError("Out-of-bounds or degenerate box")
        np.testing.assert_allclose(np.array(line[1:], dtype=float), expected, atol=2e-5, rtol=0)
        if class_id in ROAD_CLASSES:
            labels.append(str(class_id) + " " + " ".join(f"{v:.8f}" for v in expected))
            counts[name] += 1
        else:
            excluded[name] += 1
    return "\n".join(labels) + ("\n" if labels else ""), counts, excluded


def prepare():
    import torch
    from ultralytics import YOLO, settings

    settings.update({"sync": False, "wandb": False})
    torch.set_num_threads(4)
    names = YOLO(str(WEIGHTS)).names
    dataset_key = {
        "onnx_sha256": digest(GRAPH),
        "preprocessing_sha256": digest(ROOT / "src/aegis_perception/edge/common.py"),
    }
    state = DATA / "progress.json"
    previous = json.loads(state.read_text()) if state.exists() else {}
    if previous and previous["contract"] != dataset_key:
        raise ValueError(
            "DAWN cache contract changed; choose a new output directory, do not mix views"
        )
    completed = previous.get("completed", {})
    session = ort_session(GRAPH)
    rows, counts, excluded = [], Counter(), Counter()
    image_paths = sorted((ROOT / "data/raw/dawn").rglob("*.jpg"))
    if PER_WEATHER:
        import hashlib

        image_paths = [
            p
            for weather in ("fog", "rain", "snow")
            for p in sorted(
                (
                    p
                    for p in image_paths
                    if p.relative_to(ROOT / "data/raw/dawn").parts[0] == weather
                ),
                key=lambda p: hashlib.sha256(p.name.encode()).hexdigest(),
            )[:PER_WEATHER]
        ]
    for index, path in enumerate(image_paths, 1):
        weather = path.relative_to(ROOT / "data/raw/dawn").parts[0]
        stem = weather + "__" + path.stem
        labels, class_counts, omitted = annotations(path)
        counts.update(class_counts)
        excluded.update(omitted)
        source_hash = digest(path)
        expected_files = [DATA / view / "images" / f"{stem}.png" for view in VIEWS]
        if completed.get(stem) != source_hash or not all(p.exists() for p in expected_files):
            with Image.open(path) as image:
                image = image.convert("RGB")
                x, geometry = preprocess(image)
                restored = postprocess(session.run(None, {"degraded": x})[0], geometry)
                control = postprocess(x, geometry)
                for view, output in zip(VIEWS, (image, control, restored), strict=False):
                    target = DATA / view / "images" / f"{stem}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output.save(target)
            completed[stem] = source_hash
            write_json(state, {"contract": dataset_key, "completed": completed})
        for view in VIEWS:
            target = DATA / view / "labels" / f"{stem}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(labels)
        rows.append(
            {
                "id": stem,
                "weather": weather,
                "source": str(path.relative_to(ROOT)),
                "source_sha256": source_hash,
                "road_objects": sum(class_counts.values()),
            }
        )
        if index % 20 == 0:
            print(f"Prepared {index}/{len(image_paths)} paired DAWN images", flush=True)
    expected_count = 3 * PER_WEATHER if PER_WEATHER else 704
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} DAWN images, found {len(rows)}")
    for view in VIEWS:
        for weather in ("all", "fog", "rain", "snow"):
            selected = [r for r in rows if weather == "all" or r["weather"] == weather]
            listing = DATA / view / f"{weather}.txt"
            listing.write_text(
                "".join(str(DATA / view / "images" / (r["id"] + ".png")) + "\n" for r in selected)
            )
            config = {
                "path": str(DATA / view),
                "train": str(listing),
                "val": str(listing),
                "names": names,
            }
            (DATA / view / f"{weather}.yaml").write_text(yaml.safe_dump(config))
    duplicates = {h: n for h, n in Counter(r["source_sha256"] for r in rows).items() if n > 1}
    write_json(
        DATA / "audit.json",
        {
            "contract": dataset_key,
            "images": len(rows),
            "rows": rows,
            "counts": dict(counts),
            "excluded_objects_outside_six_road_classes": dict(excluded),
            "weather_counts": dict(Counter(r["weather"] for r in rows)),
            "exact_duplicate_hashes": duplicates,
            "scope": "Score-blind balanced engineering pilot; not the full DAWN benchmark"
            if PER_WEATHER
            else "Full available 704-image DAWN evaluation",
            "selection": "First SHA256(filename) entries per weather; no score-based selection",
            "annotation_mapping": (
                "Source TXT class ID minus one; every box and class verified against VOC XML"
            ),
            "source": "https://data.mendeley.com/datasets/766ygrbt8y/3",
            "license": "CC BY-NC 3.0",
            "limits": (
                "No smoke labels; detector pretraining overlap unverified. These "
                "are evaluation-only views, not new training splits."
            ),
        },
    )


def evaluate():
    import torch
    from ultralytics import YOLO, settings

    settings.update({"sync": False, "wandb": False})
    torch.set_num_threads(4)
    audit = json.loads((DATA / "audit.json").read_text())
    results = OUT / ("detection_pilot72" if PER_WEATHER else "detection_full704")
    contract = {
        "audit_sha256": digest(DATA / "audit.json"),
        "detector_sha256": digest(WEIGHTS),
        "imgsz": 640,
        "rect": False,
        "batch": 1,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "classes": ROAD_CLASSES,
        "augment": False,
        "precision": "FP32",
        "ultralytics": "8.3.0",
    }
    for view in VIEWS:
        for weather in ("all", "fog", "rain", "snow"):
            target = results / f"{view}_{weather}.json"
            if target.exists() and json.loads(target.read_text()).get("contract") == contract:
                continue
            started = time.monotonic()
            model = YOLO(str(WEIGHTS))
            metrics = model.val(
                data=str(DATA / view / f"{weather}.yaml"),
                device="cpu",
                imgsz=640,
                batch=1,
                workers=0,
                conf=0.001,
                iou=0.7,
                max_det=300,
                classes=ROAD_CLASSES,
                rect=False,
                augment=False,
                half=False,
                plots=False,
                save_json=False,
                verbose=False,
                project=str(results / "runs"),
                name=f"{view}_{weather}",
                exist_ok=True,
            )
            per_class = []
            for i, class_id in enumerate(metrics.box.ap_class_index):
                p, r, ap50, ap = metrics.box.class_result(i)
                per_class.append(
                    {
                        "class_id": int(class_id),
                        "name": model.names[int(class_id)],
                        "precision": p,
                        "recall": r,
                        "map50": ap50,
                        "map50_95": ap,
                    }
                )
            write_json(
                target,
                {
                    "view": view,
                    "weather": weather,
                    "contract": contract,
                    "images": audit["images"]
                    if weather == "all"
                    else audit["weather_counts"][weather],
                    "metrics": metrics.results_dict,
                    "per_class": per_class,
                    "elapsed_seconds": time.monotonic() - started,
                    "environment": environment(),
                    "note": (
                        "P/R at max-F1 confidence; mAP over confidence sweep. CPU "
                        "evaluation, not edge FPS."
                    ),
                },
            )
            print(f"SAVED {target}", flush=True)
    records = [
        json.loads((results / f"{v}_{w}.json").read_text())
        for v in VIEWS
        for w in ("all", "fog", "rain", "snow")
    ]
    write_json(results / "comparison.json", {"complete": True, "evaluations": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate", "all"))
    parser.add_argument(
        "--full", action="store_true", help="Evaluate all 704 images; several CPU hours"
    )
    args = parser.parse_args()
    if args.full:
        PER_WEATHER = 0
        DATA = OUT / "dawn_full704"
    if args.action in ("prepare", "all"):
        prepare()
    if args.action in ("evaluate", "all"):
        evaluate()
