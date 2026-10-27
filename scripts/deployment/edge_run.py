#!/usr/bin/env python3
"""Run Restormer -> frozen YOLOv8n on files/video/camera; benchmark actual wall time."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
from PIL import Image

from aegis_perception.edge.common import (
    COCO_ROAD,
    digest,
    environment,
    ort_session,
    postprocess,
    preprocess,
    write_json,
)
from aegis_perception.edge.detector import decode, letterbox


class Backend:
    def __init__(self, path):
        self.path = Path(path)
        self.trt = self.path.suffix == ".engine"
        self.requested_precision = None
        if self.trt:
            from aegis_perception.edge.trt_runtime import TrtSession

            record = json.loads(self.path.with_suffix(".json").read_text())
            if not record["validation"]["passed"] or record["engine_sha256"] != digest(self.path):
                raise ValueError("Unvalidated or changed engine")
            self.requested_precision = record.get("contract", {}).get("precision")
            self.session = TrtSession(path)
            self.shape = self.session.input_shape
        else:
            self.session = ort_session(path)
            self.shape = tuple(self.session.get_inputs()[0].shape)
            self.name = self.session.get_inputs()[0].name

    def __call__(self, array):
        if self.trt:
            return self.session.run(array)
        return self.session.run(None, {self.name: array})[0]

    def close(self):
        if self.trt:
            self.session.close()


def process(rgb, restorer, detector, mode, conf):
    started = time.perf_counter()
    if mode == "restored":
        height, width = restorer.shape[2:]
        x, geometry = preprocess(Image.fromarray(rgb), height, width)
        prepared = time.perf_counter()
        output = restorer(x)
        restored_at = time.perf_counter()
        rgb = np.asarray(postprocess(output, geometry))
    elif mode == "resize_control":
        x, geometry = preprocess(Image.fromarray(rgb))
        rgb = np.asarray(postprocess(x, geometry))
        prepared = restored_at = time.perf_counter()
    else:
        prepared = restored_at = time.perf_counter()
    x, geometry = letterbox(rgb)
    detector_start = time.perf_counter()
    output = detector(x)
    detector_end = time.perf_counter()
    boxes = decode(output, geometry, conf=conf)
    finished = time.perf_counter()
    return (
        rgb,
        boxes,
        {
            "restoration_preprocess_ms": (prepared - started) * 1000,
            "restoration_call_ms": (restored_at - prepared) * 1000,
            "restoration_post_and_detector_pre_ms": (detector_start - restored_at) * 1000,
            "detector_call_ms": (detector_end - detector_start) * 1000,
            "decode_ms": (finished - detector_end) * 1000,
            "pipeline_ms": (finished - started) * 1000,
        },
    )


def draw(rgb, boxes):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    names = {v: k for k, v in COCO_ROAD.items()}
    for x1, y1, x2, y2, score, cls in boxes:
        cv2.rectangle(bgr, (round(x1), round(y1)), (round(x2), round(y2)), (70, 210, 40), 2)
        cv2.putText(
            bgr,
            f"{names[int(cls)]} {score:.2f}",
            (round(x1), max(15, round(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (70, 210, 40),
            1,
            cv2.LINE_AA,
        )
    return bgr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restorer", type=Path)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument(
        "--source", required=True, help="Image, video, directory of images, or camera:0"
    )
    parser.add_argument("--mode", choices=("raw", "restored", "resize_control"), default="raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat image input for sustained benchmark"
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes all frames")
    parser.add_argument("--save-visuals", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or not 0 < args.conf < 1:
        parser.error("Invalid repeat/warmup/confidence")
    if args.mode == "restored" and args.restorer is None:
        parser.error("--restorer is required for restored mode")
    if args.output.exists():
        parser.error("Output directory already exists; use a new path to preserve previous results")
    args.output.mkdir(parents=True)
    cv2.setNumThreads(1)
    restorer = Backend(args.restorer) if args.mode == "restored" else None
    detector = Backend(args.detector)
    if detector.shape != (1, 3, 640, 640):
        raise ValueError("Detector must use static [1,3,640,640] input")
    source = Path(args.source)
    images = []
    if source.is_dir():
        images = sorted(
            p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
    elif source.suffix.lower() in (".jpg", ".jpeg", ".png"):
        images = [source]
    capture = None
    if not images:
        capture = cv2.VideoCapture(
            int(args.source.split(":")[1]) if args.source.startswith("camera:") else args.source
        )
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {args.source}")
    writer = None
    timings = []
    started = None
    index = 0
    try:
        with (args.output / "detections.jsonl").open("w") as stream:
            while True:
                frame_start = time.perf_counter()
                if images:
                    if index >= len(images) * args.repeat:
                        break
                    path = images[index % len(images)]
                    frame = cv2.imread(str(path))
                    source_name = str(path)
                else:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    source_name = f"frame:{index}"
                if frame is None:
                    raise ValueError("Unreadable image")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if index == 0:
                    for _ in range(args.warmup):
                        process(rgb, restorer, detector, args.mode, args.conf)
                    started = time.perf_counter()
                    frame_start = started  # First frame decode excluded after warmup.
                restored, boxes, row = process(rgb, restorer, detector, args.mode, args.conf)
                if args.save_visuals:
                    overlay = draw(restored, boxes)
                    if images:
                        if not cv2.imwrite(str(args.output / f"{index:06d}.jpg"), overlay):
                            raise OSError("Failed to write image")
                    else:
                        if writer is None:
                            fps = capture.get(cv2.CAP_PROP_FPS)
                            fps = fps if np.isfinite(fps) and fps > 0 else 20
                            writer = cv2.VideoWriter(
                                str(args.output / "annotated.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps,
                                (overlay.shape[1], overlay.shape[0]),
                            )
                            if not writer.isOpened():
                                raise RuntimeError("Video encoder unavailable")
                        writer.write(overlay)
                stream.write(
                    json.dumps(
                        {"source": source_name, "boxes_xyxy_score_coco_class": boxes.tolist()}
                    )
                    + "\n"
                )
                row["frame_wall_ms"] = (time.perf_counter() - frame_start) * 1000
                timings.append(row)
                index += 1
                if args.max_frames and index >= args.max_frames:
                    break
        elapsed = time.perf_counter() - started if started else 0
        if not timings:
            raise ValueError("No frames processed")
        summary = {
            key: {
                "median": float(np.median([r[key] for r in timings])),
                "p95": float(np.percentile([r[key] for r in timings], 95)),
            }
            for key in timings[0]
        }
        write_json(
            args.output / "benchmark.json",
            {
                "frames": len(timings),
                "elapsed_seconds": elapsed,
                "measured_throughput_fps": len(timings) / elapsed,
                "timings_ms": summary,
                "mode": args.mode,
                "source": args.source,
                "warmup_excluded": args.warmup,
                "save_visuals": args.save_visuals,
                "detector_sha256": digest(args.detector),
                "restorer_sha256": digest(args.restorer) if restorer else None,
                "restorer_file_bytes": args.restorer.stat().st_size if restorer else None,
                "detector_file_bytes": args.detector.stat().st_size,
                "restorer_requested_precision": restorer.requested_precision if restorer else None,
                "detector_requested_precision": detector.requested_precision,
                "environment": environment(),
                "detector_backend": "TensorRT" if detector.trt else "ONNX Runtime CPU",
                "restorer_backend": ("TensorRT" if restorer.trt else "ONNX Runtime CPU")
                if restorer
                else None,
                "detector_shape": list(detector.shape),
                "restoration_shape": list(restorer.shape) if restorer else None,
                "confidence": args.conf,
                "scope": (
                    "Synchronous batch-one pipeline. Model calls include host/device "
                    "copies. Frame wall includes I/O except first-frame decode; "
                    "elapsed includes output. Not sensor-to-display latency."
                ),
                "accuracy_note": (
                    "No ground-truth mAP computed by this demo. See separate annotated evaluation."
                ),
            },
        )
        print(
            json.dumps(
                {"frames": len(timings), "fps": len(timings) / elapsed, "output": str(args.output)}
            )
        )
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        detector.close()
        if restorer is not None:
            restorer.close()


if __name__ == "__main__":
    main()
