#!/usr/bin/env python3
"""Standalone Python 3.6 / TensorRT 8.2 YOLOv8n runner; no cv_bridge or torch.

Always validate against the frozen detector fixtures before demonstrating.
Only road classes are displayed. Optional Restormer uses nano_restormer.py.
"""

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np

FIXTURE_SHA256 = "4691860117a9abd2c3d3ee416ea4a4f0534cc1c39db1bfcbf4575a4a0096932e"
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def digest(path):
    value = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Engine:
    def __init__(self, path, kind="yolo"):
        import tensorrt as trt
        import pycuda.driver as cuda

        self.cuda = cuda
        self.version = trt.__version__
        if not self.version.startswith("8."):
            raise RuntimeError("This runner requires the TensorRT 8 binding API")
        cuda.init()
        # Share the device primary context between both models on the 4GB Nano.
        self.device_context = cuda.Device(0).retain_primary_context()
        self.device_context.push()
        self.allocations = []
        try:
            self.logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self.logger, "")
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
            if self.engine is None or self.engine.num_bindings != 2:
                raise RuntimeError("Expected a valid engine with exactly two bindings")
            inputs = [i for i in range(2) if self.engine.binding_is_input(i)]
            outputs = [i for i in range(2) if not self.engine.binding_is_input(i)]
            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError("Expected one input and one output")
            self.input_index, self.output_index = inputs[0], outputs[0]
            if kind not in ("yolo", "restormer"):
                raise ValueError("Unknown engine kind")
            self.input_shape = tuple(self.engine.get_binding_shape(self.input_index))
            allowed = (
                ((1, 3, 640, 640),) if kind == "yolo" else ((1, 3, 256, 448), (1, 3, 128, 224))
            )
            if self.input_shape not in allowed:
                raise ValueError("Unsupported static input shape: " + str(self.input_shape))
            self.output_shape = (1, 84, 8400) if kind == "yolo" else self.input_shape
            expected = {self.input_index: self.input_shape, self.output_index: self.output_shape}
            for i in range(2):
                if tuple(self.engine.get_binding_shape(i)) != expected[i]:
                    raise RuntimeError(
                        "Unexpected engine shape: " + str(self.engine.get_binding_shape(i))
                    )
                if np.dtype(trt.nptype(self.engine.get_binding_dtype(i))) != np.dtype("float32"):
                    raise RuntimeError("Expected FP32 engine input/output buffers")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Could not create execution context")
            self.stream = cuda.Stream()
            self.host = []
            for i in range(2):
                host = cuda.pagelocked_empty(int(np.prod(expected[i])), np.float32)
                self.host.append(host)
                self.allocations.append(cuda.mem_alloc(host.nbytes))
            self.bindings = [int(pointer) for pointer in self.allocations]
        except Exception:
            self.device_context.pop()
            self.close()
            raise
        self.device_context.pop()

    def run(self, value):
        self.device_context.push()
        try:
            return self._run(value)
        finally:
            self.device_context.pop()

    def _run(self, value):
        if value.shape != self.input_shape or not np.isfinite(value).all():
            raise ValueError("Invalid input; expected " + str(self.input_shape))
        np.copyto(self.host[self.input_index], value.ravel())
        self.cuda.memcpy_htod_async(
            self.allocations[self.input_index], self.host[self.input_index], self.stream
        )
        if not self.context.execute_async_v2(self.bindings, self.stream.handle):
            raise RuntimeError("TensorRT execution failed")
        self.cuda.memcpy_dtoh_async(
            self.host[self.output_index], self.allocations[self.output_index], self.stream
        )
        self.stream.synchronize()
        output = self.host[self.output_index].reshape(self.output_shape).copy()
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite model output")
        return output

    def close(self):
        if self.device_context is None:
            return
        self.device_context.push()
        if hasattr(self, "stream"):
            self.stream.synchronize()
        self.context = None
        self.engine = None
        self.runtime = None
        for allocation in self.allocations:
            allocation.free()
        self.allocations = []
        self.host = []
        self.stream = None
        self.device_context.pop()
        self.device_context.detach()
        self.device_context = None


def validate(engine, fixtures):
    rows = []
    with np.load(str(fixtures), allow_pickle=False) as data:
        for key in sorted(k for k in data.files if k.startswith("input_")):
            actual = engine.run(data[key])
            expected = data[key.replace("input_", "expected_")]
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError("Invalid validation output")
            difference = actual - expected
            scores = float(np.max(np.abs(difference[:, 4:])))
            boxes = float(np.sqrt(np.mean(difference[:, :4] ** 2)))
            rows.append(
                {
                    "sample": key,
                    "score_max_abs_error": scores,
                    "box_rmse_pixels": boxes,
                    "passed": scores <= 0.02 and boxes <= 0.5,
                }
            )
    return {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "rows": rows,
        "scope": "Three-image numerical gate, not detection accuracy evaluation",
    }


def letterbox(bgr):
    height, width = bgr.shape[:2]
    gain = min(640.0 / height, 640.0 / width)
    new_w, new_h = round(width * gain), round(height * gain)
    dw, dh = (640 - new_w) / 2.0, (640 - new_h) / 2.0
    left, top = round(dw - 0.1), round(dh - 0.1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.copyMakeBorder(
        rgb, top, round(dh + 0.1), left, round(dw + 0.1), cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    value = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return value, (gain, left, top, width, height)


def nms(boxes, scores, threshold=0.7):
    order = np.argsort(-scores, kind="mergesort")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        others = order[1:]
        lo = np.maximum(boxes[i, :2], boxes[others, :2])
        hi = np.minimum(boxes[i, 2:], boxes[others, 2:])
        inter = np.maximum(hi - lo, 0).prod(axis=1)
        area = np.maximum(boxes[others, 2:] - boxes[others, :2], 0).prod(axis=1)
        own = np.maximum(boxes[i, 2:] - boxes[i, :2], 0).prod()
        order = others[inter / np.maximum(own + area - inter, 1e-9) <= threshold]
    return np.asarray(keep, dtype=int)


def decode(output, geometry, conf=0.25):
    if output.shape != (1, 84, 8400) or not np.isfinite(output).all():
        raise ValueError("Invalid YOLO output")
    rows = output[0].T
    classes = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), classes + 4]
    valid = (scores > conf) & np.isin(classes, list(CLASSES))
    rows, classes, scores = rows[valid], classes[valid], scores[valid]
    boxes = np.concatenate((rows[:, :2] - rows[:, 2:4] / 2, rows[:, :2] + rows[:, 2:4] / 2), axis=1)
    keep = []
    for class_id in CLASSES:
        indices = np.where(classes == class_id)[0]
        keep.extend(indices[nms(boxes[indices], scores[indices])])
    keep = np.asarray(keep, dtype=int)
    keep = keep[np.argsort(-scores[keep], kind="mergesort")[:300]]
    boxes = boxes[keep]
    gain, left, top, width, height = geometry
    boxes = (boxes - np.array([left, top, left, top])) / gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return np.column_stack((boxes, scores[keep], classes[keep])).astype(np.float32)


def annotate(frame, boxes):
    result = frame.copy()
    for x1, y1, x2, y2, confidence, class_id in boxes:
        cv2.rectangle(result, (round(x1), round(y1)), (round(x2), round(y2)), (0, 220, 0), 2)
        cv2.putText(
            result,
            "{} {:.2f}".format(CLASSES[int(class_id)], confidence),
            (round(x1), max(18, round(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
        )
    return result


class Preview:
    """Localhost-only browser preview; access remotely through an SSH tunnel."""

    def __init__(self, port, restored=False):
        self.jpeg = None
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/frame.jpg"):
                    body = preview.jpeg
                    code, mime = (200, "image/jpeg") if body else (503, "text/plain")
                    body = body or b"Waiting for a processed frame"
                elif self.path == "/":
                    code, mime = 200, "text/html"
                    body = (
                        b"<title>Aegis Nano YOLO</title><h2>YOLOv8n detector only</h2>"
                        b"<p>Latest processed frame. No Restormer. "
                        b"Refresh stops when runner exits.</p>"
                        b"<img id='view' style='max-width:100%' src='/frame.jpg' "
                        b"onload='refresh()' onerror='refresh()'>"
                        b"<script>function refresh(){setTimeout(function(){"
                        b"document.getElementById('view').src='/frame.jpg?t='+Date.now();"
                        b"},300);}</script>"
                    )
                    if restored:
                        body = body.replace(b"YOLOv8n detector only", b"Restormer + YOLOv8n")
                        body = body.replace(
                            b"No Restormer.", b"Left: input. Right: restored + detections."
                        )
                else:
                    code, mime, body = 404, "text/plain", b"Not found"
                self.send_response(code)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", port), Handler)
        self.worker = threading.Thread(target=self.server.serve_forever)
        self.worker.daemon = True
        self.worker.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()


def ros_frames(topic, limit):
    import rospy
    from sensor_msgs.msg import CompressedImage

    rospy.init_node("aegis_nano_detector", anonymous=True, disable_signals=True)
    condition = threading.Condition()
    latest = [None]

    def receive(message):
        with condition:
            latest[0] = message
            condition.notify()

    subscriber = rospy.Subscriber(topic, CompressedImage, receive, queue_size=1, buff_size=2**24)
    count = 0
    try:
        while not rospy.is_shutdown() and (limit == 0 or count < limit):
            with condition:
                if not condition.wait_for(lambda: latest[0] is not None, timeout=15):
                    raise RuntimeError("No camera messages for 15 seconds: " + topic)
                message, latest[0] = latest[0], None
            frame = cv2.imdecode(
                np.frombuffer(bytes(message.data), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                raise RuntimeError("Could not decode compressed camera image")
            yield frame, {"ros_stamp": message.header.stamp.to_sec(), "topic": topic}
            count += 1
    finally:
        subscriber.unregister()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--validate-only", action="store_true")
    sources.add_argument("--image", type=Path)
    sources.add_argument("--topic")
    parser.add_argument("--max-frames", type=int, default=300, help="0 runs until Ctrl+C")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--serve", type=int, default=0, help="Localhost browser port, e.g. 8080")
    parser.add_argument("--restorer", type=Path)
    parser.add_argument("--restorer-fixtures", type=Path)
    args = parser.parse_args()
    if not 0 < args.conf < 1 or args.max_frames < 0:
        parser.error("Require 0 < conf < 1 and max-frames >= 0")
    if args.serve and not args.topic:
        parser.error("--serve is for live --topic mode")
    if bool(args.restorer) != bool(args.restorer_fixtures):
        parser.error("Provide both --restorer and --restorer-fixtures")
    if digest(args.fixtures) != FIXTURE_SHA256:
        raise ValueError("Detector fixtures do not match the frozen release")
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    engine = Engine(args.engine)
    restorer = None
    preview = None
    frames = None
    try:
        contract = {
            "engine_sha256": digest(args.engine),
            "fixtures_sha256": FIXTURE_SHA256,
            "runner_sha256": digest(__file__),
            "tensorrt": engine.version,
        }
        cache = args.engine.with_suffix(".nano-validation.json")
        previous = json.loads(cache.read_text()) if cache.is_file() else {}
        if previous.get("passed") and all(previous.get(k) == v for k, v in contract.items()):
            print("Reusing matching detector numerical check.", flush=True)
            validation = previous
        else:
            print("Checking all three frozen detector fixtures once...", flush=True)
            validation = validate(engine, args.fixtures)
        validation.update(
            {
                "engine_sha256": digest(args.engine),
                "fixtures_sha256": FIXTURE_SHA256,
                "runner_sha256": digest(__file__),
                "tensorrt": engine.version,
            }
        )
        write_json(args.output / "validation.json", validation)
        write_json(cache, validation)
        print("Detector check passed: " + str(validation["passed"]), flush=True)
        if not validation["passed"]:
            print("REJECTED: numerical gate failed. Demo will not run.", flush=True)
            return 2
        if args.restorer:
            from nano_restormer import check_restorer, restore

            restorer = Engine(args.restorer, kind="restormer")
            report = check_restorer(restorer, args.restorer, args.restorer_fixtures)
            write_json(args.output / "restormer_validation.json", report)
            if not report["passed"]:
                print(
                    "REJECTED: Restormer check failed. See restormer_validation.json.", flush=True
                )
                return 2
        if args.validate_only:
            return 0
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise ValueError("Cannot read image: " + str(args.image))
            frames = iter([(frame, {"image": str(args.image)})])
        else:
            frames = ros_frames(args.topic, args.max_frames)
            if args.serve:
                preview = Preview(args.serve, restored=restorer is not None)
                print("Preview: http://127.0.0.1:{}/ via SSH tunnel".format(args.serve), flush=True)
        durations = []
        start = time.monotonic()
        interrupted = False
        try:
            with (args.output / "detections.jsonl").open("w") as stream:
                for count, (frame, source) in enumerate(frames, 1):
                    before = time.monotonic()
                    processed = restore(restorer, frame) if restorer else frame
                    restored_at = time.monotonic()
                    value, geometry = letterbox(processed)
                    boxes = decode(engine.run(value), geometry, args.conf)
                    processing_ms = (time.monotonic() - before) * 1000
                    durations.append(processing_ms)
                    rendered = annotate(processed, boxes)
                    if restorer:
                        rendered = np.concatenate((frame, rendered), axis=1)
                    ok, jpeg = cv2.imencode(".jpg", rendered)
                    if not ok:
                        raise RuntimeError("Cannot encode output image")
                    encoded = jpeg.tobytes()
                    (args.output / "latest.jpg").write_bytes(encoded)
                    if count == 1:
                        (args.output / "first.jpg").write_bytes(encoded)
                    if preview:
                        preview.jpeg = encoded
                    stream.write(
                        json.dumps(
                            {
                                "frame": count,
                                "source": source,
                                "boxes_xyxy_score_coco_class": boxes.tolist(),
                                "processing_ms": processing_ms,
                                "restoration_ms": (restored_at - before) * 1000,
                            }
                        )
                        + "\n"
                    )
                    if count == 1 or count % 30 == 0:
                        stream.flush()
                        print(
                            "Frame {}: {} road objects; {:.1f} ms processing".format(
                                count, len(boxes), processing_ms
                            ),
                            flush=True,
                        )
        except KeyboardInterrupt:
            interrupted = True
        elapsed = time.monotonic() - start
        write_json(
            args.output / "summary.json",
            {
                "frames": len(durations),
                "elapsed_seconds": elapsed,
                "interrupted": interrupted,
                "processed_fps_including_capture_and_output": len(durations) / max(elapsed, 1e-9),
                "median_preprocess_inference_decode_ms": float(np.median(durations))
                if durations
                else None,
                "scope": "Restormer + YOLO demo" if restorer else "YOLO-only demo",
                "timing_note": "FPS includes capture/output; not mAP or a sustained benchmark",
            },
        )
        print("Results saved in " + str(args.output), flush=True)
    finally:
        if frames is not None and hasattr(frames, "close"):
            frames.close()
        if preview:
            preview.close()
        if restorer:
            restorer.close()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
