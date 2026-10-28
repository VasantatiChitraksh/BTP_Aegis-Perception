#!/usr/bin/env python3
"""rover_infer_trt.py — Real-time TensorRT inference for JetAuto Pro & Jetson Orin Nano.

This script runs the de-weathering engine in real-time with:
- Zero PyTorch overhead (uses PyCUDA / TensorRT C-bindings, <150MB RAM footprint)
- Direct support for live camera streams (3D Depth Camera / USB / CSI camera)
- Real-time FPS and latency monitor suitable for ROS robot car control loops.

Usage on Jetson:
    # Single image test
    python quantization/scripts/rover_infer_trt.py \
        --engine artifacts/tensorrt/liteweatherformer_fp16.engine \
        --image data/raw/rain/realrain1k/rain/0001.jpg \
        --output artifacts/rover_test_output.jpg

    # Live camera test on JetAuto Pro
    python quantization/scripts/rover_infer_trt.py \
        --engine artifacts/tensorrt/liteweatherformer_fp16.engine \
        --camera 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time TensorRT inference on Jetson / Rover")
    parser.add_argument("--engine", type=Path, required=True, help="Path to .engine or .trt file")
    parser.add_argument("--image", type=Path, default=None, help="Path to static input image")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index (e.g. 0)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/rover_restored.jpg"))
    parser.add_argument("--input-size", type=int, nargs=2, default=(256, 256), help="H W model size")
    return parser.parse_args()


class TRTInferenceWrapper:
    def __init__(self, engine_path: Path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate host and device buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def infer(self, image_np: np.ndarray) -> np.ndarray:
        import pycuda.driver as cuda

        # Copy input to pagelocked host buffer and transfer to device
        np.copyto(self.inputs[0]["host"], image_np.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        
        # Execute context
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output back from device to host
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        return self.outputs[0]["host"]


def preprocess_cv2(img_bgr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    import cv2
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    # Normalize to [-1, 1] as expected by the generator models
    norm = (resized.astype(np.float32) / 255.0) * 2.0 - 1.0
    # Transpose to (1, C, H, W)
    chw = np.transpose(norm, (2, 0, 1))
    return np.ascontiguousarray(np.expand_dims(chw, axis=0))


def postprocess_cv2(output_flat: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    import cv2
    # Reshape from flat buffer to (3, H, W)
    chw = output_flat.reshape((3, target_h, target_w))
    hwc = np.transpose(chw, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 255]
    img = np.clip((hwc + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def run_image(wrapper: TRTInferenceWrapper, args: argparse.Namespace) -> None:
    import cv2
    img_bgr = cv2.imread(str(args.image))
    if img_bgr is None:
        print(f"[ERROR] Could not read image {args.image}")
        return

    h, w = args.input_size
    inp = preprocess_cv2(img_bgr, h, w)
    
    # Warm-up
    for _ in range(5):
        wrapper.infer(inp)

    # Timed run
    start = time.perf_counter()
    out = wrapper.infer(inp)
    dur_ms = (time.perf_counter() - start) * 1000.0

    restored_bgr = postprocess_cv2(out, h, w)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), restored_bgr)
    print(f"[SUCCESS] Inference Latency: {dur_ms:.2f} ms ({1000.0 / dur_ms:.1f} FPS)")
    print(f"[SUCCESS] Saved restored image to {args.output}")


def run_camera(wrapper: TRTInferenceWrapper, args: argparse.Namespace) -> None:
    import cv2
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return

    h, w = args.input_size
    print(f"[INFO] Streaming from camera {args.camera}. Press 'q' to exit.")
    
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        inp = preprocess_cv2(frame, h, w)
        out = wrapper.infer(inp)
        restored = postprocess_cv2(out, h, w)

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        cv2.putText(
            restored, f"JetAuto TRT FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.imshow("JetAuto Perception - Restored", restored)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    try:
        wrapper = TRTInferenceWrapper(args.engine)
    except Exception as exc:
        print(f"[ERROR] Failed to initialize TensorRT: {exc}")
        print("Note: PyCUDA and TensorRT are required on Jetson.")
        sys.exit(1)

    if args.image:
        run_image(wrapper, args)
    elif args.camera is not None:
        run_camera(wrapper, args)
    else:
        print("Specify either --image or --camera.")


if __name__ == "__main__":
    main()
