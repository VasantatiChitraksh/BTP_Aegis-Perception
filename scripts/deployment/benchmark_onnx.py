#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ONNX batch-1 model latency")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=4, default=(1, 3, 256, 256))
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.model), providers=[args.provider])
    input_name = session.get_inputs()[0].name
    sample = np.random.default_rng(42).uniform(-1, 1, size=args.shape).astype(np.float32)
    for _ in range(args.warmup):
        session.run(None, {input_name: sample})
    latencies = []
    for _ in range(args.runs):
        started = time.perf_counter_ns()
        session.run(None, {input_name: sample})
        latencies.append((time.perf_counter_ns() - started) / 1e6)
    values = np.asarray(latencies)
    p50 = float(np.percentile(values, 50))
    p95 = float(np.percentile(values, 95))
    print(
        {
            "provider": args.provider,
            "shape": args.shape,
            "warmup": args.warmup,
            "runs": args.runs,
            "latency_ms_p50": p50,
            "latency_ms_p95": p95,
            "fps_from_p50": 1000.0 / p50,
        }
    )


if __name__ == "__main__":
    main()
