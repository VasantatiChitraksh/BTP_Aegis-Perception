#!/usr/bin/env python3
"""Record target capabilities; never install drivers or change power settings."""

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from aegis_perception.edge.common import environment, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("results/preflight.json"))
parser.add_argument(
    "--local", action="store_true", help="CPU development check; does not certify Jetson"
)
args = parser.parse_args()
record = {"environment": environment(), "errors": [], "local_only": args.local}
for package in ("numpy", "PIL", "cv2") + (() if args.local else ("tensorrt",)):
    try:
        importlib.import_module(package)
    except ImportError as error:
        record["errors"].append(str(error))
record["memory_info"] = (
    Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else None
)
if not args.local and not record["errors"]:
    try:
        import tensorrt as trt

        from aegis_perception.edge.trt_runtime import Cuda

        record["cuda_memory"] = Cuda().memory()
        version = tuple(map(int, trt.__version__.split(".")[:2]))
        if not ((version[0] == 8 and version >= (8, 5)) or version[0] == 10):
            record["errors"].append(
                "Supported TensorRT API range is 8.5+ or 10.x. JetPack must supply "
                "matching libraries."
            )
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        record["fast_fp16"] = bool(builder.platform_has_fast_fp16)
        record["fast_int8"] = bool(builder.platform_has_fast_int8)
    except Exception as error:
        record["errors"].append(str(error))
record["tools"] = {
    name: shutil.which(name) for name in ("tegrastats", "nvpmodel", "jetson_clocks", "trtexec")
}
if record["tools"]["nvpmodel"]:
    query = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=15)
    record["power_mode_query"] = query.stdout + query.stderr
record["passed"] = not record["errors"]
write_json(args.output, record)
print(json.dumps(record, indent=2))
sys.exit(0 if record["passed"] else 1)
