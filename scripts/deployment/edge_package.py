#!/usr/bin/env python3
"""Create a checksummed Restormer device-test kit without training checkpoints or full datasets."""

import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from aegis_perception.edge.common import OUT, digest, environment, write_json

release = ROOT / "artifacts/releases"
release.mkdir(parents=True, exist_ok=True)
name = "aegis_restormer_edge_20261024"
destination = release / name
if destination.exists():
    raise FileExistsError(
        f"Package directory already exists: {destination}; "
        "preserve or remove this generated directory before rebuilding"
    )
destination.mkdir()


def copy(relative):
    source = ROOT / relative
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


for name in (
    "edge_build.py",
    "edge_preflight.py",
    "edge_run.py",
    "edge_device.sh",
    "edge_verify_package.py",
    "edge_summarize_device.py",
):
    copy("scripts/deployment/" + name)
for name in ("__init__.py", "common.py", "detector.py", "quality.py", "trt_runtime.py"):
    copy("src/aegis_perception/edge/" + name)
copy("src/aegis_perception/__init__.py")
for path in (
    "configs/deployment/edge_runtime_requirements.txt",
    "docs/restormer_edge_handoff.txt",
    "LICENSE",
    "data/datasets.yaml",
):
    copy(path)
for model, folder in [("restormer", "256x448/restormer"), ("yolov8n", "yolov8n")]:
    directory = OUT / "exports" / folder
    validation = json.loads((directory / "validation.json").read_text())
    if not validation["passed"]:
        raise ValueError(f"Unvalidated export: {model}")
    onnx = directory / ("restoration.onnx" if model == "restormer" else "yolov8n.onnx")
    if validation["onnx_sha256"] != digest(onnx):
        raise ValueError(f"Changed graph: {model}")
    for path in directory.iterdir():
        if path.suffix in (".onnx", ".npz", ".json"):
            copy(str(path.relative_to(ROOT)))
samples = destination / "samples"
samples.mkdir()
for weather in ("fog", "rain", "snow"):
    image = sorted((ROOT / "data/raw/dawn" / weather).rglob("*.jpg"))[0]
    shutil.copy2(image, samples / (weather + ".jpg"))
for relative in (
    "artifacts/edge_v1/dawn_annotation_audit.json",
    "artifacts/edge_v1/local_preflight.json",
    "artifacts/edge_v1/local_verification.json",
    "artifacts/edge_v1/local_smoke/benchmark.json",
    "artifacts/edge_v1/detection_pilot72/comparison.json",
    "artifacts/edge_v1/dawn_pilot72/audit.json",
    "artifacts/edge_v1/reports/restormer_edge_readiness.pdf",
    "artifacts/edge_v1/reports/restormer_edge_readiness.json",
    "artifacts/edge_v1/reports/detection_pilot72.csv",
):
    copy(relative)
notice = """THIRD PARTY MATERIALS / RESEARCH HANDOFF
Restormer architecture: MIT, Syed Waqas Zamir and contributors (2022).
https://github.com/swz30/Restormer
YOLOv8n: Ultralytics AGPL-3.0 / enterprise terms.
https://github.com/ultralytics/ultralytics/tree/v8.3.0
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
DAWN samples: https://data.mendeley.com/datasets/766ygrbt8y/3, CC BY-NC 3.0.
Validation/calibration tensors derive from the project's licensed weather datasets.
See data/datasets.yaml for source provenance. Those data licenses still apply.
Do not redistribute the image-bearing package as if it were entirely MIT code.
The unmodified upstream Restormer license follows:
\n"""
restormer_source = (ROOT / "src/aegis_perception/models/restormer.py").read_text()
notice += (
    "\n".join(
        line.removeprefix("# ").removeprefix("#") for line in restormer_source.splitlines()[:21]
    )
    + "\n"
)
(destination / "THIRD_PARTY_NOTICES.txt").write_text(notice)
shutil.copy2(ROOT / "docs/restormer_edge_handoff.txt", destination / "START_HERE.txt")
files = {
    str(p.relative_to(destination)): digest(p)
    for p in sorted(destination.rglob("*"))
    if p.is_file()
}
write_json(
    destination / "PACKAGE.json",
    {
        "name": destination.name,
        "files": files,
        "scope": (
            "Restormer + frozen YOLOv8n. TensorRT engines are NOT prebuilt. Device results pending."
        ),
        "local_environment": environment(),
    },
)
archive = release / (destination.name + ".tar.gz")
with tarfile.open(archive, "w:gz", compresslevel=3) as tar:
    for relative in ["PACKAGE.json", *files]:
        tar.add(destination / relative, arcname=destination.name + "/" + relative, recursive=False)
(release / (archive.name + ".sha256")).write_text(digest(archive) + "  " + archive.name + "\n")
print(
    json.dumps(
        {"archive": str(archive), "bytes": archive.stat().st_size, "files": len(files)}, indent=2
    )
)
