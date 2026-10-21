#!/usr/bin/env python3
"""Frozen, sequential attention Pix2Pix and Restormer comparison."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aegis_perception.config import load_config, validate_restoration_config
from aegis_perception.data.catalog import sha256
from aegis_perception.data.manifest import load_manifest
from aegis_perception.data.protocol import verify_protocol
from aegis_perception.reproducibility import write_json

NAMES = ("attention_pix2pix", "restormer")
CONFIGS = tuple(Path(f"configs/restoration/baselines_v1/{n}.yaml") for n in NAMES)
OUTPUT = Path("artifacts/restoration/baselines_v1")
BOUNDARIES = (20, 40, 60)
STAGE_SCRIPT = "scripts/restoration/train_baseline_stage.sbatch"
TRAIN_COMMAND = (sys.executable, "scripts/restoration/train.py")
EVALUATE_COMMAND = (sys.executable, "scripts/restoration/evaluate.py")
LOG_PREFIX = "aegis-base"


def checked_configs():
    configs = [load_config(p) for p in CONFIGS]
    for c in configs:
        validate_restoration_config(c)
        verify_protocol(c["data"])
        if c["train"].get("validation_guard"):
            raise ValueError("Use the full training recipe, not the earlier guarded fine-tune")
        if c["model"]["name"] == "restormer":
            if not c["model"].get("pretrained") or not c["train"].get("expected_pretrained_sha256"):
                raise ValueError("Restormer must start from verified official pretrained weights")
        if c["train"]["epochs"] != BOUNDARIES[-1]:
            raise ValueError("Suite stages and epoch budget differ")
        if c["data"] != configs[0]["data"]:
            raise ValueError("All models must use identical data and sampling")
        if c["run"]["seed"] != configs[0]["run"]["seed"]:
            raise ValueError("All models must use the same declared sampling seed")
        if c["train"]["batch_size"] != configs[0]["train"]["batch_size"]:
            raise ValueError("All models must use the same global batch size")
    return configs


def ensure_initializations(configs):
    """Reuse the existing official weights; fail before queuing if unavailable or changed."""
    for c in configs:
        if not c["model"].get("pretrained"):
            continue
        path = Path(c["model"]["pretrained"])
        expected = c["train"]["expected_pretrained_sha256"]
        if not path.exists():
            source = (
                Path(
                    os.environ.get(
                        "AEGIS_PRETRAINED_DIR",
                        f"/scratch/{os.environ.get('USER', '')}/"
                        "aegis-four-weather/artifacts/pretrained",
                    )
                )
                / path.name
            )
            if not source.is_file() or sha256(source) != expected:
                raise ValueError(f"Verified official pretrained weights required: {source}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(source.resolve())
        if sha256(path) != expected:
            raise ValueError(f"Official initialization checksum mismatch: {path}")


def verify_release():
    release = ROOT / "baseline_release.json"
    if not release.exists():
        return  # Development workspace; packaged deployments always carry this file.
    for name, digest in json.loads(release.read_text())["files"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Packaged source/config was changed: {name}")


def package(output, file_names=None):
    checked_configs()
    if output.exists():
        raise FileExistsError(output)
    files = set()
    for folder in ("src",):
        files.update(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    files.update(ROOT / p for p in CONFIGS)
    if file_names is None:
        file_names = (
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "docs/baseline_suite.txt",
            "artifacts/data/baselines_v1/preparation.json",
            "artifacts/data/baselines_v1/dataset_audit.txt",
            "tests/conftest.py",
            "tests/test_baseline_suite.py",
            "tests/test_training.py",
            "configs/restoration/four_weather_restormer.yaml",
            "scripts/restoration/baseline_suite.py",
            "scripts/restoration/train.py",
            "scripts/restoration/evaluate.py",
            "scripts/restoration/compare_weather_protocol.py",
            "scripts/restoration/train_baseline_stage.sbatch",
            "scripts/restoration/submit_baseline_suite.sh",
            "scripts/data/audit_manifest.py",
            "data/manifests/weather_v3/active.csv",
            "data/manifests/weather_v3/reserve.csv",
            "data/manifests/weather_v3/protocol.json",
            "data/manifests/clean/four_weather.csv",
            "artifacts/restoration/four_weather_restormer_seed42/metrics_test.json",
        )
    for name in file_names:
        files.add(ROOT / name)
    release = {
        "suite": "baselines-v1",
        "models": list(NAMES),
        "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep provenance in the bundle, not in the mutable development checkout.
    import io

    partial = output.with_suffix(output.suffix + ".partial")
    with tarfile.open(partial, "w:gz") as archive:
        for p in sorted(files):
            archive.add(p, arcname=p.relative_to(ROOT), recursive=False)
        blob = (json.dumps(release, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo("baseline_release.json")
        info.size, info.mode = len(blob), 0o644
        archive.addfile(info, io.BytesIO(blob))
    partial.replace(output)
    output.with_name(output.name + ".sha256").write_text(f"{sha256(output)}  {output.name}\n")
    print(f"Packaged {len(files)} files: {output} ({output.stat().st_size / 2**20:.2f} MiB)")


def submit(resume=False):
    configs = checked_configs()
    verify_release()
    ensure_initializations(configs)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    journal = OUTPUT / "jobs.json"
    state = json.loads(journal.read_text()) if journal.exists() else {"attempts": []}
    if state["attempts"]:
        if not resume:
            raise ValueError(
                "Suite already submitted. Use --resume only after its jobs have ended."
            )
        active = set(
            subprocess.check_output(
                ["squeue", "--noheader", f"--user={os.environ['USER']}", "--format=%A"],
                text=True,
            ).split()
        )
        previous = {j["job_id"] for a in state["attempts"] for j in a}
        if active & previous:
            raise ValueError(f"Suite jobs still active: {sorted(active & previous)}")
    elif any((Path(c["run"]["output_dir"]) / "last.pt").exists() for c in configs):
        if not resume:
            raise ValueError("Existing checkpoints found; explicitly submit with --resume")
    jobs = []
    state["attempts"].append(jobs)
    previous = None
    try:
        for config, name in zip(CONFIGS, NAMES, strict=True):
            for boundary in BOUNDARIES:
                command = [
                    "sbatch",
                    "--parsable",
                    "--kill-on-invalid-dep=yes",
                    f"--job-name=ag-{name[:8]}-{boundary}",
                ]
                if previous:
                    command.append(f"--dependency=afterok:{previous}")
                command += [
                    STAGE_SCRIPT,
                    str(config),
                    str(boundary),
                ]
                raw = subprocess.check_output(command, text=True).strip()
                job_id = raw.split(";")[0]
                if not job_id.isdigit():
                    raise ValueError(f"Unexpected sbatch response: {raw!r}")
                jobs.append(dict(job_id=job_id, model=name, through_epoch=boundary))
                write_json(journal, state)  # Preserve IDs even if a later submission fails.
                print(f"{name:20} through epoch {boundary:2}: job {job_id}", flush=True)
                previous = job_id
    except Exception:
        print(
            f"Submission stopped; already submitted jobs are recorded in {journal}", file=sys.stderr
        )
        raise
    print(f"Submitted {len(NAMES)} model(s) sequentially (three resumable stages each).")
    print(f"Logs: /scratch/{os.environ['USER']}/{LOG_PREFIX}-<JOBID>.out / .err")
    print(f"Results: {OUTPUT}/comparison.csv; journal: {journal}")


def preflight(configs):
    """On the allocated GPU: decode/leakage audit plus worst-shape inference checks."""
    import torch
    from PIL import Image

    from aegis_perception.inference import predict
    from aegis_perception.models import build_generator

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Suite requires exactly one allocated GPU with BF16 support")
    stamp = OUTPUT / "preflight.json"
    if stamp.exists():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "dataset_audit.txt").open("w") as log:
        subprocess.run(
            [sys.executable, "scripts/data/audit_manifest.py", configs[0]["data"]["manifest"]],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    records = load_manifest(configs[0]["data"]["manifest"])
    shapes = set()
    for record in records:
        if record.split != "train":
            with Image.open(record.input_path) as im:
                shapes.add((im.height, im.width))
    # Check the three largest padded shapes per architecture; no test scores/targets.
    checks = []
    for c in configs:
        model = build_generator(c["model"]).cuda().eval()
        multiple = getattr(model, "spatial_multiple", 64)
        biggest = sorted(
            shapes,
            key=lambda s: ((s[0] + multiple - 1) // multiple) * ((s[1] + multiple - 1) // multiple),
            reverse=True,
        )[:3]
        for h, w in biggest:
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                sample = torch.zeros(1, 3, h, w, device="cuda")
                prediction = predict(model, sample, precision="bf16")
                if prediction.shape != sample.shape or not torch.isfinite(prediction).all():
                    raise RuntimeError("Native-resolution preflight failed")
            checks.append(
                dict(
                    model=c["run"]["name"],
                    height=h,
                    width=w,
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                )
            )
            del sample, prediction
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        stamp,
        {
            "gpu": torch.cuda.get_device_name(0),
            "checks": checks,
            "note": "Shape checks only, no test metrics or model selection.",
        },
    )


def run_stage(config_path, through_epoch):
    import torch

    verify_release()
    configs = checked_configs()
    ensure_initializations(configs)
    if config_path not in CONFIGS or through_epoch not in BOUNDARIES:
        raise ValueError("Unknown suite config or stage")
    config = configs[CONFIGS.index(config_path)]
    preflight(configs)
    directory = Path(config["run"]["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    last = directory / "last.pt"
    completed = 0
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        for key in ("model", "data", "train", "run"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError("Existing checkpoint belongs to a different experiment")
        completed = int(checkpoint["epoch"])
        del checkpoint
        gc.collect()
    if completed < through_epoch:
        command = [
            *TRAIN_COMMAND,
            "--config",
            str(config_path),
            "--stop-after-epochs",
            str(through_epoch - completed),
        ]
        if last.exists():
            command += ["--resume", str(last)]
        subprocess.run(command, check=True)
    else:
        print(f"Already completed epoch {completed}; stage {through_epoch} skipped", flush=True)
    if through_epoch == config["train"]["epochs"]:
        best, metrics = directory / "best.pt", directory / "metrics_test.json"
        if not best.exists():
            raise RuntimeError("Completed baseline without a best checkpoint")
        previous = json.loads(metrics.read_text()) if metrics.exists() else {}
        if previous.get("checkpoint_sha256") != sha256(best):
            subprocess.run(
                [
                    *EVALUATE_COMMAND,
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(best),
                    "--split",
                    "test",
                    "--device",
                    "cuda",
                ],
                check=True,
            )
        report()
        if config["model"]["name"] == "restormer":
            compare_previous(metrics, directory / "comparison_previous.json")


def compare_previous(metrics, output):
    # Reuse saved original predictions on EXACTLY the new test IDs and metric protocol.
    import runpy

    compare = runpy.run_path(str(ROOT / "scripts/restoration/compare_weather_protocol.py"))[
        "compare"
    ]
    result = compare(
        Path("artifacts/restoration/four_weather_restormer_seed42/metrics_test.json"),
        Path("data/manifests/clean/four_weather.csv"),
        Path("data/manifests/weather_v3/active.csv"),
        metrics,
    )
    regressions = []
    for weather in ("macro_weather", "rain", "fog", "snow", "smoke"):
        old = result["baseline"]["scene_balanced_summary"][weather]
        new = result["candidate"]["scene_balanced_summary"][weather]
        for metric in ("psnr_db", "ssim"):
            if new[metric] < old[metric]:
                regressions.append(
                    dict(weather=weather, metric=metric, delta=new[metric] - old[metric])
                )
    result["assessment"] = {
        "matches_or_exceeds_previous_on_all_reported_metrics": not regressions,
        "regressions": regressions,
        "note": "Observed benchmark comparison, not a guarantee or a checkpoint-selection step. "
        "The validation-selected candidate and original model are both preserved. "
        "No old checkpoint is overwritten or automatically replaced.",
    }
    write_json(output, result)
    print(f"Comparison with previous successful Restormer: {output}", flush=True)
    print(result["assessment"], flush=True)


def report(configs=None, names=None):
    rows = []
    for name, config in zip(
        NAMES if names is None else names,
        checked_configs() if configs is None else configs,
        strict=True,
    ):
        path = Path(config["run"]["output_dir"]) / "metrics_test.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if result["manifest_sha256"] != sha256(Path(config["data"]["manifest"])):
            raise ValueError("Cannot compare results on different manifests")
        for weather in ("macro_weather", "rain", "fog", "snow", "smoke"):
            metrics = result["scene_balanced_summary"][weather]
            counts = result["counts"].get(weather, {})
            rows.append(
                dict(
                    model=name,
                    weather=weather,
                    checkpoint_epoch=result["checkpoint_epoch"],
                    psnr_db=metrics["psnr_db"],
                    ssim=metrics["ssim"],
                    psnr_gain_db=metrics["psnr_gain_db"],
                    ssim_gain=metrics["ssim_gain"],
                    images=counts.get(
                        "images", sum(c["images"] for c in result["counts"].values())
                    ),
                    psnr_worse_than_raw=counts.get("psnr_worse_than_raw", ""),
                    ssim_worse_than_raw=counts.get("ssim_worse_than_raw", ""),
                )
            )
    if rows:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        destination = OUTPUT / "comparison.csv"
        temporary = destination.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
        print(f"Updated {destination}", flush=True)


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("package")
    pack.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("submit")
    sub.add_argument("--resume", action="store_true")
    stage = commands.add_parser("stage")
    stage.add_argument("config", type=Path)
    stage.add_argument("through_epoch", type=int)
    commands.add_parser("report")
    args = parser.parse_args()
    if args.command == "package":
        package(args.output)
    elif args.command == "submit":
        submit(args.resume)
    elif args.command == "stage":
        run_stage(args.config, args.through_epoch)
    else:
        report()


if __name__ == "__main__":
    main()
