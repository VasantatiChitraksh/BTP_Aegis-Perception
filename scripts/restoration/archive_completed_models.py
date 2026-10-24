#!/usr/bin/env python3
"""Verify completed runs and create a compact, non-duplicating local model library."""

import argparse
import gc
import json
import os

import torch
from report_six_models import ROOT, collect

from aegis_perception.checkpoints import save_checkpoint
from aegis_perception.data.catalog import sha256
from aegis_perception.evaluation import summarize_samples
from aegis_perception.official_runtime import activate, build_generator

LIBRARY = ROOT / "artifacts/models"


def link(source, destination):
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Conflicting library link: {destination}")
    elif destination.exists():
        raise FileExistsError(f"Refusing to replace existing file: {destination}")
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def archive(allow_partial=False):
    torch.set_num_threads(4)
    data = collect(require_complete=not allow_partial)
    ready = {k: v for k, v in data["models"].items() if v["status"] == "completed"}
    # Unlike a metrics-only report, this library requires the evaluated weights locally.
    for model in ready.values():
        checkpoint = ROOT / model["directory"] / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Download the evaluated checkpoint: {checkpoint}")
        if sha256(checkpoint) != model["metrics"]["checkpoint_sha256"]:
            raise ValueError(f"Evaluated checkpoint hash mismatch: {checkpoint}")
        recomputed = summarize_samples(model["metrics"]["samples"])
        if recomputed["counts"] != model["metrics"]["counts"]:
            raise ValueError(f"Regression count mismatch: {model['label']}")
        for metric, value in recomputed["headline"].items():
            recorded = model["metrics"]["headline"][metric]
            matches = (
                abs(value - recorded) < 1e-9
                if isinstance(value, (int, float))
                else value == recorded
            )
            if not matches:
                raise ValueError(f"Headline mismatch: {model['label']}/{metric}")
    activate()
    LIBRARY.mkdir(parents=True, exist_ok=True)
    inventory = {
        "complete": len(ready) == 6,
        "manifest_sha256": data["manifest_sha256"],
        "models": {},
    }
    for name, model in ready.items():
        source = ROOT / model["directory"]
        destination = LIBRARY / name
        destination.mkdir(exist_ok=True)
        print(f"Verifying and indexing {model['label']}", flush=True)
        state = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
        if state["epoch"] != model["metrics"]["checkpoint_epoch"]:
            raise ValueError(f"Selected epoch mismatch: {name}")
        generator = build_generator(state["config"]["model"])
        generator.load_state_dict(state["generator"], strict=True)
        if sum(p.numel() for p in generator.parameters()) != model["parameters"]:
            raise ValueError(f"Model identity/parameter count mismatch: {name}")
        for file in (
            "best.pt",
            "last.pt",
            "history.json",
            "metrics_test.json",
            "selection_status.json",
            "run.json",
        ):
            if (source / file).is_file():
                link(source / file, destination / file)
        link(source, destination / "training_run")
        # A separate generator-only export leaves every original checkpoint untouched.
        compact = destination / "inference.pt"
        provenance = destination / "inference_provenance.json"
        digest = model["metrics"]["checkpoint_sha256"]
        existing = json.loads(provenance.read_text()) if provenance.exists() else {}
        if not (
            compact.exists()
            and existing.get("source_best_sha256") == digest
            and existing.get("sha256") == sha256(compact)
        ):
            if compact.exists() and not provenance.exists():
                raise FileExistsError(f"Unmanaged export exists: {compact}")
            save_checkpoint(
                compact,
                {
                    "generator": state["generator"],
                    "config": state["config"],
                    "epoch": state["epoch"],
                    "manifest_sha256": data["manifest_sha256"],
                    "source_best_sha256": digest,
                    "official_source_lock": "configs/restoration/official_v1/sources.json"
                    if name in ("transweather", "promptir", "ramit")
                    else None,
                },
            )
            provenance.write_text(
                json.dumps({"source_best_sha256": digest, "sha256": sha256(compact)}, indent=2)
                + "\n"
            )
        exported = torch.load(compact, map_location="cpu", weights_only=False)
        if exported["epoch"] != state["epoch"] or exported["config"] != state["config"]:
            raise ValueError(f"Compact export metadata mismatch: {name}")
        if exported["generator"].keys() != state["generator"].keys() or any(
            not torch.equal(value, exported["generator"][key])
            for key, value in state["generator"].items()
        ):
            raise ValueError(f"Compact export changed selected weights: {name}")
        del exported
        (destination / "config.json").write_text(json.dumps(state["config"], indent=2) + "\n")
        model["local_artifacts"] = {
            "library": str(destination.relative_to(ROOT)),
            "best_bytes": (source / "best.pt").stat().st_size,
            "inference_bytes": compact.stat().st_size,
            "inference_sha256": sha256(compact),
        }
        inventory["models"][name] = {
            "label": model["label"],
            "run": model["directory"],
            "best_epoch": state["epoch"],
            "best_sha256": digest,
            "parameters": model["parameters"],
            **model["local_artifacts"],
        }
        checksums = []
        for file in sorted(destination.iterdir()):
            if file.is_file() and file.name != "SHA256SUMS":
                checksums.append(f"{sha256(file)}  {file.name}\n")
        (destination / "SHA256SUMS").write_text("".join(checksums))
        del generator, state
        gc.collect()
    (LIBRARY / "index.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (LIBRARY / "README.txt").write_text(
        "AEGIS / VERIFIED MODEL LIBRARY\n\n"
        f"Locally verified: {len(ready)}/6 completed benchmark models.\n"
        "best.pt: original validation-selected checkpoint, linked to its training run.\n"
        "last.pt: last resumable training state, where locally available.\n"
        "inference.pt: generator + config + selected epoch; optimizer/discriminator omitted.\n"
        "training_run/: complete original run, including downloaded periodic snapshots.\n"
        "metrics_test.json / history.json: evaluated scores and all 60 training epochs.\n"
        "config.json / SHA256SUMS: exact model configuration and local integrity records.\n\n"
        "Links save disk space. To copy a model folder elsewhere, use cp -L or rsync -L;\n"
        "omit training_run and last.pt for a small inference-only handoff.\n"
        "Weights also require this project's model code. Call official_runtime.activate()\n"
        "before load_generator for all six architectures. Official models additionally\n"
        "require artifacts/official_sources and configs/restoration/official_v1/sources.json.\n"
        "inference.pt is not an ONNX or TensorRT engine and has not been quantized.\n"
        "\nLoading example (run with this repository's src on PYTHONPATH):\n"
        "  from aegis_perception.official_runtime import activate\n"
        "  from aegis_perception.checkpoints import load_generator\n"
        "  activate()\n"
        "  model, metadata = load_generator(\n"
        "      'artifacts/models/restormer/inference.pt', device='cpu')\n"
        "Inputs: NCHW RGB tensors normalized to [-1, 1]; use the project's predict()\n"
        "helper for native shapes/padding. Keep input/output normalization unchanged.\n\n"
        "The original older Restormer is retained outside this six-model comparison.\n"
    )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Index available models without claiming six-model completion",
    )
    args = parser.parse_args()
    data = archive(args.allow_partial)
    (LIBRARY / "verified_results.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"Library: {LIBRARY}")


if __name__ == "__main__":
    main()
