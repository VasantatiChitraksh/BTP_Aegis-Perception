#!/usr/bin/env python3
"""Export matched, score-blind native-resolution examples from six evaluated checkpoints."""

import argparse
import gc
import json
import shutil

import torch
from PIL import Image
from report_six_models import ROOT, collect

from aegis_perception.checkpoints import load_generator
from aegis_perception.data.catalog import sha256
from aegis_perception.data.paired import normalized_tensor_to_pil, pil_to_normalized_tensor
from aegis_perception.inference import predict
from aegis_perception.official_runtime import activate

OUT = ROOT / "artifacts/reports/six_models_20261024/examples"
OLD = ROOT / "artifacts/reports/three_models_20261021"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    data = collect(require_complete=not args.allow_partial)
    old = json.loads((OLD / "verified_report_data.json").read_text())["audit"]
    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "audit.json"
    cache = json.loads(audit_path.read_text()) if audit_path.exists() else {"models": {}}
    audit = {
        "selection": old["selection"],
        "examples": old["examples"],
        "precision": "CPU FP32 previews; benchmark tables remain DGX BF16",
        "manifest_sha256": data["manifest_sha256"],
        "models": {},
    }
    torch.set_num_threads(4)
    activate()
    for example in old["examples"]:
        weather = example["weather"]
        # Verify cached source previews against the actual frozen test pair.
        for kind in ("input", "target"):
            with (
                Image.open(example[f"source_{kind}"]) as actual,
                Image.open(OLD / f"{weather}_{kind}.png") as cached,
            ):
                if actual.convert("RGB").tobytes() != cached.convert("RGB").tobytes():
                    raise ValueError(f"Source preview mismatch: {weather}/{kind}")
            shutil.copy2(OLD / f"{weather}_{kind}.png", OUT / f"{weather}_{kind}.png")
    aliases = {"attention_pix2pix": "gan", "restormer": "restormer", "liteweatherformer": "lite"}
    for name, entry in data["models"].items():
        if entry["status"] != "completed":
            continue
        digest = entry["metrics"]["checkpoint_sha256"]
        saved = cache["models"].get(name, {})
        files = [OUT / f"{e['weather']}_{name}.png" for e in old["examples"]]
        if saved.get("checkpoint_sha256") == digest and all(
            f.exists() and saved.get("files", {}).get(f.name) == sha256(f) for f in files
        ):
            audit["models"][name] = saved
            continue
        print(f"Native CPU examples: {entry['label']}", flush=True)
        alias = aliases.get(name)
        if alias:
            if old["models"][alias]["checkpoint_sha256"] != digest:
                raise ValueError(f"Archived preview came from different weights: {name}")
            for example, file in zip(old["examples"], files, strict=True):
                shutil.copy2(OLD / f"{example['weather']}_{alias}.png", file)
        else:
            checkpoint = ROOT / entry["directory"] / "best.pt"
            if not checkpoint.is_file() or sha256(checkpoint) != digest:
                raise ValueError(f"Missing or mismatched weights: {name}")
            model, state = load_generator(checkpoint, device="cpu")
            del state
            for example, file in zip(old["examples"], files, strict=True):
                with Image.open(OUT / f"{example['weather']}_input.png") as image:
                    x = pil_to_normalized_tensor(image.convert("RGB")).unsqueeze(0)
                with torch.inference_mode():
                    restored = predict(model, x)[0]
                if restored.shape != x[0].shape or not torch.isfinite(restored).all():
                    raise ValueError(f"Invalid native output: {name}/{example['weather']}")
                normalized_tensor_to_pil(restored).save(file)
                print(f"  {example['weather']} exported", flush=True)
                del x, restored
            del model
            gc.collect()
        audit["models"][name] = {
            "checkpoint_sha256": digest,
            "files": {f.name: sha256(f) for f in files},
        }
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Matched examples for {len(audit['models'])}/6 models: {OUT}")


if __name__ == "__main__":
    main()
