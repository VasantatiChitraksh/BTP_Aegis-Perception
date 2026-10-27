#!/usr/bin/env python3
"""Package the exact six report inference checkpoints, with provenance and notices."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/releases/aegis_six_models_20261024.tar.gz",
    )
    args = parser.parse_args()
    index = json.loads((ROOT / "artifacts/models/index.json").read_text())
    files = ["artifacts/models/index.json", "artifacts/models/README.txt"]
    for name, entry in index["models"].items():
        folder = entry["library"]
        if digest(ROOT / folder / "inference.pt") != entry["inference_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
        files.extend(f"{folder}/{filename}" for filename in (
            "inference.pt", "config.json", "inference_provenance.json",
        ))
    files.extend([
        "configs/restoration/official_v1/sources.json",
        "docs/official_model_training.txt",
        "artifacts/reports/six_models_20261024/aegis_six_model_benchmark.json",
    ])
    notices = {}
    lock = json.loads((ROOT / "configs/restoration/official_v1/sources.json").read_text())
    for name, entry in lock["models"].items():
        for source in entry["sources"]:
            path = Path(source["path"])
            if path.name.lower().startswith(("readme", "license")):
                if digest(ROOT / path) != source["sha256"]:
                    raise ValueError(f"Notice checksum mismatch: {path}")
                notices[f"artifacts/models/notices/{name}/{path.name}"] = ROOT / path
    # Explicit paths only: never archive training_run links, optimizer state or raw datasets.
    members = {relative: ROOT / relative for relative in files} | notices
    checksums = {relative: digest(path) for relative, path in members.items()}
    notice = (
        "AEGIS SIX-MODEL RESEARCH CHECKPOINT HANDOFF\n"
        "Frozen 24 October 2026 comparison; Restormer selected epoch 51.\n"
        "These are FP32 generator checkpoints, not quantized engines or resume states.\n"
        "Use the matching repository and docs/quantization_handoff.txt.\n"
        "Upstream architecture code is fetched separately with the pinned source manifest.\n"
        "Original model terms continue to apply; this archive does not relicense weights.\n"
        "PromptIR: academic/noncommercial license; preserve its included notice.\n"
        "TransWeather: included MIT notice. RAMiT: no standalone license in pinned source;\n"
        "its original README/citation is included, with no additional rights asserted.\n"
        "Unequal pretraining and possible pretraining overlap limit benchmark comparisons.\n"
        "No Jetson speed or quantized-quality result is established by this archive.\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checksum_path = args.output.with_name(args.output.name + ".sha256")
    if args.output.exists() or checksum_path.exists():
        raise FileExistsError("Choose a new output path; existing releases are preserved")
    with tarfile.open(args.output, "x:gz", compresslevel=6) as archive:
        for relative, path in members.items():
            archive.add(path, arcname=relative, recursive=False)
        for relative, content in (
            ("artifacts/models/RELEASE_SHA256.json", json.dumps(checksums, indent=2) + "\n"),
            ("artifacts/models/THIRD_PARTY_NOTICES.txt", notice),
        ):
            data = content.encode()
            member = tarfile.TarInfo(relative)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    checksum_path.write_text(f"{digest(args.output)}  {args.output.name}\n")
    print(f"Packaged {len(index['models'])} verified models: {args.output}")


if __name__ == "__main__":
    main()
