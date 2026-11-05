"""Refresh manuscript evidence from original local artifacts; never modifies them."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path

import yaml

PACK = Path(__file__).resolve().parent
ROOT = PACK.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Evidence review date, YYYY-MM-DD")
    args = parser.parse_args()
    provenance = {
        "as_of": args.date,
        "source_repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scope": "Manuscript evidence snapshot, not a full training or deployment package",
        "sources": {},
        "outputs": {},
    }

    def source(name):
        path = ROOT / name
        provenance["sources"][name] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return path

    def read(name):
        return json.loads(source(name).read_text())

    def write(name, obj, sources, derivation):
        path = PACK / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        provenance["outputs"][name] = {"sources": sources, "derivation": derivation}

    def copy(src, dst):
        path = source(src)
        out = PACK / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out)
        provenance["outputs"][dst] = {"sources": [src], "derivation": "Unchanged byte copy"}
        if dst.endswith(".csv"):
            out.write_text(path.read_text())
            provenance["outputs"][dst]["derivation"] = "CSV copy with LF newlines; values unchanged"
        if dst.endswith(".pdf"):
            text = out.with_suffix(".txt")
            subprocess.run(["pdftotext", "-layout", str(out), str(text)], check=True)
            extracted = "\n".join(line.rstrip(" \t\r") for line in text.read_text().split("\n"))
            text.write_text(extracted.rstrip() + "\n")
            provenance["outputs"][text.relative_to(PACK).as_posix()] = {
                "sources": [src],
                "derivation": "pdftotext -layout; trailing whitespace trimmed; PDF authoritative",
            }

    report_dir = "artifacts/reports/six_models_20261024"
    edge_dir = "artifacts/edge_v1/reports"
    joint_dir = "artifacts/restoration/joint_v3_dgx"
    rain_dir = "artifacts/restoration/rain_restormer_seed42"
    quant_dir = "docs/quantization"
    custom_source = f"{quant_dir}/restormer_custom_screen_20261030.json"
    copies = {
        f"{report_dir}/aegis_six_model_benchmark.pdf": "reports/six_model_benchmark.pdf",
        f"{edge_dir}/restormer_edge_readiness.pdf": "reports/restormer_edge_readiness.pdf",
        f"{joint_dir}/joint_research_report.pdf": "reports/joint_research_report.pdf",
        f"{rain_dir}/rain_training_report.pdf": "reports/rain_baseline_historical.pdf",
        f"{report_dir}/aegis_six_model_benchmark.csv": "tables/six_models.csv",
        f"{edge_dir}/detection_pilot72.csv": "tables/detection_pilot72.csv",
        f"{joint_dir}/comparison.csv": "tables/joint_test.csv",
        f"{joint_dir}/validation_comparison.csv": "tables/joint_validation.csv",
        f"{edge_dir}/restormer_compression_screen.csv": "tables/pruning_screen.csv",
        "artifacts/data/research_joint_v3/audit.json": "evidence/data_audit.json",
        "data/manifests/weather_v3/protocol.json": "evidence/data_protocol.json",
        f"{edge_dir}/restormer_edge_readiness.json": "evidence/edge_readiness.json",
        custom_source: "evidence/custom_quantization_screen.json",
    }
    for src, dst in copies.items():
        copy(src, dst)

    # Keep source hashes for the exact already-tracked pair lists and later device notes.
    for name in [
        "data/manifests/weather_v3/active.csv",
        "data/manifests/weather_v3/reserve.csv",
        "docs/nano_yolo_demo.txt",
        "configs/restoration/official_v1/sources.json",
    ]:
        source(name)

    protocol = read("data/manifests/weather_v3/protocol.json")
    with (PACK / "tables/dataset_split.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["weather", "split", "pairs", "backgrounds", "clusters"])
        for weather, splits in sorted(protocol["counts"].items()):
            for split in ["train", "val", "test"]:
                d = splits[split]
                writer.writerow([weather, split, d["images"], d["backgrounds"], d["clusters"]])
    provenance["outputs"]["tables/dataset_split.csv"] = {
        "sources": ["data/manifests/weather_v3/protocol.json"],
        "derivation": "Flatten counts object",
    }

    catalog = yaml.safe_load(source("data/datasets.yaml").read_text())
    ids = {
        "realrain1k",
        "reside6k_train",
        "reside6k_test",
        "snow_cityscapes",
        "smokebench",
        "dawn_fog",
        "dawn_rain",
        "dawn_snow",
    }
    write(
        "evidence/dataset_sources.json",
        {
            "note": "Catalog metadata; usage and provenance caveats are in DATASET.md",
            "datasets": [d for d in catalog["datasets"] if d["id"] in ids],
        },
        ["data/datasets.yaml"],
        "Select only source catalog entries used in restoration or DAWN pilot",
    )

    index_path = "artifacts/models/index.json"
    index = read(index_path)
    six = {"manifest_sha256": index["manifest_sha256"], "models": {}}
    sources = [index_path]
    image_rows = []
    for name, identity in index["models"].items():
        run = identity["run"]
        paths = {
            k: f"{run}/{filename}"
            for k, filename in {
                "run": "run.json",
                "metrics": "metrics_test.json",
                "history": "history.json",
            }.items()
        }
        records = {k: read(p) for k, p in paths.items()}
        image_rows.extend({"model": name, **row} for row in records["metrics"].pop("samples"))
        six["models"][name] = {
            "label": identity["label"],
            "model_library_identity": identity,
            **records,
        }
        sources.extend(paths.values())
    write(
        "evidence/six_models.json",
        six,
        sources,
        "Combine exact model index/run/history/metrics; samples moved losslessly to compressed CSV",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=["model"] + sorted(set(image_rows[0]) - {"model"}), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(image_rows)
    (PACK / "evidence/per_image_restoration.csv.gz").write_bytes(
        gzip.compress(buffer.getvalue().encode(), mtime=0)
    )
    provenance["outputs"]["evidence/per_image_restoration.csv.gz"] = {
        "sources": [s for s in sources if s.endswith("metrics_test.json")],
        "derivation": "All six samples lists; add model key; CSV + deterministic gzip; no images",
    }

    joint_dir = "artifacts/restoration/joint_v3_dgx"
    joint_paths = [
        f"{joint_dir}/{s}" for s in ["selection_before_test.json", "calibration.json", "jobs.json"]
    ]
    joint = {Path(p).stem: read(p) for p in joint_paths}
    joint["arms"] = {}
    for name in [
        "control",
        "spatial",
        "detail",
        "combined",
        "recipe_only",
        "combined_ssim",
        "combined_ffl",
        "combined_recipe",
    ]:
        fields = {}
        for key, path in {
            "run": f"{joint_dir}/{name}_seed42/run.json",
            "history": f"{joint_dir}/{name}_seed42/history.json",
            "selection_status": f"{joint_dir}/{name}_seed42/selection_status.json",
            "test_metrics": f"{joint_dir}/test/{name}.json",
        }.items():
            fields[key] = read(path)
            if key == "test_metrics":
                fields[key].pop("samples")
            joint_paths.append(path)
        joint["arms"][name] = fields
    original = f"{joint_dir}/test/original.json"
    joint["original_test_metrics"] = read(original)
    joint["original_test_metrics"].pop("samples")
    joint_paths.append(original)
    write(
        "evidence/joint_experiments.json",
        joint,
        joint_paths,
        "Combine run/history/selection/calibration/jobs/test metrics; omit per-image samples",
    )

    old_dir = "artifacts/restoration/rain_restormer_seed42"
    old = {}
    old_paths = []
    for name in ["run", "metrics_test", "metrics_test_pretrained", "history"]:
        path = f"{old_dir}/{name}.json"
        old[name] = read(path)
        if isinstance(old[name], dict):
            old[name].pop("samples", None)
        old_paths.append(path)
    write(
        "evidence/rain_baseline_historical.json",
        old,
        old_paths,
        "Historical rain run and history; test/pretrained metrics with samples omitted",
    )

    # Validate lower-resolution export provenance without bundling graphs/fixtures.
    export_sources = [
        f"artifacts/edge_v1/exports/{shape}/restormer/validation.json"
        for shape in ["256x448", "128x224", "128x128"]
    ]
    write(
        "evidence/restormer_export_checks.json",
        {Path(p).parts[-3]: read(p) for p in export_sources},
        export_sources,
        "Local numerical export reports, not TensorRT device measurements",
    )

    (PACK / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    paths = sorted(
        p
        for p in PACK.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts
    )
    (PACK / "SHA256SUMS").write_text(
        "".join(f"{sha(p)}  {p.relative_to(PACK).as_posix()}\n" for p in paths)
    )
    print(f"Wrote {len(paths)} evidence/document files plus SHA256SUMS")


if __name__ == "__main__":
    main()
