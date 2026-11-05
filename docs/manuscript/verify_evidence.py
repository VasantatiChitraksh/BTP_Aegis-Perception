"""Verify the manuscript pack and recompute its six-model aggregates (stdlib only)."""

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

PACK = Path(__file__).resolve().parent
ROOT = PACK.parents[1]


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL: " + message)


def mean(values):
    return sum(values) / len(values)


def main():
    listed = set()
    for line in (PACK / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = PACK / name
        require(path.is_file(), f"missing {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"checksum {name}")
        listed.add(name)
    actual = {
        p.relative_to(PACK).as_posix()
        for p in PACK.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts
    }
    require(actual == listed, "manifest does not list every pack file")

    six = json.loads((PACK / "evidence/six_models.json").read_text())
    manifest = ROOT / "data/manifests/weather_v3/active.csv"
    require(
        hashlib.sha256(manifest.read_bytes()).hexdigest() == six["manifest_sha256"],
        "active dataset manifest differs",
    )
    with manifest.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    expected = {r["sample_id"]: r for r in records if r["split"] == "test"}
    require(len(expected) == 2334, "test size")
    protocol = json.loads((PACK / "evidence/data_protocol.json").read_text())
    with (PACK / "tables/dataset_split.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            group = [
                r for r in records if r["weather"] == row["weather"] and r["split"] == row["split"]
            ]
            require(len(group) == int(row["pairs"]), f"split count {row}")
            require(
                len({r["group_id"] for r in group}) == int(row["backgrounds"]), "background count"
            )
            require(len({r["cluster_id"] for r in group}) == int(row["clusters"]), "cluster count")
            require(
                len(group) == protocol["counts"][row["weather"]][row["split"]]["images"],
                "protocol count",
            )

    with (PACK / "tables/six_models.csv").open(newline="") as stream:
        table = {(r["model"], r["weather"]): r for r in csv.DictReader(stream)}
    with gzip.open(PACK / "evidence/per_image_restoration.csv.gz", "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 6 * 2334, "six-model sample count")
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
        ref = expected[row["sample_id"]]
        for key in ["weather", "group_id", "cluster_id", "stratum"]:
            require(row[key] == ref[key], f"sample membership {key}")
    require(set(by_model) == set(six["models"]), "model identities")

    for name, model in six["models"].items():
        data = by_model[name]
        require(
            len(data) == len({r["sample_id"] for r in data}) == 2334,
            f"duplicate/missing sample {name}",
        )
        require(
            [int(h["epoch"]) for h in model["history"]] == list(range(1, 61)), f"history {name}"
        )
        require(
            model["metrics"]["checkpoint_epoch"] == model["model_library_identity"]["best_epoch"],
            f"checkpoint identity {name}",
        )
        require(
            model["metrics"]["checkpoint_sha256"] == model["model_library_identity"]["best_sha256"],
            f"checkpoint hash {name}",
        )
        for metric in ["psnr_db", "ssim"]:
            strata = defaultdict(list)
            for row in data:
                strata[row["weather"], row["group_id"], row["stratum"]].append(float(row[metric]))
            backgrounds = defaultdict(list)
            for (weather, group, _), values in strata.items():
                backgrounds[weather, group].append(mean(values))
            weather_values = defaultdict(list)
            for (weather, _), values in backgrounds.items():
                weather_values[weather].append(mean(values))
            result = {w: mean(values) for w, values in weather_values.items()}
            result["macro_weather"] = mean(list(result.values()))
            for weather, value in result.items():
                require(
                    math.isclose(
                        value, float(table[model["label"], weather][metric]), abs_tol=1e-9
                    ),
                    f"reaggregation {name}/{weather}/{metric}",
                )
            require(
                math.isclose(
                    result["macro_weather"], model["metrics"]["headline"][metric], abs_tol=1e-9
                ),
                f"headline {name}/{metric}",
            )

    joint = json.loads((PACK / "evidence/joint_experiments.json").read_text())
    require(joint["selection_before_test"]["selected"] == "original", "joint selection changed")
    require(joint["selection_before_test"]["passing"] == [], "joint passing set changed")
    with (PACK / "tables/joint_test.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            metrics = (
                joint["original_test_metrics"]
                if row["model"] == "original"
                else joint["arms"][row["model"]]["test_metrics"]
            )
            for metric in ["psnr_db", "ssim"]:
                require(
                    math.isclose(
                        float(row["test_" + metric]), metrics["headline"][metric], abs_tol=1e-9
                    ),
                    "joint test summary",
                )
    edge = json.loads((PACK / "evidence/edge_readiness.json").read_text())
    reference = {(r["view"], r["weather"]): r for r in edge["detection_pilot"]}
    with (PACK / "tables/detection_pilot72.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            for key in ["images", "map50", "map50_95", "precision", "recall"]:
                require(
                    float(row[key]) == reference[row["view"], row["weather"]][key],
                    "detection table",
                )
    print(
        f"PASS: {len(listed)} file hashes; dataset counts/groups; all six native test aggregates; "
        "checkpoint identities; joint selection/results; detection table."
    )
    print("This checks recorded evidence; it does not rerun models or validate Jetson performance.")


if __name__ == "__main__":
    main()
