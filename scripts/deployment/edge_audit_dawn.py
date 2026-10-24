#!/usr/bin/env python3
"""Audit every available DAWN annotation without running restoration or detection."""

import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from aegis_perception.edge.common import OUT, digest, write_json

spec = importlib.util.spec_from_file_location("edge_dawn", Path(__file__).with_name("edge_dawn.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
counts, excluded, weather_counts, hashes = Counter(), Counter(), Counter(), defaultdict(list)
rows = []
for path in sorted((ROOT / "data/raw/dawn").rglob("*.jpg")):
    labels, objects, omitted = module.annotations(path)
    weather = path.relative_to(ROOT / "data/raw/dawn").parts[0]
    counts.update(objects)
    excluded.update(omitted)
    weather_counts[weather] += 1
    sha = digest(path)
    hashes[sha].append(str(path.relative_to(ROOT)))
    rows.append(
        {
            "source": str(path.relative_to(ROOT)),
            "weather": weather,
            "sha256": sha,
            "objects": dict(objects),
            "excluded": dict(omitted),
        }
    )
assert len(rows) == 704
write_json(
    OUT / "dawn_annotation_audit.json",
    {
        "passed": True,
        "images": len(rows),
        "weather_counts": dict(weather_counts),
        "road_class_counts": dict(counts),
        "excluded_objects": dict(excluded),
        "exact_duplicate_groups": [v for v in hashes.values() if len(v) > 1],
        "rows": rows,
        "method": (
            "Every TXT ID equals XML COCO ID+1; every normalized box checked "
            "to absolute tolerance 2e-5; dimensions and bounds checked"
        ),
        "source": "https://data.mendeley.com/datasets/766ygrbt8y/3",
        "license": "CC BY-NC 3.0",
    },
)
print(f"All {len(rows)} image/annotation pairs passed; {sum(counts.values())} road objects")
