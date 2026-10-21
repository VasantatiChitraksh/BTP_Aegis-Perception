from __future__ import annotations

import json
import random
import runpy
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from aegis_perception.data.manifest import PairRecord, load_manifest, write_manifest
from aegis_perception.data.paired import PairedImageDataset
from aegis_perception.data.protocol import verify_protocol
from aegis_perception.data.sampling import RestorationBatchSampler
from aegis_perception.evaluation import clustered_intervals, summarize_samples
from aegis_perception.training import _validate

ROOT = Path(__file__).resolve().parents[1]


def test_continuous_sampler_covers_background_strata_variants_and_rank_shards():
    rows = [
        PairRecord(
            f"{w}/{g}/{s}/{v}", Path("a"), Path("b"), "train", w, "fixture", str(g), str(s), str(g)
        )
        for w in ("rain", "smoke")
        for g in range(5)
        for s in range(2)
        for v in range(3)
    ]
    options = dict(
        probabilities={"rain": 0.5, "smoke": 0.5}, balanced=True, stratified_cycles=True, seed=42
    )
    sampler = RestorationBatchSampler(rows, 2, 6, **options)
    streams = {w: [] for w in options["probabilities"]}
    for epoch in range(1, 11):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        shards = []
        for rank in range(2):
            shard = RestorationBatchSampler(rows, 2, 6, rank=rank, world_size=2, **options)
            shard.set_epoch(epoch)
            shards.append(list(shard))
        assert batches == [a + b for a, b in zip(*shards, strict=True)]
        for batch in batches:
            assert {rows[i].weather for i, _ in batch} == set(streams)
            for i, seed in batch:
                streams[rows[i].weather].append((rows[i], seed))
    for draws in streams.values():
        assert len({seed for _, seed in draws}) == 30
        assert len({r.sample_id for r, _ in draws}) == 30
        for start in range(0, 30, 5):
            assert len({r.group_id for r, _ in draws[start : start + 5]}) == 5
        for g in range(5):
            variants = [r for r, _ in draws if r.group_id == str(g)]
            for start in range(0, 6, 2):
                assert {r.stratum for r in variants[start : start + 2]} == {"0", "1"}


def test_augmentations_do_not_depend_on_worker_rng(tmp_path):
    path = tmp_path / "input.png"
    Image.fromarray(np.random.default_rng(42).integers(0, 255, (35, 41, 3), dtype=np.uint8)).save(
        path
    )
    rows = [
        PairRecord(str(i), path, path, "train", "rain", "fixture", str(i), "q0", str(i))
        for i in range(3)
    ]
    manifest = tmp_path / "pairs.csv"
    write_manifest(manifest, rows)
    dataset = PairedImageDataset(manifest, "train", (16, 16), True, "crop")
    sampler = RestorationBatchSampler(
        rows, 1, 6, probabilities={"rain": 1}, balanced=True, stratified_cycles=True
    )
    sampler.set_epoch(1)
    first = list(DataLoader(dataset, batch_sampler=sampler, num_workers=0))
    random.seed(999)
    second = list(DataLoader(dataset, batch_sampler=sampler, num_workers=2))
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a["input"], b["input"], atol=0, rtol=0)
        torch.testing.assert_close(a["input"], a["target"], atol=0, rtol=0)


def metric_rows():
    return [
        dict(
            sample_id=str(i),
            weather="smoke",
            group_id=str(i // 2),
            cluster_id=str(i // 4),
            stratum=str(i % 2),
            psnr_db=20.0 + i,
            raw_psnr_db=15.0,
            ssim=0.7 + i / 100,
            raw_ssim=0.5,
        )
        for i in range(12)
    ]


def test_stratum_weighting_and_paired_cluster_intervals():
    rows = metric_rows()
    repeated = rows + [{**rows[0], "sample_id": f"repeat{i}"} for i in range(100)]
    a, b = summarize_samples(rows)["headline"], summarize_samples(repeated)["headline"]
    assert a.pop("aggregation") == b.pop("aggregation")
    assert a == pytest.approx(b, abs=1e-12)
    result = clustered_intervals(rows, reference=rows, repetitions=200)
    assert result["weather"]["smoke"]["clusters"] == 3
    assert result["weather"]["smoke"]["backgrounds"] == 6
    assert result["warnings"]
    assert result["macro_weather"]["psnr_db"]["percentile_95"] == [0.0, 0.0]
    changed = [{**r, "psnr_db": r["psnr_db"] + 2, "ssim": r["ssim"] + 0.01} for r in rows]
    delta = clustered_intervals(changed, reference=rows, repetitions=200)
    assert delta["macro_weather"]["psnr_db"]["percentile_95"] == [2.0, 2.0]
    with pytest.raises(ValueError, match="matching sample"):
        clustered_intervals(rows, reference=rows[:-1])
    with pytest.raises(ValueError, match="two bootstrap clusters"):
        clustered_intervals([{**rows[0], "cluster_id": "bad"}] + rows[1:])


def test_validation_uses_same_stratum_estimand_as_final_evaluation():
    target = torch.zeros(3, 3, 9, 9)
    inputs = target.clone()
    inputs[0] += 0.1
    inputs[1:] += 0.4
    batch = {
        "input": inputs,
        "target": target,
        "weather": ["smoke"] * 3,
        "group_id": ["g"] * 3,
        "stratum": ["q0", "q1", "q1"],
    }
    result = _validate(torch.nn.Identity(), [batch], "cpu", compute_ssim=True)
    expected = (-10 * np.log10(0.05**2) - 10 * np.log10(0.2**2)) / 2
    assert result["smoke_scene_psnr_db"] == pytest.approx(expected, abs=1e-4)
    assert result["macro_scene_psnr_db"] != pytest.approx(result["psnr_db"])


def test_frozen_protocol_has_disjoint_partitions_and_sequence_test_coverage():
    protocol_path = ROOT / "data/manifests/weather_v3/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    from aegis_perception.data.catalog import sha256

    verify_protocol(
        {
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path),
            "manifest": str(protocol_path.parent / "active.csv"),
        }
    )
    active = load_manifest(protocol_path.parent / "active.csv")
    reserve = load_manifest(protocol_path.parent / "reserve.csv")
    assert not {r.sample_id for r in active} & {r.sample_id for r in reserve}
    assert Counter(r.split for r in active) == {"train": 18610, "val": 2011, "test": 2334}
    clusters = {}
    for row in active + reserve:
        assert row.stratum and row.cluster_id
        assert clusters.setdefault(row.cluster_id, row.split) == row.split
    snow = [r for r in active if r.weather == "snow" and r.split == "test"]
    assert len({r.cluster_id for r in snow}) == 256
    assert set(Counter(r.cluster_id for r in snow).values()) == {3}
    for weather, splits in protocol["counts"].items():
        for split, counts in splits.items():
            assert sum((r.weather, r.split) == (weather, split) for r in active) == counts["images"]


def test_snow_selection_preserves_city_and_sequence_coverage():
    module = runpy.run_path(str(ROOT / "scripts/data/prepare_weather_protocol.py"))
    rows = [
        PairRecord(
            f"{city}/{seq}/{g}/{s}",
            Path("a"),
            Path(f"{city}_target.png"),
            "test",
            "snow",
            "fixture",
            f"{city}/{seq}/{g}",
            f"q{s}",
            f"{city}/{seq}",
        )
        for city in ("a", "b")
        for seq in range(5)
        for g in range(3)
        for s in range(3)
    ]
    selected = module["snow_panel"](rows, 4)
    assert len(selected) == 12
    assert len({r.cluster_id for r in selected}) == 4
    assert {r.target_path.name[0] for r in selected} == {"a", "b"}
    assert selected == module["snow_panel"](list(reversed(rows)), 4)
    capped = module["cap_strata"]([replace(r, group_id="one") for r in rows], 2)
    assert len(capped) == 6


def test_comparison_checks_pair_identity_raw_scores_and_frozen_provenance(tmp_path):
    from aegis_perception.data.catalog import sha256

    compare = runpy.run_path(str(ROOT / "scripts/restoration/compare_weather_protocol.py"))[
        "compare"
    ]
    rows = [
        PairRecord(
            str(i),
            tmp_path / f"in{i}",
            tmp_path / f"gt{i}",
            "test",
            "smoke",
            "fixture",
            str(i // 2),
            str(i % 2),
            str(i // 4),
        )
        for i in range(12)
    ]
    old_manifest, active = tmp_path / "old.csv", tmp_path / "active.csv"
    write_manifest(old_manifest, [replace(r, stratum="", cluster_id="") for r in rows])
    write_manifest(active, rows)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"manifest_sha256": {"active.csv": sha256(active)}}))
    old = {
        "split": "test",
        "manifest_sha256": sha256(old_manifest),
        "protocol": {"color": "RGB"},
        "samples": metric_rows(),
    }
    candidate = {
        **old,
        "manifest_sha256": sha256(active),
        "training_seed": 42,
        "condition_sha256": "same-training-condition",
    }
    old_path, new_path = tmp_path / "old.json", tmp_path / "new.json"
    old_path.write_text(json.dumps(old))
    new_path.write_text(json.dumps(candidate))
    result = compare(old_path, old_manifest, active, new_path)
    assert result["paired_change"]["macro_weather"]["psnr_db"]["estimate"] == 0
    candidate["samples"][0]["raw_ssim"] = 0.1
    new_path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="Raw metric mismatch"):
        compare(old_path, old_manifest, active, new_path)
    active.write_text(active.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen protocol"):
        compare(old_path, old_manifest, active)


def test_seed_summary_requires_all_declared_seeds_and_one_condition(tmp_path):
    summarize = runpy.run_path(str(ROOT / "scripts/restoration/summarize_weather_seeds.py"))[
        "summarize"
    ]
    paths = []
    for seed in (42, 43, 44):
        scores = {
            w: {"psnr_db": float(seed), "ssim": 0.8}
            for w in ("rain", "fog", "snow", "smoke", "macro_weather")
        }
        report = {
            k: "hash"
            for k in (
                "protocol_sha256",
                "manifest_sha256",
                "baseline_metrics_sha256",
                "condition_sha256",
            )
        }
        report.update(training_seed=seed, candidate={"scene_balanced_summary": scores})
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    result = summarize(paths)
    assert result["scores"]["smoke"]["psnr_db"]["mean"] == 43
    assert result["scores"]["smoke"]["psnr_db"]["sample_std"] == 1
    with pytest.raises(ValueError, match="exactly seeds"):
        summarize([paths[0], paths[0], paths[1]])
    report["condition_sha256"] = "other loss"
    paths[-1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="condition_sha256"):
        summarize(paths)
