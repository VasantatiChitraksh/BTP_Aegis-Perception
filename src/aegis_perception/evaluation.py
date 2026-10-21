"""Aggregation that keeps weather prevalence and repeated frames explicit."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np


def background_means(samples, metrics):
    """Average variants within strata, then strata within each background."""
    strata = defaultdict(lambda: defaultdict(list))
    for row in samples:
        strata[row["weather"], row["group_id"]][row.get("stratum") or "all"].append(row)
    return {
        key: {
            m: float(np.mean([np.mean([r[m] for r in variants]) for variants in groups.values()]))
            for m in metrics
        }
        for key, groups in strata.items()
    }


def balanced_panel(records, per_weather: int = 224, seed: int = 42):
    """Select by IDs only, cycling backgrounds before taking more variants."""
    if per_weather <= 0:
        raise ValueError("per_weather must be positive")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        if not row.group_id:
            raise ValueError("Balanced panels require audited background group IDs")
        grouped[row.weather][row.group_id].append(row)

    def order(value):
        return hashlib.sha256(f"panel:{seed}:{value}".encode()).digest()

    selected = []
    for weather, groups in sorted(grouped.items()):
        if sum(map(len, groups.values())) < per_weather:
            raise ValueError(f"{weather} has fewer than {per_weather} available pairs")
        variants = [
            sorted(groups[group], key=lambda r: order(r.sample_id))
            for group in sorted(groups, key=order)
        ]
        chosen = []
        depth = 0
        while len(chosen) < per_weather:
            for rows in variants:
                if depth < len(rows):
                    chosen.append(rows[depth])
                    if len(chosen) == per_weather:
                        break
            depth += 1
        selected.extend(chosen)
    return selected


def summarize_samples(samples: list[dict]) -> dict:
    """Mean images within background, backgrounds within weather, then weather."""
    if not samples:
        raise ValueError("Cannot summarize an empty evaluation")
    metrics = ("psnr_db", "ssim", "raw_psnr_db", "raw_ssim", "psnr_gain_db", "ssim_gain")
    rows = []
    identifiers = set()
    for sample in samples:
        row = dict(sample)
        if row["sample_id"] in identifiers:
            raise ValueError(f"Duplicate evaluation sample: {row['sample_id']}")
        identifiers.add(row["sample_id"])
        if not row.get("group_id"):
            raise ValueError("Scene-balanced reporting requires group_id")
        row["psnr_gain_db"] = row["psnr_db"] - row["raw_psnr_db"]
        row["ssim_gain"] = row["ssim"] - row["raw_ssim"]
        if any(not math.isfinite(row[m]) for m in metrics):
            raise ValueError(f"Non-finite metrics for {row['sample_id']}")
        rows.append(row)

    def mean(items):
        return {m: float(np.mean([r[m] for r in items])) for m in metrics}

    weather_rows = defaultdict(list)
    groups = defaultdict(list)
    for row in rows:
        weather_rows[row["weather"]].append(row)
        groups[row["weather"], row["group_id"]].append(row)
    weather = {}
    scenes = {}
    counts = {}
    backgrounds = background_means(rows, metrics)
    for condition, items in sorted(weather_rows.items()):
        scene_rows = [v for (w, _), v in backgrounds.items() if w == condition]
        weather[condition] = mean(items)
        scenes[condition] = mean(scene_rows)
        counts[condition] = {
            "images": len(items),
            "background_groups": len(scene_rows),
            "clusters": len({r.get("cluster_id") or r["group_id"] for r in items}),
            "psnr_worse_than_raw": sum(r["psnr_gain_db"] < 0 for r in items),
            "ssim_worse_than_raw": sum(r["ssim_gain"] < 0 for r in items),
            "both_worse_than_raw": sum(r["psnr_gain_db"] < 0 and r["ssim_gain"] < 0 for r in items),
        }
    macro_image = mean(list(weather.values()))
    macro_scene = mean(list(scenes.values()))
    weather["all"] = mean(rows)
    scenes["macro_weather"] = macro_scene
    return {
        "headline": {
            "aggregation": "equal weather, background, stratum; mean images within stratum"
            if any(r.get("stratum") not in (None, "", "all") for r in rows)
            else "equal weather, equal background within weather",
            **macro_scene,
        },
        "summary": weather,
        "macro_weather_summary": macro_image,
        "scene_balanced_summary": scenes,
        "counts": counts,
    }


def clustered_intervals(samples, *, reference=None, repetitions=2000, seed=20261015):
    """Percentile cluster bootstrap, preserving weather and paired observations.

    Resample entire known clusters, never individual video frames. Intervals
    describe sampling uncertainty conditional on this model and test selection;
    they do not include unobserved domains or training-seed variability.
    """
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    summarize_samples(samples)  # Reject duplicates/nonfinite/missing groups.
    rows = [dict(r) for r in samples]
    metrics = ("psnr_db", "ssim")
    if reference is not None:
        summarize_samples(reference)
        previous = {r["sample_id"]: r for r in reference}
        if set(previous) != {r["sample_id"] for r in rows}:
            raise ValueError("Paired intervals require exactly matching sample IDs")
        for row in rows:
            old = previous[row["sample_id"]]
            for field in ("weather", "group_id", "stratum", "cluster_id"):
                if row.get(field) != old.get(field):
                    raise ValueError(f"Paired metadata mismatch: {field}")
            for metric in metrics:
                row[metric] -= old[metric]
    means = background_means(rows, metrics)
    membership = {}
    for row in rows:
        key = row["weather"], row["group_id"]
        cluster = row.get("cluster_id") or row["group_id"]
        if key in membership and membership[key] != cluster:
            raise ValueError("A background cannot belong to two bootstrap clusters")
        membership[key] = cluster
    grouped = defaultdict(lambda: defaultdict(list))
    for key, values in means.items():
        grouped[key[0]][membership[key]].append([values[m] for m in metrics])
    rng = np.random.default_rng(seed)
    distributions, points, weather = [], [], {}
    warnings = []

    def describe(point, draws):
        return {
            m: {
                "estimate": float(point[i]),
                "percentile_95": [float(v) for v in np.quantile(draws[:, i], [0.025, 0.975])],
            }
            for i, m in enumerate(metrics)
        }

    for condition, clusters in sorted(grouped.items()):
        values = [np.asarray(v) for _, v in sorted(clusters.items())]
        sums = np.array([v.sum(axis=0) for v in values])
        sizes = np.array([len(v) for v in values])
        point = sums.sum(axis=0) / sizes.sum()
        index = rng.integers(0, len(values), size=(repetitions, len(values)))
        draws = sums[index].sum(axis=1) / sizes[index].sum(axis=1)[:, None]
        detail = {
            "clusters": len(values),
            "backgrounds": int(sizes.sum()),
            "metrics": describe(point, draws),
        }
        if len(values) > 1:
            omitted = (sums.sum(axis=0) - sums) / (sizes.sum() - sizes)[:, None]
            detail["leave_one_cluster_out_range"] = {
                m: [float(omitted[:, i].min()), float(omitted[:, i].max())]
                for i, m in enumerate(metrics)
            }
        if len(values) < 20:
            warnings.append(f"{condition}: only {len(values)} clusters; interval is exploratory.")
        weather[condition] = detail
        distributions.append(draws)
        points.append(point)
    return {
        "method": "paired cluster bootstrap" if reference is not None else "cluster bootstrap",
        "seed": seed,
        "repetitions": repetitions,
        "weather": weather,
        "macro_weather": describe(np.mean(points, axis=0), np.mean(distributions, axis=0)),
        "warnings": warnings,
        "scope": "Conditional on fixed model/test selection; no training-seed uncertainty. "
        "95% intervals are marginal, not simultaneous significance tests.",
    }
