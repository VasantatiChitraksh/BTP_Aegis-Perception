#!/usr/bin/env python3
"""Verify the frozen benchmark and write a six-model CSV/JSON/PDF comparison."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from aegis_perception.data.catalog import sha256
from aegis_perception.data.manifest import load_manifest
from aegis_perception.evaluation import summarize_samples

LABELS = {
    "attention_pix2pix": "Attention Pix2Pix",
    "restormer": "Restormer",
    "liteweatherformer": "LiteWeather v2",
    "transweather": "TransWeather",
    "promptir": "PromptIR",
    "ramit": "RAMiT",
}
PARAMETERS = {"transweather": 38052078, "promptir": 35592263, "ramit": 934651}
WEATHERS = ("rain", "fog", "snow", "smoke")
OUT = ROOT / "artifacts/restoration/official_v1"


def collect(require_complete=False):
    references = json.loads((ROOT / "configs/restoration/official_v1/references.json").read_text())
    manifest = ROOT / "data/manifests/weather_v3/active.csv"
    records = {r.sample_id: r for r in load_manifest(manifest, "test")}
    models, expected_protocol, raw_scores = {}, None, None
    for name, label in LABELS.items():
        directory = ROOT / (
            references[name]["directory"]
            if name in references
            else f"artifacts/restoration/official_v1/{name}_seed42"
        )
        metrics_path, history_path = directory / "metrics_test.json", directory / "history.json"
        info = {
            "label": label,
            "status": "pending",
            "directory": str(directory.relative_to(ROOT)),
            "parameters": references[name]["parameters"]
            if name in references
            else PARAMETERS[name],
        }
        models[name] = info
        if not metrics_path.exists():
            if require_complete:
                raise FileNotFoundError(f"Final six-model report requires {metrics_path}")
            continue
        if name in references:
            for file, digest in references[name]["files"].items():
                if sha256(directory / file) != digest:
                    raise ValueError(f"Archived reference changed: {name}/{file}")
        metrics, history = (
            json.loads(metrics_path.read_text()),
            json.loads(history_path.read_text()),
        )
        if metrics["split"] != "test" or metrics["panel_sha256"] is not None:
            raise ValueError(f"Not the complete frozen test evaluation: {name}")
        if metrics["manifest_sha256"] != sha256(manifest):
            raise ValueError(f"Manifest mismatch: {name}")
        samples = metrics["samples"]
        if len(samples) != len(records) or {s["sample_id"] for s in samples} != set(records):
            raise ValueError(f"Test sample coverage/duplicates mismatch: {name}")
        for sample in samples:
            for field in ("weather", "group_id", "stratum", "cluster_id"):
                if sample[field] != getattr(records[sample["sample_id"]], field):
                    raise ValueError(f"Test group metadata mismatch: {name}")
        if expected_protocol is None:
            expected_protocol = metrics["protocol"]
            raw_scores = {s["sample_id"]: (s["raw_psnr_db"], s["raw_ssim"]) for s in samples}
        if metrics["protocol"] != expected_protocol:
            raise ValueError(f"Metric/precision/spatial protocol mismatch: {name}")
        for s in samples:
            if any(
                abs(a - b) > 1e-9
                for a, b in zip(
                    raw_scores[s["sample_id"]], (s["raw_psnr_db"], s["raw_ssim"]), strict=True
                )
            ):
                raise ValueError(f"Different raw test images/metrics: {name}")
        recomputed = summarize_samples(samples)
        for weather, values in recomputed["scene_balanced_summary"].items():
            for key, value in values.items():
                if abs(value - metrics["scene_balanced_summary"][weather][key]) > 1e-9:
                    raise ValueError(f"Aggregation mismatch: {name}/{weather}/{key}")
        if [h["epoch"] for h in history] != list(range(1, 61)):
            raise ValueError(f"Expected the complete 60-epoch history: {name}")
        selected = max(history, key=lambda h: h["val_macro_scene_psnr_db"])
        if metrics["checkpoint_epoch"] != selected["epoch"]:
            raise ValueError(f"Checkpoint selection mismatch: {name}")
        if (directory / "best.pt").exists() and sha256(directory / "best.pt") != metrics[
            "checkpoint_sha256"
        ]:
            raise ValueError(f"Checkpoint differs from evaluated weights: {name}")
        info.update(
            status="completed",
            metrics=metrics,
            history=history,
            metrics_sha256=sha256(metrics_path),
            history_sha256=sha256(history_path),
        )
    return {"manifest_sha256": sha256(manifest), "protocol": expected_protocol, "models": models}


def pdf_report(data, path):
    from reportlab.graphics.shapes import Drawing, Line, PolyLine, String
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 12
    models = data["models"]
    complete = sum(m["status"] == "completed" for m in models.values())
    story = []

    def p(text, style="BodyText"):
        story.extend([Paragraph(text, styles[style]), Spacer(1, 8)])

    def table(rows, best_columns=(), minimize=False):
        styling = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18354a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f7")]),
        ]
        for col in best_columns:
            values = [(i, float(row[col])) for i, row in enumerate(rows[1:], 1) if row[col] != "--"]
            if values:
                best = (min if minimize else max)(v for _, v in values)
                for i, value in values:
                    if value == best:
                        styling += [
                            ("FONTNAME", (col, i), (col, i), "Helvetica-Bold"),
                            ("BACKGROUND", (col, i), (col, i), colors.HexColor("#d6eee0")),
                        ]
        widths = [132] + [375 / (len(rows[0]) - 1)] * (len(rows[0]) - 1)
        item = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle(styling))
        story.extend([item, Spacer(1, 12)])

    p("Six-model weather restoration benchmark", "Title")
    p(
        f"{'FINAL' if complete == 6 else 'PROGRESS REPORT'} | {complete}/6 models completed. "
        "Bold green cells mark the best displayed result among completed models; tied "
        "values share highlighting."
    )
    rows = [["Model", "Params (M)", "Best epoch", "PSNR (dB) ↑", "SSIM ↑", "Gain (dB) ↑"]]
    for m in models.values():
        scores = m.get("metrics", {}).get("headline", {})
        rows.append(
            [
                m["label"],
                f"{m['parameters'] / 1e6:.3f}",
                str(m.get("metrics", {}).get("checkpoint_epoch", "--")),
                f"{scores['psnr_db']:.2f}" if scores else "--",
                f"{scores['ssim']:.4f}" if scores else "--",
                f"{scores['psnr_gain_db']:.2f}" if scores else "--",
            ]
        )
    table(rows, (3, 4, 5))
    p("Shared benchmark", "Heading2")
    table(
        [
            ["Weather", "Train", "Validation", "Test"],
            ["Rain", 773, 112, 224],
            ["Fog", 5102, 897, 1000],
            ["Snow", 5382, 618, 768],
            ["Smoke", 7353, 384, 342],
            ["Total", 18610, 2011, 2334],
        ]
    )
    p(
        "60 epochs; 4,096 seeded draws/epoch with replacement; batch 4; 61,440 optimizer updates. "
        "Weather draw probabilities: rain/fog/snow/smoke = 0.30/0.30/0.30/0.10. "
        "Paired 256-pixel crops and flips; validation and test at native size. "
        "Best checkpoint selected using validation macro scene PSNR. "
        "Headline metrics give each weather equal weight, then backgrounds and strata."
    )
    p("Initialization and scope", "Heading2")
    p(
        "Attention Pix2Pix and LiteWeather v2 started randomly. Restormer used official "
        "deraining weights. "
        "TransWeather uses official All-Weather weights; PromptIR uses its official "
        "all-in-one checkpoint; "
        "RAMiT uses its official deraining checkpoint. These are fine-tuned systems with "
        "unequal pretraining, "
        "not an architecture-only comparison or a reproduction of each paper's original "
        "training recipe. "
        "The old project 'Small/Tiny' implementations are excluded."
    )
    story.append(PageBreak())
    p("Weather-specific quality and failures", "Title")
    for metric, title, digits in (
        ("psnr_db", "PSNR (dB), higher is better", 2),
        ("ssim", "SSIM, higher is better", 4),
    ):
        p(title, "Heading2")
        rows = [["Model", *[w.title() for w in WEATHERS]]]
        for m in models.values():
            scores = m.get("metrics", {}).get("scene_balanced_summary")
            rows.append(
                [
                    m["label"],
                    *[f"{scores[w][metric]:.{digits}f}" if scores else "--" for w in WEATHERS],
                ]
            )
        table(rows, (1, 2, 3, 4))
    p("Smoke images worse than degraded input (lower is better)", "Heading2")
    rows = [["Model", "PSNR worse /342", "SSIM worse /342", "Both worse /342"]]
    for m in models.values():
        c = m.get("metrics", {}).get("counts", {}).get("smoke")
        rows.append(
            [
                m["label"],
                *[
                    str(c[k]) if c else "--"
                    for k in ("psnr_worse_than_raw", "ssim_worse_than_raw", "both_worse_than_raw")
                ],
            ]
        )
    table(rows, (1, 2, 3), minimize=True)
    p(
        "Smoke has only six test backgrounds: 342 correlated images do not provide 342 "
        "independent scenes. "
        "Per-image values and clustered uncertainty intervals remain in metrics_test.json. "
        "Improved average PSNR/SSIM does not imply every image improved."
    )
    story.append(PageBreak())
    p("Training progress and interpretation", "Title")
    palette = ["#ad5c36", "#6554a4", "#16809c", "#be3e5b", "#668829", "#b58313"]
    for key, title in (
        ("val_macro_scene_psnr_db", "Validation PSNR (dB)"),
        ("val_macro_scene_ssim", "Validation SSIM"),
    ):
        drawing = Drawing(500, 180)
        completed = [(i, m) for i, m in enumerate(models.values()) if m["status"] == "completed"]
        vals = [h[key] for _, m in completed for h in m["history"]]
        if vals:
            low, high = min(vals), max(vals)
            margin = max((high - low) * 0.05, 0.001)
            low, high = low - margin, high + margin
            drawing.add(String(40, 166, title, fontSize=10))
            for fraction in (0, 0.5, 1):
                y = 25 + fraction * 125
                drawing.add(Line(40, y, 490, y, strokeColor=colors.HexColor("#dde3e8")))
                drawing.add(String(0, y, f"{low + fraction * (high - low):.3f}", fontSize=7))
            for i, m in completed:
                points = [
                    v
                    for h in m["history"]
                    for v in (
                        40 + (h["epoch"] - 1) * 450 / 59,
                        25 + (h[key] - low) * 125 / (high - low),
                    )
                ]
                drawing.add(
                    PolyLine(points, strokeColor=colors.HexColor(palette[i]), strokeWidth=1.3)
                )
            drawing.add(String(40, 8, "Epoch 1", fontSize=8))
            drawing.add(String(452, 8, "Epoch 60", fontSize=8))
        story.append(drawing)
    p(
        "Legend: "
        + " | ".join(
            f'<font color="{palette[i]}">{m["label"]}</font>' for i, m in enumerate(models.values())
        )
    )
    p("Limits of the comparison", "Heading2")
    p(
        "Single fine-tuning seed per model; no claim of statistical superiority from "
        "tiny differences. "
        "The test split has been evaluated previously. Official pretraining datasets have not been "
        "exhaustively checked for overlap with our benchmark; PromptIR uses RESIDE pretraining, "
        "so fog results require particular caution. Parameter count is not measured Jetson FPS. "
        "Deployment latency, quantized quality and YOLOv8 detection performance require "
        "separate tests."
    )
    p(
        "Provenance: configs/restoration/official_v1/sources.json pins official source "
        "revisions and "
        "weight hashes. comparison.json retains verified results, histories and artifact hashes. "
        "No pending model has an inferred or fabricated score."
    )
    SimpleDocTemplate(
        str(path), pagesize=A4, rightMargin=38, leftMargin=38, topMargin=32, bottomMargin=32
    ).build(story)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    data = collect(args.require_complete)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(data, indent=2) + "\n")
    rows = []
    for name, model in data["models"].items():
        for weather in ("macro_weather", *WEATHERS):
            scores = model.get("metrics", {}).get("scene_balanced_summary", {}).get(weather, {})
            rows.append(
                {
                    "model": name,
                    "status": model["status"],
                    "weather": weather,
                    "parameters": model["parameters"],
                    **{
                        key: scores.get(key, "")
                        for key in ("psnr_db", "ssim", "psnr_gain_db", "ssim_gain")
                    },
                }
            )
    with (OUT / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_pdf:
        pdf_report(data, OUT / "six_model_report.pdf")
    completed = sum(m["status"] == "completed" for m in data["models"].values())
    print(f"Comparison: {OUT}; completed {completed}/6")


if __name__ == "__main__":
    main()
