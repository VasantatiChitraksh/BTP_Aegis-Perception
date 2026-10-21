#!/usr/bin/env python3
"""Build a three-page report; never invent missing training or test measurements.

Run with system python3 (matplotlib/reportlab). Optional preview images are
generated separately with the project's torch environment.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from aegis_perception.data.manifest import load_manifest

RUN = ROOT / "artifacts/restoration/baselines_v1/attention_pix2pix_seed42"
OUT = RUN / "report"
WEATHERS = ("rain", "fog", "snow", "smoke")
INK, BLUE, ORANGE = "#18354a", "#177e9d", "#c15b35"
WIDTH = A4[0] - 80


def read(path):
    return json.loads(path.read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = read(OUT / "log_evidence.json")
    have_metrics = (RUN / "metrics_test.json").exists()
    have_history = (RUN / "history.json").exists()
    metrics = read(RUN / "metrics_test.json") if have_metrics else evidence
    history = read(RUN / "history.json") if have_history else evidence["history"]
    manifest = ROOT / "data/manifests/weather_v3/active.csv"
    records = load_manifest(manifest)
    counts = Counter((r.weather, r.split) for r in records)
    if have_metrics:
        assert metrics["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
        expected = {r.sample_id for r in records if r.split == "test"}
        assert {r["sample_id"] for r in metrics["samples"]} == expected
        assert len(metrics["samples"]) == len(expected)
    if have_history:
        assert [int(r["epoch"]) for r in history] == list(range(1, 61))
    previews = read(OUT / "previews.json") if (OUT / "previews.json").exists() else None
    if previews:
        assert have_metrics and previews["checkpoint_sha256"] == metrics["checkpoint_sha256"]
    complete = have_metrics and have_history and previews is not None
    destination = OUT / (
        "attention_pix2pix_report.pdf" if complete else "attention_pix2pix_preliminary_report.pdf"
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.25))
    for ax, key, title in zip(
        axes, ("psnr_db", "ssim"), ("Balanced test PSNR (dB)", "Balanced test SSIM"), strict=True
    ):
        h = metrics["headline"]
        values = [h[f"raw_{key}"], h[key]]
        ax.bar(["Degraded input", "Attention Pix2Pix"], values, color=["#afbdc6", BLUE], width=0.6)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(0, 1 if key == "ssim" else 30)
        for i, value in enumerate(values):
            ax.text(
                i,
                value + (0.015 if key == "ssim" else 0.45),
                f"{value:.4f}" if key == "ssim" else f"{value:.2f}",
                ha="center",
            )
    fig.tight_layout()
    fig.savefig(OUT / "test_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.35))
    for ax, key, label in (
        (axes[0], "val_macro_scene_psnr_db", "Validation PSNR (dB)"),
        (axes[1], "val_macro_scene_ssim", "Validation SSIM"),
    ):
        valid = [r for r in history if key in r]
        ax.plot(
            [r["epoch"] for r in valid],
            [r[key] for r in valid],
            color=BLUE,
            marker=".",
            linestyle="-" if have_history else "none",
        )
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 61)
        ax.grid(alpha=0.18)
        ax.axvline(metrics["checkpoint_epoch"], color=ORANGE, linestyle="--", alpha=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "validation.png", dpi=180)
    plt.close(fig)

    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    for name, file in (("Body", "DejaVuSans.ttf"), ("Bold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold")
    styles = {
        "title": ParagraphStyle(
            "title",
            fontName="Bold",
            fontSize=23,
            leading=28,
            textColor=colors.HexColor(INK),
            spaceAfter=8,
        ),
        "heading": ParagraphStyle(
            "heading",
            fontName="Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor(BLUE),
            spaceBefore=9,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Body",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor(INK),
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="Body",
            fontSize=7.3,
            leading=10,
            textColor=colors.HexColor("#516575"),
            spaceAfter=5,
        ),
    }
    story = []

    def p(text, style="body"):
        return Paragraph(text, styles[style])

    def add(text, style="body"):
        story.append(p(text, style))

    def table(rows, widths):
        t = Table(
            [[p(str(c), "small") for c in row] for row in rows], colWidths=widths, hAlign="LEFT"
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3eef3")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f5f8fa")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.extend([t, Spacer(1, 6)])

    def figure(path, height):
        story.append(Image(str(path), width=WIDTH, height=height))
        story.append(Spacer(1, 5))

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#cbd8df"))
        canvas.line(40, 35, A4[0] - 40, 35)
        canvas.setFont("Body", 7)
        canvas.drawString(40, 22, "AEGIS PERCEPTION | Attention Pix2Pix | 20 October 2026")
        canvas.drawRightString(A4[0] - 40, 22, f"{doc.page} / 3")

    add("Attention Pix2Pix<br/>Training &amp; Test Report", "title")
    add("Seed 42 · DGX jobs 47624–47626 · 60 epochs · One A100", "small")
    if not complete:
        add(
            "<b>PRELIMINARY:</b> "
            + (
                "Downloaded measurements available; restoration previews are pending."
                if have_metrics and have_history
                else "Based on user-pasted completion logs and local frozen configuration. "
                "Full run files have not yet been retrieved; missing values are not inferred."
            ),
            "small",
        )
    h = metrics["headline"]
    add(
        f"<b>Result:</b> balanced test PSNR <b>{h['psnr_db']:.2f} dB</b> "
        f"and SSIM <b>{h['ssim']:.4f}</b>, "
        f"gains of <b>{h['psnr_gain_db']:.2f} dB</b> and <b>{h['ssim_gain']:.4f}</b> "
        "over degraded inputs. "
        "<b>Smoke is a major failure condition:</b> 341/342 smoke pairs lose SSIM. "
        "This model is a measured baseline, not evidence of reliable restoration in every weather."
    )
    figure(OUT / "test_summary.png", 139)
    add("Data and evaluation protocol", "heading")
    table(
        [["Weather / source", "Train", "Validation", "Test"]]
        + [
            [f"{w.title()} / {source}", *[f"{counts[w, s]:,}" for s in ("train", "val", "test")]]
            for w, source in zip(
                WEATHERS,
                ("RealRain-1k-H", "RESIDE-6K", "SnowCityscapes", "SmokeBench"),
                strict=True,
            )
        ]
        + [["Total", "18,610", "2,011", "2,334"]],
        [WIDTH - 210, 70, 70, 70],
    )
    add(
        "Native-resolution paired RGB evaluation; padding is removed before scoring. "
        "The headline averages images within stratum, strata within background, backgrounds "
        "within weather, then weights the four weather types equally (25% each). "
        "The unequal test counts therefore do not let snow dominate the headline.",
        "small",
    )
    add("Architecture and optimization", "heading")
    add(
        "Six-level U-Net (base width 64), four additive skip-attention gates, "
        "an ungated highest-resolution skip, and a conditional 94×94 PatchGAN. "
        "This is the repository's attention Pix2Pix variant, not a Transformer."
    )
    add(
        "Random initialization; generator objective = adversarial BCE + 100 × L1 "
        "on [-1,1] images. Adam: learning rate 10⁻⁴, betas (0.5, 0.999), no weight decay; "
        "cosine decay to 10⁻⁶, BF16, gradient clip 1. Batch 4, 256×256 paired crops/flips. "
        "60 × 4,096 draws = 61,440 updates. Sampling with replacement: rain/fog/snow/smoke "
        "= 30/30/30/10%, then uniform background and uniform variant.",
        "small",
    )

    story.append(PageBreak())
    add("Convergence and failure analysis", "title")
    add(
        f"Checkpoint selected at epoch <b>{metrics['checkpoint_epoch']}</b> using "
        "balanced validation PSNR; test scores do not select the checkpoint. "
        "SSIM is recorded but is not the selection criterion."
    )
    figure(OUT / "validation.png", 150)
    add(
        "Solid lines use the full saved history."
        if have_history
        else "Only epochs present in the pasted logs are plotted. "
        "Gaps are intentionally unconnected; "
        "these points are not a reconstructed 60-epoch learning curve.",
        "small",
    )
    if have_metrics:
        add("Per-weather test means (background/stratum balanced)", "heading")
        table(
            [["Weather", "PSNR dB", "SSIM", "ΔPSNR dB", "ΔSSIM"]]
            + [
                [
                    w.title(),
                    f"{r['psnr_db']:.2f}",
                    f"{r['ssim']:.4f}",
                    f"{r['psnr_gain_db']:+.2f}",
                    f"{r['ssim_gain']:+.4f}",
                ]
                for w in WEATHERS
                for r in [metrics["scene_balanced_summary"][w]]
            ],
            [95, 100, 100, 110, WIDTH - 405],
        )
    else:
        add(
            "Per-weather mean PSNR/SSIM were not printed in the supplied logs. "
            "They will be added from metrics_test.json; regression counts below are available.",
            "small",
        )
    add("Pairs worse than the degraded input", "heading")
    table(
        [["Weather", "Test pairs", "PSNR lower", "SSIM lower", "Both lower"]]
        + [
            [
                w.title(),
                str(r["images"]),
                f"{r['psnr_worse_than_raw']} ({100 * r['psnr_worse_than_raw'] / r['images']:.1f}%)",
                f"{r['ssim_worse_than_raw']} ({100 * r['ssim_worse_than_raw'] / r['images']:.1f}%)",
                str(r["both_worse_than_raw"]),
            ]
            for w in WEATHERS
            for r in [metrics["counts"][w]]
        ],
        [75, 65, 130, 130, WIDTH - 400],
    )
    add(
        "<b>Interpretation:</b> snow has no PSNR/SSIM regressions in this test. "
        "Fog and rain contain failures despite the overall gain. Smoke loses PSNR on "
        "52.3% and SSIM on 99.7% of pairs; the aggregate result cannot establish that "
        "the smoke issue is solved."
    )
    add(
        "<b>Limits:</b> one training seed; smoke test images span only six background groups, "
        "so 342 frames are not 342 independent scenes. This is a previously inspected benchmark. "
        "Counts describe frequency, not the magnitude or cause of errors. "
        "Adversarial training alone does not establish improved visual quality, "
        "YOLO accuracy, or Jetson latency.",
        "small",
    )

    story.append(PageBreak())
    add("Visual evidence", "title")
    lookup = {r.sample_id: r for r in records if r.split == "test"}
    if previews:
        selected = previews["samples"]
        add(
            "Input → restored → clean reference. One median PSNR-gain test pair per weather, "
            "plus the smoke pair with the largest SSIM loss. These are diagnostic selections, "
            "not an independent visual-quality study.",
            "small",
        )
        add(
            "Restored previews use the verified checkpoint with native-resolution CPU FP32 "
            "inference, then are reduced for display. Official numerical results above remain "
            "the saved DGX BF16 evaluation; preview pixels may differ slightly.",
            "small",
        )
    else:
        selected = [
            {
                "sample_id": min(
                    (r.sample_id for r in lookup.values() if r.weather == w),
                    key=lambda s: hashlib.sha256(s.encode()).hexdigest(),
                ),
                "weather": w,
            }
            for w in WEATHERS
        ]
        add(
            "<b>Dataset examples only — no GAN output is shown yet.</b> "
            "Each row contains a degraded input and its paired clean reference. "
            "One test pair per weather is selected by the smallest SHA-256 of its ID, "
            "without using scores. IDs retain original source folder names; every example "
            "belongs to the current frozen test split. Restored outputs require the checkpoint.",
            "small",
        )
    visual_provenance = []
    for i, item in enumerate(selected):
        record = lookup[item["sample_id"]]
        pairs = [("Degraded input", record.input_path)]
        if previews:
            pairs.append(("Restored (CPU FP32)", OUT / item["prediction"]))
        pairs.append(("Clean reference", record.target_path))
        cells, labels = [], []
        max_w = (WIDTH - 12 * (len(pairs) - 1)) / len(pairs)
        for j, (label, path) in enumerate(pairs):
            with PILImage.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((850, 450))
                thumbnail = OUT / f"display_{i}_{j}.png"
                im.save(thumbnail)
                scale = min(max_w / im.width, (76 if previews else 98) / im.height)
                cells.append(
                    Image(str(thumbnail), width=im.width * scale, height=im.height * scale)
                )
            labels.append(p(label, "small"))
        add(escape(item["weather"].title() + " · " + record.sample_id), "small")
        t = Table([labels, cells], colWidths=[WIDTH / len(pairs)] * len(pairs))
        t.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.extend([t, Spacer(1, 8)])
        visual_provenance.append(
            {
                "sample_id": record.sample_id,
                "input": str(record.input_path),
                "target": str(record.target_path),
            }
        )
    add("Evidence and reproducibility", "heading")
    add(
        "Configuration: configs/restoration/baselines_v1/attention_pix2pix.yaml. "
        "Split: data/manifests/weather_v3/active.csv. Run artifacts: "
        "attention_pix2pix_seed42/{history.json, metrics_test.json, best.pt}. "
        "Source measurements, figure files, and selected image IDs accompany this report.",
        "small",
    )
    SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        title="Attention Pix2Pix Training and Test Report",
        author="Aegis Perception",
        leftMargin=40,
        rightMargin=40,
        topMargin=34,
        bottomMargin=47,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    (OUT / "report_provenance.json").write_text(
        json.dumps(
            {
                "complete": complete,
                "full_metrics": have_metrics,
                "full_history": have_history,
                "restored_previews": previews is not None,
                "examples": visual_provenance,
                "sources": {
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [
                        manifest,
                        OUT / "log_evidence.json",
                        *([RUN / "metrics_test.json"] if have_metrics else []),
                        *([RUN / "history.json"] if have_history else []),
                    ]
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(destination)


if __name__ == "__main__":
    main()
