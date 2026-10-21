#!/usr/bin/env python3
"""Generate a three-page, artifact-verified comparison of the completed models."""

from __future__ import annotations

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

OUT = ROOT / "artifacts/reports/three_models_20261021"
NAMES = ("gan", "lite", "restormer")
LABELS = {"gan": "Attention Pix2Pix", "lite": "LiteWeather v2", "restormer": "Restormer"}
PALETTE = {"gan": "#bd613b", "lite": "#177e9d", "restormer": "#6056a5"}
WEATHERS = ("rain", "fog", "snow", "smoke")
WIDTH = A4[0] - 76


def main():
    data = json.loads((OUT / "verified_report_data.json").read_text())
    results, histories, audit = data["results"], data["histories"], data["audit"]
    manifest = ROOT / "data/manifests/weather_v3/active.csv"
    counts = Counter((r.weather, r.split) for r in load_manifest(manifest))
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 2.3))
    for ax, key, title in zip(
        axes,
        ("val_macro_scene_psnr_db", "val_macro_scene_ssim"),
        ("Validation PSNR (dB)", "Validation SSIM"),
        strict=True,
    ):
        for name in NAMES:
            h = histories[name]
            ax.plot(
                [r["epoch"] for r in h],
                [r[key] for r in h],
                color=PALETTE[name],
                label=LABELS[name],
                linewidth=1.3,
            )
            best = results[name]["checkpoint_epoch"]
            ax.scatter([best], [h[best - 1][key]], color=PALETTE[name], marker="*", s=60, zorder=5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.15)
        ax.set_xlim(1, 60)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=200)
    plt.close(fig)

    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    for name, file in [("Body", "DejaVuSans.ttf"), ("Bold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold")
    styles = {
        "title": ParagraphStyle(
            "title",
            fontName="Bold",
            fontSize=21,
            leading=25,
            textColor=colors.HexColor("#173447"),
            spaceAfter=8,
        ),
        "heading": ParagraphStyle(
            "heading",
            fontName="Bold",
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#147e91"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Body",
            fontSize=9,
            leading=12.5,
            textColor=colors.HexColor("#173447"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="Body",
            fontSize=7.3,
            leading=10,
            textColor=colors.HexColor("#496171"),
            spaceAfter=5,
        ),
        "tiny": ParagraphStyle(
            "tiny",
            fontName="Body",
            fontSize=6.5,
            leading=8.5,
            textColor=colors.HexColor("#496171"),
            spaceAfter=4,
        ),
    }
    story = []

    def p(text, style="body"):
        return Paragraph(str(text), styles[style])

    def add(text, style="body"):
        story.append(p(text, style))

    def ranked(value, values, display, lower=False):
        order = sorted(set(values), reverse=not lower)
        if value == order[0]:
            return f'<font color="#075e49"><b>{display}</b></font>'
        if len(order) > 1 and value == order[1]:
            return f'<font color="#203f63"><u>{display}</u></font>'
        return display

    def cell_rank(value, values, lower=False):
        return value == (min(values) if lower else max(values))

    def table(rows, widths, compact=False, best_cells=()):
        t = Table(
            [
                [
                    p(
                        f'<font color="#ffffff"><b>{c}</b></font>' if i == 0 else c,
                        "tiny" if compact else "small",
                    )
                    for c in row
                ]
                for i, row in enumerate(rows)
            ],
            colWidths=widths,
            hAlign="LEFT",
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173447")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f3f7f9")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#147e91")),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#cbd8df")),
                    *[
                        ("BACKGROUND", cell, cell, colors.HexColor("#e1f2e9"))
                        for cell in best_cells
                    ],
                ]
            )
        )
        story.extend([t, Spacer(1, 5)])

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#cbd8df"))
        canvas.line(38, 34, A4[0] - 38, 34)
        canvas.setFont("Body", 7)
        canvas.drawString(
            38, 21, "AEGIS PERCEPTION | Three-model restoration benchmark | 21 October 2026"
        )
        canvas.drawRightString(A4[0] - 38, 21, f"{doc.page} / 3")

    add("Three-model restoration benchmark", "title")
    add("Attention Pix2Pix · LiteWeatherTransformer v2 · Restormer", "body")
    add("Completed 60-epoch runs · Frozen four-weather split · Seed 42 · One A100 per job", "small")
    add(
        "<b>Main finding:</b> Restormer has the highest test PSNR/SSIM in every weather. "
        "LiteWeather outperforms the attention GAN in every weather on both mean metrics, "
        "with far fewer generator parameters. Smoke remains the weakest condition; "
        "<b>LiteWeather still reduces mean smoke SSIM relative to the input.</b>"
    )
    add("1. Shared data and scoring protocol", "heading")
    table(
        [["Weather / source", "Train", "Validation", "Test", "Test backgrounds"]]
        + [
            [
                f"{w.title()} / {source}",
                *[f"{counts[w, s]:,}" for s in ("train", "val", "test")],
                str(results["gan"]["counts"][w]["background_groups"]),
            ]
            for w, source in zip(
                WEATHERS,
                ("RealRain-1k-H", "RESIDE-6K", "SnowCityscapes", "SmokeBench"),
                strict=True,
            )
        ]
        + [["Total", "18,610", "2,011", "2,334", "1,028"]],
        [180, 70, 85, 70, WIDTH - 405],
    )
    add(
        "Metrics compare restored RGB images with paired clean references at native resolution "
        "(DGX BF16 inference; FP32 metric calculation). The headline averages images within "
        "strata, strata within background, backgrounds within weather, then assigns each weather "
        "<b>25% weight</b>. Test counts therefore cannot make snow dominate the headline. "
        "Known-group/target leakage checks and frozen manifests are shared across all runs.",
        "small",
    )
    add("2. Architecture, initialization and training", "heading")
    table(
        [["Model", "Architecture / initialization", "Parameters ↓"]]
        + [
            [
                LABELS[n],
                desc,
                ranked(
                    audit["models"][n]["parameters"],
                    [audit["models"][k]["parameters"] for k in NAMES],
                    f"{audit['models'][n]['parameters'] / 1e6:.2f} M",
                    lower=True,
                ),
            ]
            for n, desc in [
                (
                    "gan",
                    "Six-level U-Net, four additive skip gates; conditional 94×94 PatchGAN. "
                    "Random initialization.",
                ),
                (
                    "lite",
                    "Custom compact Transformer; channel attention at 1/4 and 1/8 scale, "
                    "content-conditioned prompts. Random initialization.",
                ),
                (
                    "restormer",
                    "Transformer width 48, blocks [4,6,6,8], four refinement blocks. "
                    "Official pretrained deraining initialization.",
                ),
            ]
        ],
        [110, WIDTH - 200, 90],
        best_cells=[
            (2, i + 1)
            for i, n in enumerate(NAMES)
            if cell_rank(
                audit["models"][n]["parameters"],
                [audit["models"][k]["parameters"] for k in NAMES],
                lower=True,
            )
        ],
    )
    add(
        "<b>Shared budget:</b> 60 epochs × 4,096 sampled draws, batch 4 = <b>61,440 updates</b> "
        "per model; paired 256×256 crops/flips. Sampling with replacement: rain/fog/snow/smoke "
        "= 30/30/30/10%, then uniform background and variant. Learning rate 10⁻⁴, cosine decay "
        "to 10⁻⁶, BF16, gradient clip 1.",
        "small",
    )
    add(
        "<b>Objectives:</b> GAN: adversarial BCE + 100×L1, Adam (0.5,0.999), no decay. "
        "LiteWeather and Restormer: L1, AdamW (0.9,0.999), weight decay 10⁻⁴. "
        "Each model selects its own highest balanced-validation-PSNR checkpoint; SSIM is logged, "
        "not optimized as a separate loss or used for selection.",
        "small",
    )
    add("3. Size and measured run cost", "heading")
    table(
        [["Model", "Selected epoch", "best.pt size ↓", "Train + validation time ↓"]]
        + [
            [
                LABELS[n],
                results[n]["checkpoint_epoch"],
                ranked(
                    audit["models"][n]["best_pt_bytes"],
                    [audit["models"][k]["best_pt_bytes"] for k in NAMES],
                    f"{audit['models'][n]['best_pt_bytes'] / 2**20:.1f} MiB",
                    lower=True,
                ),
                ranked(
                    audit["models"][n]["epoch_train_val_hours"],
                    [audit["models"][k]["epoch_train_val_hours"] for k in NAMES],
                    f"{audit['models'][n]['epoch_train_val_hours']:.2f} h",
                    lower=True,
                ),
            ]
            for n in NAMES
        ],
        [125, 90, 110, WIDTH - 325],
        best_cells=[
            (j, i + 1)
            for i, n in enumerate(NAMES)
            for j, key in ((2, "best_pt_bytes"), (3, "epoch_train_val_hours"))
            if cell_rank(
                audit["models"][n][key], [audit["models"][k][key] for k in NAMES], lower=True
            )
        ],
    )
    add(
        '<b>Reading the tables:</b> <font color="#075e49"><b>bold green</b></font> = best; '
        "<u>underlined</u> = next distinct result. Ties share a rank. "
        "↑ higher is better; ↓ lower is better. Epoch and dataset counts are not ranked.",
        "tiny",
    )
    ratio = audit["models"]["restormer"]["parameters"] / audit["models"]["lite"]["parameters"]
    add(
        f"LiteWeather has <b>{ratio:.1f}× fewer parameters</b> than Restormer. Checkpoint files "
        "include optimizer/training state; their size is not inference RAM. Times sum recorded "
        "epoch training + validation, excluding queueing, setup, initial validation "
        "and final test. "
        "No Jetson FPS, quantization, or YOLO benefit has been measured. Different objectives and "
        "pretraining make this a recipe comparison, not a controlled architecture-only ablation.",
        "small",
    )

    story.append(PageBreak())
    add("Accuracy, convergence and failures", "title")
    add(
        '<font color="#075e49"><b>Bold green = best</b></font> · '
        "<u>Underline = runner-up</u> · Rank each metric among the three models; "
        "input is a reference. Ties share rank; highlighting is not a significance claim.",
        "tiny",
    )
    table(
        [["Model / input", "PSNR ↑ dB [95% interval]", "SSIM ↑ [95% interval]"]]
        + [
            [
                LABELS[n],
                ranked(
                    r["headline"]["psnr_db"],
                    [results[k]["headline"]["psnr_db"] for k in NAMES],
                    f"{r['headline']['psnr_db']:.2f}",
                )
                + " "
                + f"[{r['uncertainty']['macro_weather']['psnr_db']['percentile_95'][0]:.2f}, "
                f"{r['uncertainty']['macro_weather']['psnr_db']['percentile_95'][1]:.2f}]",
                ranked(
                    r["headline"]["ssim"],
                    [results[k]["headline"]["ssim"] for k in NAMES],
                    f"{r['headline']['ssim']:.4f}",
                )
                + " "
                + f"[{r['uncertainty']['macro_weather']['ssim']['percentile_95'][0]:.4f}, "
                f"{r['uncertainty']['macro_weather']['ssim']['percentile_95'][1]:.4f}]",
            ]
            for n in NAMES
            for r in [results[n]]
        ]
        + [["Degraded input", "18.62", "0.7135"]],
        [125, 190, WIDTH - 315],
        best_cells=[
            (j, i + 1)
            for i, n in enumerate(NAMES)
            for j, key in ((1, "psnr_db"), (2, "ssim"))
            if cell_rank(results[n]["headline"][key], [results[k]["headline"][key] for k in NAMES])
        ],
    )
    add(
        "Intervals: 2,000 cluster-bootstrap repetitions, conditional on this model/test set. "
        "They exclude training-seed uncertainty and are not simultaneous significance tests. "
        "Smoke has only six clusters; its uncertainty estimates are exploratory.",
        "tiny",
    )
    story.append(Image(str(OUT / "training_curves.png"), width=WIDTH, height=137))
    add(
        "Full 60-epoch validation histories; stars mark the checkpoint selected for final testing.",
        "tiny",
    )
    add("Per-weather test means — PSNR ↑ dB / SSIM ↑", "heading")
    table(
        [["Weather", "Input", "Attention Pix2Pix", "LiteWeather v2", "Restormer"]]
        + [
            [
                w.title(),
                f"{results['gan']['scene_balanced_summary'][w]['raw_psnr_db']:.2f} / "
                f"{results['gan']['scene_balanced_summary'][w]['raw_ssim']:.4f}",
                *[
                    ranked(
                        results[n]["scene_balanced_summary"][w]["psnr_db"],
                        [results[k]["scene_balanced_summary"][w]["psnr_db"] for k in NAMES],
                        f"{results[n]['scene_balanced_summary'][w]['psnr_db']:.2f}",
                    )
                    + " / "
                    + ranked(
                        results[n]["scene_balanced_summary"][w]["ssim"],
                        [results[k]["scene_balanced_summary"][w]["ssim"] for k in NAMES],
                        f"{results[n]['scene_balanced_summary'][w]['ssim']:.4f}",
                    )
                    for n in NAMES
                ],
            ]
            for w in WEATHERS
        ],
        [62, 103, 126, 114, WIDTH - 405],
        compact=True,
        best_cells=[
            (j + 2, i + 1)
            for i, w in enumerate(WEATHERS)
            for j, n in enumerate(NAMES)
            if any(
                cell_rank(
                    results[n]["scene_balanced_summary"][w][key],
                    [results[k]["scene_balanced_summary"][w][key] for k in NAMES],
                )
                for key in ("psnr_db", "ssim")
            )
        ],
    )
    add("Images worse than input — PSNR count ↓ / SSIM count ↓", "heading")
    table(
        [["Weather", "Test pairs", "Attention Pix2Pix", "LiteWeather v2", "Restormer"]]
        + [
            [
                w.title(),
                counts[w, "test"],
                *[
                    ranked(
                        results[n]["counts"][w]["psnr_worse_than_raw"],
                        [results[k]["counts"][w]["psnr_worse_than_raw"] for k in NAMES],
                        str(results[n]["counts"][w]["psnr_worse_than_raw"]),
                        lower=True,
                    )
                    + " / "
                    + ranked(
                        results[n]["counts"][w]["ssim_worse_than_raw"],
                        [results[k]["counts"][w]["ssim_worse_than_raw"] for k in NAMES],
                        str(results[n]["counts"][w]["ssim_worse_than_raw"]),
                        lower=True,
                    )
                    for n in NAMES
                ],
            ]
            for w in WEATHERS
        ],
        [65, 75, 125, 125, WIDTH - 390],
        compact=True,
        best_cells=[
            (j + 2, i + 1)
            for i, w in enumerate(WEATHERS)
            for j, n in enumerate(NAMES)
            if any(
                cell_rank(
                    results[n]["counts"][w][key],
                    [results[k]["counts"][w][key] for k in NAMES],
                    lower=True,
                )
                for key in ("psnr_worse_than_raw", "ssim_worse_than_raw")
            )
        ],
    )
    add(
        "PSNR and SSIM counts may overlap and must not be added. These counts are image-weighted; "
        "the mean scores above are background/stratum-balanced. Frames are not independent scenes.",
        "tiny",
    )
    add(
        "<b>Smoke:</b> GAN mean change = −0.36 dB / −0.0677 SSIM; LiteWeather = "
        "+0.77 dB / <b>−0.0178 SSIM</b>; Restormer = +2.34 dB / +0.0351 SSIM. "
        "LiteWeather loses SSIM on 249/342 smoke images (72.8%); Restormer on 37/342 (10.8%). "
        "Higher overall accuracy does not mean the smoke problem is solved.",
        "small",
    )

    story.append(PageBreak())
    add("Matched restoration examples", "title")
    add(
        "The same four test images are processed by each verified best checkpoint. "
        "Columns show degraded input, GAN, LiteWeather, Restormer and clean reference.",
        "small",
    )
    columns = [
        ("Input", "input"),
        ("Attention GAN", "gan"),
        ("LiteWeather v2", "lite"),
        ("Restormer", "restormer"),
        ("Clean reference", "target"),
    ]
    for example in audit["examples"]:
        w = example["weather"]
        add(escape(f"{w.title()} · {example['sample_id']}"), "tiny")
        images = []
        for _label, suffix in columns:
            path = OUT / f"{w}_{suffix}.png"
            with PILImage.open(path) as im:
                factor = min((WIDTH / 5 - 7) / im.width, 91 / im.height)
                images.append(Image(str(path), width=im.width * factor, height=im.height * factor))
        t = Table([[p(label, "tiny") for label, _ in columns], images], colWidths=[WIDTH / 5] * 5)
        t.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        story.extend([t, Spacer(1, 8)])
    add(
        "Selection is score-blind: lowest SHA-256 of sample ID per weather among test images "
        "≤500,000 pixels, limiting CPU preview cost. Examples are not a representative visual "
        "survey. IDs preserve source-folder names, even when the frozen split assigns them to "
        "test. All restored images use native-resolution CPU FP32 inference; displayed images "
        "are scaled with aspect ratio preserved. Official tables use saved DGX BF16 scores, "
        "so preview pixels can differ slightly.",
        "tiny",
    )
    add("Conclusion", "heading")
    add(
        "<b>Restormer:</b> strongest accuracy baseline. <b>LiteWeather:</b> compact candidate "
        "that beats the GAN, but trades accuracy for size and needs smoke improvement. "
        "<b>Attention Pix2Pix:</b> adversarial baseline; positive overall gains hide mean "
        "smoke degradation. Preserve this frozen protocol for future models.",
        "small",
    )
    add("Verified evidence", "heading")
    add(
        "Downloaded best.pt SHA-256 hashes match metrics_test.json for all three models. "
        "All 60-epoch histories, selected epochs, 2,334 test IDs, weather/group/stratum labels, "
        "raw scores and scoring protocols were checked; headline and per-weather means were "
        "recomputed from saved per-image scores. Full metrics, source hashes, configs, "
        "parameter counts, selected image IDs and CPU preview scores accompany this PDF in "
        "verified_report_data.json. Earlier Restormer runs are outside this report.",
        "tiny",
    )
    add(
        "Limits: one seed, previously inspected benchmark, six smoke backgrounds, different "
        "pretraining/objectives; no proof of new-domain generalization or deployment performance.",
        "tiny",
    )
    output = OUT / "three_model_restoration_report_highlighted.pdf"
    SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=38,
        rightMargin=38,
        topMargin=31,
        bottomMargin=46,
        title="Three-model restoration benchmark",
        author="Aegis Perception",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    main()
