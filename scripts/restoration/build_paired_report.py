#!/usr/bin/env python3
"""Build a compact, two-model report from logged results and frozen local configs."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
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

OUT = ROOT / "artifacts/reports/attention_pix2pix_restormer_20261021"
WEATHERS = ("rain", "fog", "snow", "smoke")
NAMES = ("attention_pix2pix", "restormer")
INK, TEAL, ORANGE = "#173447", "#147e91", "#bd613b"
WIDTH = A4[0] - 80


def main():
    evidence_path = OUT / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    results = evidence["models"]
    config_paths = [ROOT / f"configs/restoration/baselines_v1/{n}.yaml" for n in NAMES]
    configs = [yaml.safe_load(p.read_text()) for p in config_paths]
    assert configs[0]["data"] == configs[1]["data"]
    manifest = ROOT / configs[0]["data"]["manifest"]
    records = load_manifest(manifest)
    counts = Counter((r.weather, r.split) for r in records)
    for name in NAMES:
        result = results[name]
        assert sum(r["images"] for r in result["counts"].values()) == 2334
        for w in WEATHERS:
            assert result["counts"][w]["images"] == counts[w, "test"]
        h = result["headline"]
        for key in ("psnr_db", "ssim"):
            gain = "psnr_gain_db" if key == "psnr_db" else "ssim_gain"
            assert abs(h[key] - h[f"raw_{key}"] - h[gain]) < 1e-10
        assert configs[NAMES.index(name)]["train"]["epochs"] == 60
    for key in ("raw_psnr_db", "raw_ssim"):
        assert results[NAMES[0]]["headline"][key] == results[NAMES[1]]["headline"][key]

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "text.color": INK}
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.2))
    for ax, key, label in zip(axes, ("psnr_db", "ssim"), ("PSNR (dB)", "SSIM"), strict=True):
        values = [
            results[NAMES[0]]["headline"][f"raw_{key}"],
            *[results[n]["headline"][key] for n in NAMES],
        ]
        ax.bar(
            ["Input", "Attention GAN", "Restormer"],
            values,
            color=["#b2c0c8", ORANGE, TEAL],
            width=0.62,
        )
        ax.set_title(label + " · equal-weather test score", loc="left", fontweight="bold")
        ax.set_ylim(0, 1.08 if key == "ssim" else 38)
        for i, v in enumerate(values):
            ax.text(
                i,
                v + (0.018 if key == "ssim" else 0.6),
                f"{v:.4f}" if key == "ssim" else f"{v:.2f}",
                ha="center",
            )
    fig.tight_layout()
    fig.savefig(OUT / "test_comparison.png", dpi=200)
    plt.close(fig)

    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    for name, file in [("Body", "DejaVuSans.ttf"), ("Bold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold")
    styles = {
        "title": ParagraphStyle(
            "title",
            fontName="Bold",
            fontSize=22,
            leading=27,
            textColor=colors.HexColor(INK),
            spaceAfter=8,
        ),
        "heading": ParagraphStyle(
            "heading",
            fontName="Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor(TEAL),
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
            fontSize=7.6,
            leading=10.8,
            textColor=colors.HexColor("#4c6372"),
            spaceAfter=6,
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
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2eef2")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f3f7f9")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend([t, Spacer(1, 6)])

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#cbd8df"))
        canvas.line(40, 35, A4[0] - 40, 35)
        canvas.setFont("Body", 7)
        canvas.drawString(
            40, 22, "AEGIS PERCEPTION | Frozen four-weather benchmark | 21 October 2026"
        )
        canvas.drawRightString(A4[0] - 40, 22, f"{doc.page} / 2")

    add("Four-weather restoration<br/>Attention Pix2Pix &amp; Restormer", "title")
    add("Completed training and test evaluation · Seed 42 · One A100 · 21 October 2026", "small")
    add(
        "<b>Purpose.</b> Establish a common, reproducible dataset protocol for comparing "
        "restoration models. Both runs completed 60 epochs and were evaluated on the same "
        "2,334 test pairs. This report covers these two runs only."
    )
    add("One frozen dataset protocol", "heading")
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
        "<b>Reporting rule:</b> average within stratum, then background, then weather; "
        "each weather contributes <b>25%</b> to the headline PSNR and SSIM. Unequal image "
        "counts therefore cannot make snow dominate the reported score. Test images stay "
        "held out; reserve pairs are not recycled into training.",
        "small",
    )
    add("Models and training recipe", "heading")
    table(
        [
            ["Setting", "Attention Pix2Pix", "Restormer"],
            [
                "Architecture",
                "Six-level U-Net, width 64; four additive skip-attention gates; "
                "conditional 94×94 PatchGAN",
                "Transformer; width 48; encoder blocks [4,6,6,8]; "
                "heads [1,2,4,8]; four refinement blocks",
            ],
            ["Initialization", "Random weights", "Official pretrained deraining weights"],
            ["Generator objective", "Adversarial BCE + 100 × L1", "L1 reconstruction"],
            [
                "Optimizer",
                "Adam; betas 0.5/0.999; no weight decay",
                "AdamW; betas 0.9/0.999; weight decay 10⁻⁴",
            ],
            ["Completed jobs", "47624 → 47625 → 47626", "47627 → 47628 → 47629"],
        ],
        [95, (WIDTH - 95) / 2, (WIDTH - 95) / 2],
    )
    add(
        "<b>Shared settings:</b> 60 sampled epochs, 4,096 draws/epoch, batch 4 "
        "(61,440 updates/model); paired 256×256 crops and flips; BF16; gradient clipping at 1; "
        "learning rate 10⁻⁴ with cosine decay to 10⁻⁶. Seed 42. Weather sampling "
        "rain/fog/snow/smoke = 30/30/30/10%, followed by uniform background and variant "
        "selection, with replacement.",
        "small",
    )
    add(
        "<b>Selection and evaluation:</b> each model saves its own highest balanced-validation-"
        "PSNR checkpoint. SSIM is tracked, but does not select the checkpoint. After training, "
        "that checkpoint is tested at native image resolution using RGB PSNR/SSIM. "
        "Scores are computed against clean references; “input” means the unprocessed "
        "weather-affected image.",
        "small",
    )
    add(
        "<b>Scope:</b> matched data and update budgets support comparison of these training "
        "recipes. Different initialization, objectives, and capacity mean this is not an "
        "isolated architecture or attention ablation.",
        "small",
    )

    story.append(PageBreak())
    add("Test results and interpretation", "title")
    hgan, hrest = [results[n]["headline"] for n in NAMES]
    add(
        f"<b>Restormer leads this comparison:</b> +{hrest['psnr_db'] - hgan['psnr_db']:.2f} dB "
        f"PSNR and +{hrest['ssim'] - hgan['ssim']:.4f} SSIM over Attention Pix2Pix on "
        "the same balanced test benchmark."
    )
    story.append(Image(str(OUT / "test_comparison.png"), width=WIDTH, height=139))
    table(
        [
            ["Model / input", "PSNR (dB)", "SSIM", "ΔPSNR vs input", "ΔSSIM vs input"],
            ["Degraded input", f"{hgan['raw_psnr_db']:.2f}", f"{hgan['raw_ssim']:.4f}", "—", "—"],
            *[
                [
                    label,
                    f"{h['psnr_db']:.2f}",
                    f"{h['ssim']:.4f}",
                    f"+{h['psnr_gain_db']:.2f}",
                    f"+{h['ssim_gain']:.4f}",
                ]
                for label, h in [("Attention Pix2Pix", hgan), ("Restormer", hrest)]
            ],
        ],
        [125, 80, 75, 125, WIDTH - 405],
    )
    add("Where restoration makes individual images worse", "heading")
    table(
        [["Weather", "Pairs", "GAN: PSNR / SSIM lower", "Restormer: PSNR / SSIM lower"]]
        + [
            [
                w.title(),
                counts[w, "test"],
                *[
                    f"{results[n]['counts'][w]['psnr_worse_than_raw']} / "
                    f"{results[n]['counts'][w]['ssim_worse_than_raw']}"
                    for n in NAMES
                ],
            ]
            for w in WEATHERS
        ],
        [70, 55, 190, WIDTH - 315],
    )
    add(
        "Counts compare each output with its own degraded input against the same clean target. "
        "PSNR and SSIM columns can refer to different images; they must not be added. "
        "These are failure counts, not per-weather mean PSNR/SSIM.",
        "small",
    )
    add(
        "<b>Strongest finding:</b> neither model lowers PSNR or SSIM on the 768 snow test pairs. "
        "Restormer also has no measured regressions on 1,000 fog pairs; rain has zero "
        "PSNR regressions and one SSIM regression."
    )
    add(
        "<b>Remaining weakness — smoke:</b> Attention Pix2Pix lowers PSNR on 52.3% and "
        "SSIM on 99.7% of smoke pairs. Restormer reduces those rates to <b>7.9%</b> "
        "and <b>10.8%</b>, respectively. This is a substantial improvement over the GAN, "
        "but smoke restoration is not fully solved."
    )
    add("Conclusion and evidence limits", "heading")
    add(
        "Use the frozen split as the common benchmark for subsequent models. Restormer is "
        "the stronger measured baseline of these two runs; retain Attention Pix2Pix as "
        "the adversarial baseline. Preserve the split, scoring rules, and per-weather reporting "
        "rather than changing them after seeing a test result.",
        "small",
    )
    add(
        "Single seed; only six smoke test background groups; previously inspected test set. "
        "No claim of statistical significance, new-domain generalization, visual superiority, "
        "detector gains, or Jetson performance follows from these scores.",
        "small",
    )
    add(
        "<b>Provenance:</b> exact test scores and failure counts are transcribed from the supplied "
        "DGX completion logs; settings and split counts were checked against local configs and "
        "the frozen manifest. Remote JSON results/checkpoints were not independently retrieved. "
        "Full curves, per-weather mean scores, confidence intervals, and restored-image panels "
        "are not included. Charts show logged measurements, not reconstructed predictions.",
        "small",
    )

    output = OUT / "attention_pix2pix_restormer_report.pdf"
    SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=33,
        bottomMargin=47,
        title="Attention Pix2Pix and Restormer — Four-weather benchmark",
        author="Aegis Perception",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    sources = [manifest, *config_paths, evidence_path, Path(__file__)]
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "report_date": "2026-10-21",
                "scope": list(NAMES),
                "remote_artifacts_independently_retrieved": False,
                "sources": {
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
