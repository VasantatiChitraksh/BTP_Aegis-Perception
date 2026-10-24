#!/usr/bin/env python3
"""Build a verified benchmark PDF with a matched visual comparison appendix."""

import argparse
import csv
import json
import math
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from report_six_models import ROOT, WEATHERS, collect
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from aegis_perception.data.catalog import sha256

OUT = ROOT / "artifacts/reports/six_models_20261024"
INK, TEAL, MUTED, PAPER = "#172F40", "#007E78", "#526774", "#F5F3EE"
PALETTE = ("#B85C38", "#4F56A5", "#138DA2", "#B13D6A", "#76892E", "#BB8317")
W, H = A4
M, WIDTH = 38, A4[0] - 76


def pdf_image(path, width, height):
    """Embed a 216-dpi display thumbnail; preserve native PNG evidence separately."""
    with PILImage.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((math.ceil(width * 3), math.ceil(height * 3)), PILImage.Resampling.LANCZOS)
        return ImageReader(image)


class Report:
    def __init__(self, path, complete, pages=4):
        for name, file in (("Body", "DejaVuSans.ttf"), ("Bold", "DejaVuSans-Bold.ttf")):
            pdfmetrics.registerFont(TTFont(name, "/usr/share/fonts/truetype/dejavu/" + file))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold")
        self.c = canvas.Canvas(str(path), pagesize=A4)
        self.c.setTitle("AEGIS | Six models. Four weather conditions.")
        self.c.setAuthor("AEGIS Perception | artifact-verified benchmark")
        self.complete = complete
        self.pages = pages

    def text(self, x, top, value, size=9, color=INK, bold=False):
        self.c.setFillColor(colors.HexColor(color))
        self.c.setFont("Bold" if bold else "Body", size)
        self.c.drawString(x, H - top - size, str(value))

    def paragraph(self, top, value, x=M, width=WIDTH, size=9, height=100, color=INK):
        p = Paragraph(
            value,
            ParagraphStyle(
                "p",
                fontName="Body",
                fontSize=size,
                leading=size * 1.45,
                textColor=colors.HexColor(color),
            ),
        )
        _, used = p.wrap(width, height)
        if used > height:
            raise ValueError(f"Report text overflows its panel: {value[:70]}")
        p.drawOn(self.c, x, H - top - used)

    def box(self, x, top, width, height, color):
        self.c.setFillColor(colors.HexColor(color))
        self.c.rect(x, H - top - height, width, height, fill=1, stroke=0)

    def page(self, n, title, subtitle):
        self.box(0, 0, W, H, PAPER)
        self.box(0, 0, W, 8, TEAL)
        self.text(M, 29, "AEGIS  /  RESTORATION BENCHMARK", 8, TEAL, True)
        self.text(W - 120, 29, f"24 OCT 2026   /   {n:02}", 7, MUTED)
        self.text(M, 56, title, 25, INK, True)
        self.paragraph(94, subtitle, size=9, height=40, color=MUTED)
        self.c.setStrokeColor(colors.HexColor("#D8DEDC"))
        self.c.line(M, 43, W - M, 43)
        status = "VERIFIED FINAL RESULTS" if self.complete else "PREVIEW / DOWNLOAD INCOMPLETE"
        self.text(M, H - 33, status, 7, TEAL if self.complete else "#B85C38", True)
        self.text(W - 169, H - 33, f"Frozen weather-v3  |  {n} / {self.pages}", 7, MUTED)

    def section(self, top, label):
        self.text(M, top, label.upper(), 9, TEAL, True)

    def table(self, top, headers, rows, widths, best=(), lower=(), row_height=24):
        self.box(M, top, WIDTH, row_height, INK)
        x = M
        for label, width in zip(headers, widths, strict=True):
            self.text(x + 7, top + 7, label, 7.2, "#FFFFFF", True)
            x += width
        winners = {}
        for col in (*best, *lower):
            values = [float(r[col]) for r in rows if r[col] != "—"]
            if values:
                winners[col] = (min if col in lower else max)(values)
        for i, row in enumerate(rows):
            y = top + (i + 1) * row_height
            self.box(M, y, WIDTH, row_height, "#FFFFFF" if i % 2 == 0 else "#EDEFEA")
            x = M
            for j, (value, width) in enumerate(zip(row, widths, strict=True)):
                win = j in winners and value != "—" and float(value) == winners[j]
                if win:
                    self.box(x + 1, y + 1, width - 2, row_height - 2, "#D3EBDF")
                self.text(x + 7, y + 7, value, 7.7, TEAL if win else INK, win or j == 0)
                x += width

    def line_chart(self, models, top, key, title, palette):
        left, right, bottom, height = M + 39, W - M - 8, H - top - 151, 117
        values = [r[key] for m in models for r in m["history"]]
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * 0.09, 0.001)
        lo, hi = lo - pad, hi + pad
        self.text(M, top, title, 10, INK, True)
        for i in range(4):
            y = bottom + i * height / 3
            self.c.setStrokeColor(colors.HexColor("#D8DEDC"))
            self.c.line(left, y, right, y)
            self.text(
                M,
                H - y - 5,
                f"{lo + i * (hi - lo) / 3:.3f}"
                if "ssim" in key
                else f"{lo + i * (hi - lo) / 3:.1f}",
                7,
                MUTED,
            )
        for m in models:
            color = palette[m["label"]]
            self.c.setStrokeColor(colors.HexColor(color))
            self.c.setLineWidth(1.5)
            path = self.c.beginPath()
            for i, r in enumerate(m["history"]):
                x, y = (
                    left + (r["epoch"] - 1) * (right - left) / 59,
                    bottom + (r[key] - lo) * height / (hi - lo),
                )
                (path.moveTo if i == 0 else path.lineTo)(x, y)
                if r["epoch"] == m["metrics"]["checkpoint_epoch"]:
                    chosen = x, y
            self.c.drawPath(path)
            self.c.setFillColor(colors.HexColor(color))
            self.c.circle(*chosen, 3, fill=1, stroke=0)
        for epoch in (1, 20, 40, 60):
            self.text(left + (epoch - 1) * (right - left) / 59 - 4, H - bottom + 8, epoch, 7, MUTED)


def render(data, path):
    all_models = list(data["models"].values())
    models = [m for m in all_models if m["status"] == "completed"]
    if not models:
        raise ValueError("No verified results available")
    complete = len(models) == 6
    example_dir = OUT / "examples"
    example_audit = example_dir / "audit.json"
    gallery = json.loads(example_audit.read_text()) if example_audit.exists() else None
    if gallery and set(gallery["models"]) != set(data["models"]):
        gallery = None
    if gallery:
        for name, record in gallery["models"].items():
            if record["checkpoint_sha256"] != data["models"][name]["metrics"]["checkpoint_sha256"]:
                raise ValueError(f"Gallery checkpoint mismatch: {name}")
            for file, digest in record["files"].items():
                if sha256(example_dir / file) != digest:
                    raise ValueError(f"Gallery image changed: {file}")
    report = Report(path, complete, 5 if gallery else 4)
    c = report.c
    palette = {m["label"]: PALETTE[i] for i, m in enumerate(all_models)}
    ranked = sorted(models, key=lambda m: m["metrics"]["headline"]["psnr_db"], reverse=True)
    winner = ranked[0]
    ssim_winner = max(models, key=lambda m: m["metrics"]["headline"]["ssim"])
    smallest = min(models, key=lambda m: m["parameters"])

    def headline(m):
        return m.get("metrics", {}).get("headline", {})

    def fmt(m, k, d):
        return f"{headline(m)[k]:.{d}f}" if headline(m) else "—"

    report.page(
        1,
        "Six models. One benchmark.",
        f"{len(models)}/6 completed runs verified locally. "
        "One frozen benchmark; validation-selected checkpoints. "
        "Observed test quality, with failures and model size kept visible.",
    )
    cards = (
        ("HIGHEST TEST PSNR", winner, fmt(winner, "psnr_db", 2) + " dB"),
        ("HIGHEST TEST SSIM", ssim_winner, fmt(ssim_winner, "ssim", 4)),
        ("FEWEST PARAMETERS", smallest, f"{smallest['parameters'] / 1e6:.3f} M"),
    )
    for i, (label, model, value) in enumerate(cards):
        x, width = M + i * (WIDTH + 10) / 3, (WIDTH - 20) / 3
        report.box(x, 143, width, 77, INK if i == 0 else "#FFFFFF")
        report.text(x + 12, 154, label, 6.8, "#A5DAD1" if i == 0 else TEAL, True)
        report.text(x + 12, 174, value, 21, "#FFFFFF" if i == 0 else INK, True)
        report.text(x + 12, 204, model["label"], 8, "#FFFFFF" if i == 0 else MUTED)
    report.section(243, "01 / Overall test quality")
    ordered = ranked + [m for m in all_models if m["status"] != "completed"]
    rows = [
        [
            m["label"],
            fmt(m, "psnr_db", 2),
            fmt(m, "ssim", 4),
            fmt(m, "psnr_gain_db", 2),
            fmt(m, "ssim_gain", 4),
            f"{m['parameters'] / 1e6:.3f}",
        ]
        for m in ordered
    ]
    report.table(
        265,
        ["Model", "PSNR ↑", "SSIM ↑", "ΔPSNR ↑", "ΔSSIM ↑", "Params M ↓"],
        rows,
        [139, 65, 65, 72, 72, WIDTH - 413],
        best=(1, 2, 3, 4),
        lower=(5,),
    )
    report.paragraph(
        447,
        "<b>Read the table:</b> green bold cells mark the best displayed value in each column; "
        "ties share highlighting. Deltas compare restored images with the same degraded inputs. "
        "Each weather contributes 25% of the headline score, regardless of image count.",
        size=8,
        height=50,
    )
    report.section(518, "02 / Quality versus parameter count")
    left, right, bottom, height = M + 35, W - M - 35, H - 693, 125
    lo = math.floor(min(headline(m)["psnr_db"] for m in models) - 1)
    hi = math.ceil(max(headline(m)["psnr_db"] for m in models) + 1)
    for tick in range(lo, hi + 1, max(1, math.ceil((hi - lo) / 4))):
        y = bottom + (tick - lo) * height / (hi - lo)
        c.setStrokeColor(colors.HexColor("#D8DEDC"))
        c.line(left, y, right, y)
        report.text(M + 8, H - y - 4, tick, 7, MUTED)
    for tick in (1, 2, 5, 10, 20, 40):
        x = left + (math.log10(tick) + 0.1) / 1.8 * (right - left)
        report.text(x - 5, H - bottom + 8, tick, 7, MUTED)
    for i, m in enumerate(models):
        x = left + (math.log10(m["parameters"] / 1e6) + 0.1) / 1.8 * (right - left)
        y = bottom + (headline(m)["psnr_db"] - lo) * height / (hi - lo)
        c.setFillColor(colors.HexColor(palette[m["label"]]))
        c.circle(x, y, 5, fill=1, stroke=0)
        # Numeric keys avoid overlapping long names when models have similar scores.
        report.text(x + 7, H - y - 8, i + 1, 8, palette[m["label"]], True)
    report.text(M + 150, 719, "Parameters (millions, log scale) • PSNR (dB)", 7.5, MUTED)
    report.paragraph(
        742,
        " · ".join(f"{i + 1} {escape(m['label'])}" for i, m in enumerate(models)),
        size=7,
        height=35,
    )
    c.showPage()

    report.page(
        2,
        "Where each model succeeds.",
        "Per-weather scene-balanced metrics. A strong average can still hide damaged images.",
    )
    widths = [155] + [(WIDTH - 155) / 4] * 4
    for top, metric, digits, title in (
        (139, "psnr_db", 2, "03 / PSNR by weather — higher is better"),
        (349, "ssim", 4, "04 / SSIM by weather — higher is better"),
    ):
        report.section(top, title)
        rows = [
            [
                m["label"],
                *[
                    f"{m['metrics']['scene_balanced_summary'][w][metric]:.{digits}f}"
                    if m["status"] == "completed"
                    else "—"
                    for w in WEATHERS
                ],
            ]
            for m in all_models
        ]
        report.table(
            top + 22,
            ["Model", *[w.title() for w in WEATHERS]],
            rows,
            widths,
            best=(1, 2, 3, 4),
            row_height=23,
        )
    report.section(558, "05 / Images made worse — PSNR / SSIM counts ↓")
    rows = []
    for m in all_models:
        values = []
        for w in WEATHERS:
            counts = m.get("metrics", {}).get("counts", {}).get(w)
            values.append(
                f"{counts['psnr_worse_than_raw']} / {counts['ssim_worse_than_raw']}"
                if counts
                else "—"
            )
        rows.append([m["label"], *values])
    report.table(
        581,
        ["Model", "Rain /224", "Fog /1000", "Snow /768", "Smoke /342"],
        rows,
        widths,
        row_height=23,
    )
    report.paragraph(
        754,
        "<b>Smoke deserves separate scrutiny:</b> 342 test images cover "
        "just six independent backgrounds. "
        "PSNR and SSIM regression counts overlap; do not add them together.",
        size=8,
        height=38,
    )
    c.showPage()

    report.page(
        3,
        "Learning curves. Real costs.",
        "Every curve covers 60 epochs. Dots mark the checkpoint selected "
        "by validation PSNR, not by test scores.",
    )
    report.line_chart(
        models, 142, "val_macro_scene_psnr_db", "Validation scene-balanced PSNR (dB)", palette
    )
    report.line_chart(
        models, 335, "val_macro_scene_ssim", "Validation scene-balanced SSIM", palette
    )
    report.paragraph(
        510,
        " · ".join(
            f'<font color="{palette[m["label"]]}">{escape(m["label"])}</font>' for m in models
        ),
        size=7.4,
        height=32,
    )
    report.section(560, "06 / Recorded training and checkpoint footprint")
    rows = []
    for m in all_models:
        if m["status"] != "completed":
            rows.append([m["label"], "—", "—", "—", "—"])
            continue
        files = m.get("local_artifacts", {})
        best = ROOT / m["directory"] / "best.pt"
        size = files.get("best_bytes", best.stat().st_size if best.exists() else 0)
        compact = files.get("inference_bytes")
        rows.append(
            [
                m["label"],
                str(m["metrics"]["checkpoint_epoch"]),
                f"{sum(r['epoch_seconds'] for r in m['history']) / 3600:.2f}",
                f"{size / 2**20:.1f}" if size else "—",
                f"{compact / 2**20:.1f}" if compact else "—",
            ]
        )
    report.table(
        582,
        ["Model", "Best epoch", "Epoch hours", "best.pt MiB", "Inference MiB"],
        rows,
        [155, 77, 91, 96, WIDTH - 419],
        row_height=23,
    )
    report.paragraph(
        753,
        "Epoch hours sum logged training + validation time; exclude "
        "queues, preflight and final test. "
        "Parameter count and checkpoint size do not measure Jetson "
        "latency. INT8 quality, memory and "
        "FPS alongside YOLOv8 remain unmeasured.",
        size=7.5,
        height=39,
    )
    c.showPage()

    report.page(
        4,
        "The evidence behind the scores.",
        "A reproducible protocol, an honest interpretation, and locally traceable weights.",
    )
    report.section(139, "07 / Frozen split")
    report.table(
        161,
        ["Weather / source", "Train", "Validation", "Test"],
        [
            ["Rain / RealRain1k-H", "773", "112", "224"],
            ["Fog / RESIDE6K", "5102", "897", "1000"],
            ["Snow / SnowCityscapes", "5382", "618", "768"],
            ["Smoke / SmokeBench v2", "7353", "384", "342"],
            ["Total", "18610", "2011", "2334"],
        ],
        [218, 100, 100, WIDTH - 418],
        row_height=22,
    )
    report.paragraph(
        306,
        "<b>Shared sampling:</b> seed 42; 4,096 draws/epoch with replacement; rain/fog/snow/smoke "
        "probabilities 0.30/0.30/0.30/0.10; batch 4; paired 256-pixel crops and flips. "
        "60 epochs = 61,440 generator updates. Native-size BF16 "
        "validation and test. Headline scores "
        "average within strata, then backgrounds, then weather classes.",
        size=8,
        height=61,
    )
    report.section(386, "08 / Example degraded inputs and clean targets")
    preview = ROOT / "artifacts/reports/three_models_20261021"
    for i, weather in enumerate(WEATHERS):
        x, width = M + i * (WIDTH + 8) / 4, (WIDTH - 24) / 4
        report.text(x, 407, weather.title(), 8, INK, True)
        for j, kind in enumerate(("input", "target")):
            file = preview / f"{weather}_{kind}.png"
            top = 427 + j * 79
            report.box(x, top, width, 74, "#E1E5E0")
            if file.exists():
                c.drawImage(
                    pdf_image(file, width, 74),
                    x,
                    H - top - 74,
                    width,
                    74,
                    preserveAspectRatio=True,
                    anchor="c",
                )
            else:
                report.text(x + 6, top + 30, "Preview unavailable", 7, MUTED)
    report.paragraph(
        589,
        "Top: degraded input. Bottom: clean target. Fixed score-blind "
        "examples from the prior audit; "
        "lowest sample-ID SHA256 per weather among images ≤500,000 "
        "pixels. Display thumbnails only; "
        "metrics use native resolution. These are dataset pairs, not model predictions.",
        size=7,
        height=42,
    )
    report.section(645, "09 / What the comparison can support")
    report.paragraph(
        667,
        "Attention Pix2Pix and LiteWeather v2 started randomly. Restormer and RAMiT used deraining "
        "weights; TransWeather used All-Weather weights; PromptIR used all-in-one weights. "
        "Unequal pretraining means this is a system comparison, not an isolated architecture test. "
        "Pretraining overlap is not ruled out, particularly for RESIDE fog. Single-seed ranks and "
        "small score gaps do not establish statistical superiority. Cluster-bootstrap intervals "
        "are retained in the metrics JSON; they do not measure training-seed uncertainty.",
        size=8,
        height=89,
    )
    report.paragraph(
        761,
        "<b>Local handoff:</b> artifacts/models/&lt;model&gt;/ contains "
        "best.pt, inference.pt, metrics, "
        "history and SHA256SUMS. Original run folders and earlier "
        "Restormer results remain preserved.",
        size=7,
        height=27,
    )
    if gallery:
        c.showPage()
        gallery_page(c, data, gallery, example_dir)
    c.save()


def gallery_page(c, data, audit, directory):
    """A landscape contact sheet preserves aspect ratios and compares identical scenes."""
    width, height = H, W
    c.setPageSize((width, height))
    c.setFillColor(colors.HexColor(PAPER))
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFillColor(colors.HexColor(TEAL))
    c.rect(0, height - 8, width, 8, fill=1, stroke=0)

    def text(x, top, value, size=8, color=INK, bold=False):
        c.setFillColor(colors.HexColor(color))
        c.setFont("Bold" if bold else "Body", size)
        c.drawString(x, height - top - size, str(value))

    text(28, 25, "AEGIS / VISUAL EVIDENCE", 8, TEAL, True)
    text(28, 45, "Same scene. Six restorations.", 24, INK, True)
    text(
        28,
        80,
        "Fixed, score-blind examples • Native-resolution CPU FP32 "
        "inference • No cherry-picking by model scores",
        8,
        MUTED,
    )
    columns = [
        ("input", "Degraded"),
        *[(n, m["label"]) for n, m in data["models"].items()],
        ("target", "Clean target"),
    ]
    cell_width = (width - 64) / 8
    for col, (_, label) in enumerate(columns):
        text(32 + col * cell_width, 109, label, 7, INK, True)
    for row, example in enumerate(audit["examples"]):
        top, weather = 130 + row * 100, example["weather"]
        text(32, top, weather.upper(), 7, TEAL, True)
        for col, (name, _) in enumerate(columns):
            x = 32 + col * cell_width
            c.drawImage(
                pdf_image(directory / f"{weather}_{name}.png", cell_width - 7, 74),
                x,
                height - top - 89,
                cell_width - 7,
                74,
                preserveAspectRatio=True,
                anchor="c",
            )
    text(
        28,
        549,
        "These four examples illustrate behavior; they are not a "
        "representative visual survey. Numerical rankings use all 2,334 test images in BF16.",
        7,
        MUTED,
    )
    text(
        28,
        569,
        "Full-resolution PNGs, sample IDs and checkpoint hashes are saved "
        "beside this PDF in examples/.",
        7,
        MUTED,
    )
    text(width - 55, 569, "5 / 5", 7, TEAL, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render clearly labeled partial results while downloads finish",
    )
    args = parser.parse_args()
    data = collect(require_complete=not args.preview)
    index_path = ROOT / "artifacts/models/index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for name, model in data["models"].items():
            stored = index["models"].get(name)
            if (
                stored
                and model["status"] == "completed"
                and stored["best_sha256"] == model["metrics"]["checkpoint_sha256"]
            ):
                model["local_artifacts"] = stored
    if not args.preview:
        for model in data["models"].values():
            if not (ROOT / model["directory"] / "best.pt").is_file():
                raise FileNotFoundError(f"Download best.pt before finalizing: {model['label']}")
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "layout_preview" if args.preview else "aegis_six_model_benchmark"
    (OUT / f"{stem}.json").write_text(json.dumps(data, indent=2) + "\n")
    path = OUT / f"{stem}.pdf"
    render(data, path)
    rows = []
    for model in data["models"].values():
        if model["status"] != "completed":
            continue
        metrics = model["metrics"]
        for weather in ("macro_weather", *WEATHERS):
            scores = metrics["scene_balanced_summary"][weather]
            intervals = (
                metrics["uncertainty"]["macro_weather"]
                if weather == "macro_weather"
                else metrics["uncertainty"]["weather"][weather]["metrics"]
            )
            rows.append(
                {
                    "model": model["label"],
                    "weather": weather,
                    "parameters": model["parameters"],
                    "selected_epoch": metrics["checkpoint_epoch"],
                    **{
                        key: scores[key] for key in ("psnr_db", "ssim", "psnr_gain_db", "ssim_gain")
                    },
                    **{
                        f"{key}_ci95_{end}": intervals[key]["percentile_95"][i]
                        for key in ("psnr_db", "ssim")
                        for i, end in enumerate(("low", "high"))
                    },
                }
            )
    with (OUT / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
