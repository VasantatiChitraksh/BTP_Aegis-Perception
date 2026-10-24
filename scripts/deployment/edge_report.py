#!/usr/bin/env python3
"""Build a concise evidence-first deployment readiness report from measured artifacts."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from reportlab.lib import colors
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

from aegis_perception.edge.common import OUT, write_json

REPORT = OUT / "reports"
REPORT.mkdir(parents=True, exist_ok=True)
restore = json.loads((OUT / "exports/256x448/restormer/validation.json").read_text())
detector = json.loads((OUT / "exports/yolov8n/validation.json").read_text())
audit = json.loads((OUT / "dawn_annotation_audit.json").read_text())
pilot = json.loads((OUT / "detection_pilot72/comparison.json").read_text())
if not pilot["complete"] or len(pilot["evaluations"]) != 12:
    raise ValueError("Report requires all three views and four weather scopes")
rows = []
for item in pilot["evaluations"]:
    metrics = item["metrics"]
    rows.append(
        {
            "view": item["view"],
            "weather": item["weather"],
            "images": item["images"],
            "precision": metrics["metrics/precision(B)"],
            "recall": metrics["metrics/recall(B)"],
            "map50": metrics["metrics/mAP50(B)"],
            "map50_95": metrics["metrics/mAP50-95(B)"],
        }
    )
with (REPORT / "detection_pilot72.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
lookup = {(r["view"], r["weather"]): r for r in rows}
delta = lookup["restormer", "all"]["map50_95"] - lookup["raw", "all"]["map50_95"]
control_delta = lookup["restormer", "all"]["map50_95"] - lookup["resize_control", "all"]["map50_95"]
summary = {
    "status": "Local preparation complete; Jetson execution and acceptance pending",
    "restormer_export": restore,
    "detector_export": detector,
    "detection_pilot": rows,
    "pilot_restored_minus_raw_map50_95": delta,
    "pilot_restored_minus_control_map50_95": control_delta,
    "jetson": {
        "measured": False,
        "fps": None,
        "ram_bytes": None,
        "fp16_quality": None,
        "int8_quality": None,
    },
    "decision": (
        "Keep raw detection as default; restoration remains an explicit "
        "comparison mode until device/accuracy acceptance"
    ),
    "limits": [
        "72-image exploratory pilot, no confidence intervals or superiority claim",
        "No smoke detection labels",
        "Unknown pretrained-detector data overlap",
        (
            "Four restoration and three detection export fixtures are smoke "
            "tests, not full accuracy validation"
        ),
        "256x448 deployment restoration differs from native-resolution training benchmark",
    ],
}
write_json(REPORT / "restormer_edge_readiness.json", summary)
font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
pdfmetrics.registerFont(TTFont("DV", font))
pdfmetrics.registerFont(TTFont("DVB", bold))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB")
navy = colors.HexColor("#10283F")
teal = colors.HexColor("#007F7A")
muted = colors.HexColor("#50657B")
styles = {
    "title": ParagraphStyle(
        "title", fontName="DVB", fontSize=25, leading=30, textColor=navy, spaceAfter=12
    ),
    "sub": ParagraphStyle(
        "sub", fontName="DVB", fontSize=12, leading=17, textColor=teal, spaceAfter=9
    ),
    "body": ParagraphStyle(
        "body", fontName="DV", fontSize=9, leading=14, textColor=navy, spaceAfter=8
    ),
    "small": ParagraphStyle(
        "small", fontName="DV", fontSize=7.8, leading=11, textColor=muted, spaceAfter=7
    ),
}


def P(text, style="body"):
    return Paragraph(text, styles[style])


flow = []


def title(kicker, text):
    flow.extend([P(kicker, "sub"), P(text, "title")])


def table(data, widths=None):
    obj = Table(
        [[P(str(cell), "small") for cell in row] for row in data],
        colWidths=widths,
        hAlign="LEFT",
        repeatRows=1,
    )
    obj.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2EEF1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, 0), (-1, 0), 1, teal),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#DCE3E8")),
            ]
        )
    )
    flow.extend([obj, Spacer(1, 12)])


title("AEGIS / DEPLOYMENT NOTE 01 / 24 OCT 2026", "Restormer, ready for<br/>the device test.")
flow.append(
    P(
        "Frozen weights. Verified exports. Measured local evidence. Jetson results still pending.",
        "sub",
    )
)
table(
    [
        ["DELIVERED", "EVIDENCE"],
        ["Restormer ONNX", "Epoch 51 checkpoint; 26.13M parameters; static 1×3×256×448 input."],
        [
            "Frozen YOLOv8n ONNX",
            "Official COCO weights; static 1×3×640×640 input; no detector finetuning.",
        ],
        [
            "Numerical checks",
            "Restormer: four weather inputs, largest output error "
            f"{max(r['max_abs_error'] for r in restore['validation_panel']):.2g}. "
            "YOLO: three inputs, decoder agrees with Ultralytics.",
        ],
        [
            "Runtime & acceptance",
            (
                "FP16 engine builder, FP32 fallback, optional calibrated INT8, "
                "file/video/camera runner, checksums and timing logs."
            ),
        ],
        [
            "Preserved",
            "Training weights, restoration splits and Chitraksh’s contributed files are unchanged.",
        ],
    ],
    [135, 360],
)
flow.append(P("The deployment geometry is a deliberate tradeoff", "sub"))
flow.append(
    P(
        "Each image is resized with its aspect ratio preserved, padded to "
        "256×448, restored, and returned to its original dimensions "
        "before detection. This reduces compute but may remove "
        "small-object details. The original 32.58 dB / 0.9297 "
        "native-resolution restoration score is not the score of this "
        "resized pipeline."
    )
)
flow.append(P("Local validation passed; edge acceptance has not happened", "sub"))
flow.append(
    P(
        "The laptop is CPU-only. No TensorRT engine, Jetson FPS, thermal "
        "limit, power draw or device memory result has been measured. "
        "Engines must be built and tested on the actual Jetson. The "
        "package does not contain invented hardware results."
    )
)
flow.append(
    P(
        "Local CPU smoke timing: five ONNX model calls after two warmups, median "
        f"{restore['cpu_smoke_timing_ms']['median'] / 1000:.2f} seconds. "
        "This short measurement checks execution only; it is not an edge-performance forecast.",
        "small",
    )
)
flow.append(PageBreak())
title("02 / DOES RESTORATION HELP DETECTION?", "Measure the downstream task.")
flow.append(
    P(
        (
            "72 real DAWN images: 24 fog, 24 rain, 24 snow. Images were "
            "selected by filename hash before scores were computed. All three "
            "views use the same frozen YOLOv8n and the same annotations."
        ),
        "body",
    )
)
headers = ["Weather / view", "mAP50 ↑", "mAP50–95 ↑", "Precision", "Recall"]
data = [headers]
for weather in ("all", "fog", "rain", "snow"):
    selected = [lookup[v, weather] for v in ("raw", "resize_control", "restormer")]
    best = {
        key: max(r[key] for r in selected) for key in ("map50", "map50_95", "precision", "recall")
    }
    for r in selected:
        values = []
        for key in ("map50", "map50_95", "precision", "recall"):
            value = f"{100 * r[key]:.2f}"
            values.append(f"<b>{value}</b>" if r[key] == best[key] else value)
        data.append(
            [
                weather
                + " / "
                + {"raw": "original", "resize_control": "resize only", "restormer": "restored"}[
                    r["view"]
                ],
                *values,
            ]
        )
table(data, [165, 82, 88, 80, 80])
flow.append(
    P(
        "Across the pilot, restored minus original mAP50–95 is "
        f"<b>{100 * delta:+.2f} percentage points</b>; restored minus resize-only is "
        f"<b>{100 * control_delta:+.2f} points</b>. These are observed pilot differences, "
        "not a statistical superiority claim."
    )
)
flow.append(
    P(
        (
            "Bold marks the highest value within each weather group. Values "
            "are percentages. Precision/recall use the maximum-F1 confidence; "
            "mAP uses conf=.001, IoU=.7 NMS and up to 300 detections. Overall "
            "AP is pooled across the 72 images, not the arithmetic mean of "
            "weather AP."
        ),
        "small",
    )
)
flow.append(PageBreak())
title("03 / QUALITY CONTROLS & HANDOFF", "A testable next step.")
flow.append(P("Data checks that make the comparison interpretable", "sub"))
flow.append(
    P(
        f"All {audit['images']} available DAWN annotation pairs were checked: "
        "300 fog, 200 rain and 204 snow images; 5,610 boxes across six road classes. "
        "Source TXT IDs were corrected to COCO IDs only in derived evaluation labels; "
        "every box was checked against VOC XML. One train object is outside the six-class scope. "
        f"{len(audit['exact_duplicate_groups'])} exact duplicate groups were found; "
        "image counts must not be treated as independent scene counts."
    )
)
flow.append(
    P(
        "No DAWN smoke detection labels are available in this project. "
        "COCO pretraining overlap is unverified. Pilot image choices, "
        "source hashes, class-level results and the full annotation audit "
        "are saved in JSON. The full 704-image evaluation is supplied as "
        "a separate resumable command; it was not run for this handoff."
    )
)
flow.append(P("Device acceptance sequence", "sub"))
table(
    [
        ["STEP", "PASS CONDITION"],
        [
            "1 · Verify & preflight",
            (
                "Package checksums match. CUDA/TensorRT/OpenCV imports work. "
                "Actual hardware, free memory and software versions are recorded."
            ),
        ],
        [
            "2 · Compile locally",
            (
                "Build one engine at a time. FP16 uses FP32-sensitive transformer "
                "math. No changes to system power modes or drivers."
            ),
        ],
        [
            "3 · Precision gate",
            (
                "Finite output; restoration numerical agreement plus per-fixture "
                "PSNR drop ≤0.1 dB and SSIM drop ≤0.001. Reject failed candidates "
                "and retain FP32 fallback."
            ),
        ],
        [
            "4 · Sustained run",
            (
                "Measure original, resize-only and restored paths with identical "
                "settings; record median/p95, wall throughput and tegrastats."
            ),
        ],
        [
            "5 · Adoption decision",
            (
                "Check longer videos and untouched labelled data. Keep raw "
                "detection available. Small fixture tests alone do not establish "
                "field reliability."
            ),
        ],
    ],
    [135, 360],
)
flow.append(P("Run from the extracted kit", "sub"))
flow.append(
    P(
        '<font face="DVB">bash scripts/deployment/edge_device.sh '
        "prepare<br/>bash scripts/deployment/edge_device.sh "
        "benchmark</font>"
    )
)
flow.append(
    P(
        (
            "Read START_HERE.txt first for the JetPack system-packages "
            "environment, FP32 fallback, optional INT8 experiment and camera "
            "command. The archive has ONNX graphs, four-weather fixtures, 64 "
            "training-only calibration inputs, three DAWN samples and a "
            "SHA-256 manifest."
        ),
        "small",
    )
)
flow.append(
    P(
        (
            "Sources: NVIDIA TensorRT engine compatibility and accuracy "
            "documentation; Ultralytics v8.3.0 source; DAWN Mendeley v3. Full "
            "links and third-party license notices are included in the "
            "handoff."
        ),
        "small",
    )
)


flow.append(PageBreak())
title("04 / FIXED EXAMPLES", "Inspect the actual outputs.")
flow.append(
    P(
        "One preselected pilot image per weather, shown before and after restoration. "
        "These are real degraded road images; no paired clean ground truth is available. "
        "A visually cleaner image is not proof of better object detection."
    )
)
pilot_manifest = json.loads((OUT / "dawn_pilot72/audit.json").read_text())
for weather in ("fog", "rain", "snow"):
    sample = next(row for row in pilot_manifest["rows"] if row["weather"] == weather)
    flow.append(P(weather.upper() + " / " + sample["id"], "sub"))
    panels = []
    for view in ("raw", "restormer"):
        path = OUT / "dawn_pilot72" / view / "images" / (sample["id"] + ".png")
        item = Image(str(path))
        item._restrictSize(238, 135)
        panels.append(item)
    gallery = Table(
        [[P("ORIGINAL", "small"), P("RESTORED", "small")], panels], colWidths=[248, 248]
    )
    gallery.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    flow.extend([gallery, Spacer(1, 10)])


def footer(canvas, doc):
    canvas.setFillColor(muted)
    canvas.setFont("DV", 8)
    canvas.drawString(43, 25, "AEGIS  /  FROZEN RESTORMER + YOLOv8n  /  RESEARCH HANDOFF")
    canvas.drawRightString(552, 25, str(doc.page))


SimpleDocTemplate(
    str(REPORT / "restormer_edge_readiness.pdf"),
    title="AEGIS Restormer Edge Readiness",
    author="Aegis Perception",
    pagesize=(595, 842),
    leftMargin=43,
    rightMargin=43,
    topMargin=38,
    bottomMargin=43,
).build(flow, onFirstPage=footer, onLaterPages=footer)
print(REPORT / "restormer_edge_readiness.pdf")
