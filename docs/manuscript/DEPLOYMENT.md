# Detection, compression and hardware evidence

## Current model and pipeline

The deployment candidate is the trained epoch-51 Restormer followed by frozen
COCO-pretrained YOLOv8n. No detector fine-tuning or detector-aware restoration
training is established by these experiments. The local comparison includes
raw input, a resize-only control and restored input with the same detector.

The 256x448 restoration path resizes/pads the full image, restores it and
returns it to the original geometry before detector preprocessing. It is not
native-resolution evaluation or tiled inference. YOLOv8n uses 640x640 inputs.
BF16 was the training execution precision; the exported ONNX graphs are FP32.

## Completed 72-image DAWN pilot

Values are mAP50–95 in percent, not PSNR or inference FPS. There are 24 images
per weather, selected before scoring. The overall AP is evaluated across the
combined images, not obtained by averaging weather AP values.

| Weather | Raw | Resize-only | Restormer | Restormer minus raw (percentage points) |
|---|---:|---:|---:|---:|
| All 72 images | 29.97 | 29.25 | 32.86 | +2.89 |
| Fog | 26.99 | 29.40 | 37.11 | +10.12 |
| Rain | 38.87 | 37.07 | 37.21 | -1.66 |
| Snow | 36.27 | 33.47 | 35.62 | -0.65 |

Source: [full-precision CSV](tables/detection_pilot72.csv) and
[export/pilot JSON](evidence/edge_readiness.json). This is a small exploratory
pilot without confidence intervals. There are no smoke object labels in this
pilot, no six-model detection comparison, and no completed full-704 evaluation
in this evidence pack. Precision/recall use each evaluation's maximum-F1 point;
do not call them common-fixed-threshold measurements. Raw detection remains the
default pending acceptance of restoration on the intended workload.

## Hardware identity: distinguish the two paths

The earlier plan targeted **Jetson Orin Nano (4 or 8 GB)**. Later checked-in
[operator notes](../nano_yolo_demo.txt) explicitly record a user-reported
**original Jetson Nano 4 GB**, L4T 32.7.3, Python 3.6.9 and TensorRT 8.2.1.8.
The original Nano and Orin Nano are different hardware/software targets.

For writing, use [hardware_status.json](hardware_status.json). Its reported
device fields are documentation, not live detection or runtime configuration.
When new device evidence arrives, record the board/RAM/L4T/TensorRT/power mode,
engine hashes, resolution and measured results there with their source. Do not
select a board by editing prose to make an existing performance claim fit.

| Path | Instructions | Evidence level |
|---|---|---|
| Original Nano / reported JetAutoPro setup | [Nano adapter guide](../nano_yolo_demo.txt) | Limited user-reported detector/build observations; full restored pipeline unvalidated |
| Orin deployment package | [Integrated handoff](../edge_integrated_handoff.txt), [SETUP](../../SETUP.md) | Prepared package and local checks; no verified Orin device benchmark in this pack |

A later Nano note records a YOLOv8n FP16-permitted engine build with 18.1549
queries/s over about three seconds of random input. Its fog fixture subsequently
failed a numerical gate. This is **not accepted camera FPS, detection accuracy
or Restormer throughput**. The note refers to a working FP32 detector, but this
pack does not include an independent full device acceptance record for it.

Restormer build attempts at 256x448 and 128x224 encountered tactic memory
pressure; a 128x128 ONNX trial was then prepared. Lower-resolution exports use
the same trained weights and do not reduce parameter count. No validated
Restormer engine or sustained combined Nano performance is established here.
This later history qualifies the older October 24 readiness report's statement
that device execution had not yet been performed.

## Export and compression status

- FP32 ONNX export agreement passed on a small reference panel for Restormer
  and YOLOv8n. This is a numerical smoke test, not full accuracy acceptance.
- FP16-permitted TensorRT builders and numerical checks exist. Permitting FP16
  does not mean every engine layer runs FP16. Sensitive operations can use FP32.
- Calibrated INT8 remains an experiment; no accepted complete INT8 Restormer
  device benchmark is included. The original-Nano path must not inherit Orin
  INT8/Tensor Core assumptions.
- The [pruning screen](tables/pruning_screen.csv) found substantial losses on
  four images for the tested zero-shot masks. This rejects those candidates;
  it does not establish that all pruning approaches fail.
- The [custom quantization screen](evidence/custom_quantization_screen.json)
  ran discretized weights in FP32 ONNX Runtime, without activation quantization.
  It failed fog/smoke numerical agreement and did not reduce ONNX file size.
  It is not proof of native INT8 acceleration or a validated Aegis-HAMQ result.

Keep the trained unpruned baseline as the starting point. Device speed, sustained
memory/power/thermal behavior and quantized detection utility need actual
acceptance measurements. Earlier throughput/retention projections in experimental
documents must not be transcribed into a paper's measured-results table.

Model binaries are distributed through [GitHub releases](https://github.com/VasantatiChitraksh/BTP_Aegis-Perception/releases),
not this evidence folder. Start with the [integrated release](https://github.com/VasantatiChitraksh/BTP_Aegis-Perception/releases/tag/edge-integrated-20261030)
and follow the guide for the actual device. No DGX training is needed to read
these reports or run the prepared inference workflow.
