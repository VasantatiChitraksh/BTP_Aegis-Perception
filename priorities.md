# provisional tasks:

Priority 1 — Recover data and old artifacts
Before training anything:

Recover the exact smoke archive used in semester 5.
Recover the original Colab notebook and any .pth checkpoint.
Contact the internship contributors for their notebooks/checkpoints.
Record source URL, license, archive SHA-256, image counts, and directory layout.
Confirm GPU and Jetson access with the professor.
Confirm available storage; the v2 plan estimates roughly 200–500 GB for all intended datasets.
Priority 2 — Build the smoke reproduction baseline
Place smoke data under data/raw/smoke/.
Build the paired manifest with scripts/build_pair_manifest.py.
Run scripts/audit_manifest.py.
Inspect the split manually for scene leakage.
Train vanilla Pix2Pix with seeds 42, 43, and 44.
Train Attention-Pix2Pix with the same seeds.
Train the L1-only attention ablation.
Evaluate only after checkpoint/model choices are frozen.
Compare the new held-out result with the historical result while clearly stating that the protocols differ.
This is Gate 1. Do not begin broad architecture exploration until it is complete.

Priority 3 — Establish the detector problem
Before training new weather-restoration models:

Prepare a labelled clear driving test set.
Generate seeded rain, fog, snow, and smoke at multiple severities.
Run COCO-pretrained YOLOv8n on clean and degraded views.
Prepare and audit DAWN.
Run the zero-shot YOLOv8n DAWN baseline per weather and class.
This produces the motivating result: how much adverse weather actually reduces detection performance.

Priority 4 — Extend restoration in a controlled order
Recommended order:

Rain using RainCityscapes or another paired driving dataset.
Fog using Foggy Cityscapes/RESIDE with a driving-scene primary evaluation.
Snow using Snow100K/CSD if data and time remain.
Smoke using the recovered semester-5 pairs plus labelled synthetic driving scenes for task-driven evaluation.
For the v2 minimum, complete smoke, rain, and one of fog/snow. Run both fog and snow only if the data and compute are ready without threatening the submission.

Priority 5 — Run the central detection experiment
For each weather type, evaluate the same detector on:

Clean image.
Degraded image.
Degraded resize-control image.
Vanilla-Pix2Pix restored image.
Attention-Pix2Pix restored image.
On DAWN, omit the unavailable clean view and compare raw, resize control, and restored versions of the same real images.

Report per-weather and per-class mAP@50 and mAP@50:95, not only pooled mAP.

Priority 6 — Compare detector training strategies
Run at minimum:

YOLOv8n trained/fine-tuned on clean data.
YOLOv8n trained on clean plus synthetic adverse weather.
The same detector evaluated with restoration.
This determines whether restoration contributes beyond ordinary data augmentation.

Priority 7 — Run only focused ablations
After the main result table exists:

Attention off versus on.
GAN+L1 versus L1 only.
One loss-weight experiment on one weather type.
Resolution 256 versus 512 on one weather type.
Per-weather versus one conditional model, only if time permits.
Optional TransWeather/PromptIR or lightweight preprocessing comparison.
Avoid a large hyperparameter grid across every dataset.

Priority 8 — Edge deployment
Use the best frozen pipeline and evaluate:

PyTorch FP32.
ONNX Runtime FP32.
TensorRT FP16.
TensorRT INT8 with a separate calibration set.
Measure the restoration model, detector, and end-to-end pipeline separately. Record accuracy after each conversion, batch-one p50/p95 latency, FPS, peak memory, average power, and joules/frame.

Priority 9 — Paper preparation
Begin the manuscript while experiments run:

Finalize the motivation and related-work sections now.
Freeze table/figure templates early.
Add results as machine-generated tables rather than manual transcription.
Include success and failure examples.
State dataset and protocol limitations explicitly.
Keep diffusion restoration, domain adaptation, and adversarial defence in the future-work section unless the core work finishes early.
