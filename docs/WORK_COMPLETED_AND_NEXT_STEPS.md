# Work Completed, Technical Assessment, and Next Steps

Date: 26 August 2026  
Project: Aegis Perception — Weather-Robust ADAS

## 1. Executive summary

The repository preparation and experiment-design phase is complete. The project
now has organized source documents, an audit of the previous teams' work, a
literature-backed research direction, a reproducible folder structure, and
boilerplate code for restoration, dataset handling, object detection,
evaluation, ONNX export, and edge benchmarking.

The actual research experiments are **not yet complete**. No new restoration or
detection model has been trained because the datasets, historical checkpoints,
PyTorch environment, GPU allocation, and Jetson hardware are not available in
this checkout.

The immediate next objective is to reproduce the semester-5 smoke result on a
proper held-out split. Only after that result is verified should the project
extend to rain, fog, and snow.

---

## 2. What I completed from the requested tasks

### Task 1 — Organize documentation, proposals, and repository structure

Status: **Completed**

I preserved the supplied project material and organized the previously untracked
documents into the repository:

- Original TiHAN proposal.
- Revised Weather-Robust ADAS Execution Plan v2.
- Semester-5 research and code reports.
- Semester-5 proposed workflow image.
- Internship reports and monthly progress report.
- Historical `baseline.ipynb` notebook.

I added the following project documentation:

- `docs/PROJECT_AUDIT.md` — evidence-based audit of the proposal, reports,
  notebook, and missing artifacts.
- `docs/PROPOSALS_AND_REPORTS.md` — index of all supplied source documents and
  their authority/status.
- `docs/LITERATURE_REVIEW.md` — focused paper review for multi-weather
  restoration, task-driven restoration, adverse-weather detection, diffusion,
  and efficient inference.
- `docs/EXPERIMENT_PLAN.md` — prioritized and gated list of experiments.
- `docs/REPRODUCIBILITY.md` — rules for splits, seeds, metrics, checkpoints, and
  hardware benchmarking.
- `data/README.md` — dataset structure, intended use, and evaluation limitations.
- A rewritten root `README.md` with installation, structure, and command examples.

I also created the main repository structure:

```text
configs/                 Version-controlled experiment configurations
data/                    Dataset contracts, cards, and split manifests
docs/                    Audit, research review, protocols, and plans
scripts/                 Runnable experiment entry points
src/aegis_perception/    Reusable Python package
tests/                   Lightweight automated tests
```

Large datasets, checkpoints, ONNX models, TensorRT engines, runs, and generated
images are excluded from Git through `.gitignore`.

### Task 2 — Prepare boilerplate for experiments

Status: **Boilerplate completed; model training pending**

#### Restoration implementation

I converted the useful part of the historical notebook into reusable code:

- Six-level Pix2Pix U-Net generator.
- Switchable attention gates on the skip connections.
- PatchGAN discriminator.
- GAN + weighted L1 training objective.
- L1-only ablation support.
- Pix2Pix weight initialization.
- Validation-based checkpoint selection.
- Per-epoch JSON training history.
- Per-sample held-out PSNR and SSIM evaluation.
- Run metadata containing the configuration, Git commit, software environment,
  CUDA status, and device information.

Available restoration configurations:

- Smoke Attention-Pix2Pix.
- Smoke vanilla Pix2Pix.
- Smoke Attention U-Net with L1 only.
- Rain Attention-Pix2Pix.
- Fog Attention-Pix2Pix.
- Snow Attention-Pix2Pix.

#### Dataset and split tooling

I added tools to:

- Match degraded and clean image pairs.
- Generate deterministic train/validation/test splits.
- Keep related versions of the same scene in one split.
- Detect missing or corrupt images.
- Detect exact clean-target duplicates across different splits.
- Record weather type and dataset source in a CSV manifest.

This directly fixes the main experimental weakness in `baseline.ipynb`, where
the training loader was also used for evaluation.

#### Dataset-first scope decision

The project now uses downloaded paired and real adverse-weather datasets rather
than a local weather-generation layer. Generic procedural/diffusion generation
was removed because it was not necessary for the v2 dataset-based experiments.
Diffusion-based restoration remains future work, as specified by v2.

#### Object-detection boilerplate

I added scripts for:

- YOLOv8n/s fine-tuning.
- Evaluating a single dataset condition.
- Evaluating a clean/degraded/restored condition matrix.
- Saving detector metrics as machine-readable JSON.
- Retaining per-condition Ultralytics plots and outputs.

A provisional DAWN dataset YAML is included, but its class order must be checked
against the exact downloaded annotation release before training.

#### Restoration-to-detection integration

I added a dataset restoration tool that:

- Runs a trained restoration checkpoint on a complete image tree.
- Preserves relative paths and original image dimensions.
- Optionally produces a bicubic downsample/upsample control dataset.

The resize control is important because the historical restoration model accepts
256×256 images. Without this control, a detector comparison could mistakenly
attribute resolution loss to the restoration network.

#### Export and edge preparation

I added tools for:

- Exporting the restoration generator to ONNX.
- Comparing ONNX output against the PyTorch checkpoint.
- Measuring ONNX batch-one p50/p95 latency and FPS.

TensorRT engine creation and `tegrastats` power measurements must be performed on
the actual NVIDIA/Jetson environment.

### Task 3 — Review research and define experiments before the main project

Status: **Completed**

The literature review covers:

- Pix2Pix and Attention U-Net foundations.
- All-in-One bad-weather restoration.
- TransWeather.
- PromptIR.
- Weather-general and weather-specific restoration.
- Adverse Weather Removal with Codebook Priors.
- RestoreX-AI and task-driven restoration.
- ERUP-YOLO and detector-oriented preprocessing.
- WeatherDiffusion, DiffIR, and recent task-oriented diffusion restoration.
- DAWN, BDD100K, RainCityscapes, ACDC, and SynFog datasets.

The experiment plan now defines:

1. Dataset/provenance audit.
2. Smoke reproduction.
3. Attention versus vanilla ablation.
4. GAN+L1 versus L1-only ablation.
5. Zero-shot detector degradation baseline.
6. Rain and fog/snow extension.
7. Same-scene clean/degraded/resize-control/restored detection comparison.
8. Real-weather DAWN raw/resize-control/restored comparison.
9. Detector fine-tuning strategy comparison.
10. Focused loss, resolution, and conditional-model ablations.
11. ONNX, TensorRT FP16/INT8, latency, memory, power, and accuracy evaluation.

Every phase has a pass/fail gate and a defined fallback so that the team does not
spend the submission window exploring architectures before establishing a valid
baseline.

### Version-control work

The completed work was separated into six reviewable commits:

```text
26ca2a8 docs: archive project proposals and audit prior work
9fcc60b docs: define literature-backed experiment roadmap
7b182e1 feat: add reproducible restoration experiment scaffold
1e74eff feat: add weather detection and edge export tooling
cbfd8d7 test: cover configs manifests weather and model shapes
551fedc docs: add project setup and experiment quickstart
```

These six commits are now present on `origin/main`.

---

## 3. Important findings from the previous work

### 3.1 The notebook is the attention model, not a separate baseline

`baseline.ipynb` implements the semester-5 Attention-Pix2Pix smoke model. It does
not implement the reported vanilla Pix2Pix comparison.

Its recorded result is approximately:

- PSNR: 25.30 dB.
- SSIM: 0.7874.

However, the same dataset loader is used for training and evaluation. There is
no held-out split, fixed split manifest, saved checkpoint, dataset checksum, or
per-image metric output. Therefore, these numbers should be described as a
historical prototype result, not a verified test-set baseline.

### 3.2 The internship reports contain useful leads, not verified results

The reports claim successful rain, fog, snow, and smoke removal, YOLOv5/DETR
integration, Raspberry Pi deployment, and 10–15 FPS performance. Their code,
checkpoints, label mappings, raw logs, and split lists are absent.

One report describes DETR but its appendix begins with OpenCV YOLO code. This
does not prove the work is invalid, but it means the claims cannot currently be
used as paper evidence.

The correct action is to recover those artifacts if possible and classify every
claim as reproduced, not reproduced, or not testable.

### 3.3 DAWN cannot provide the proposed three-way clean comparison

DAWN contains real adverse-weather road scenes, but no matched clean image for
each adverse image. Therefore:

- On paired/synthetic driving scenes: use clean, degraded, resize-control, and
  restored views.
- On DAWN: use raw adverse, resize-control, and restored views.

Using an unrelated clear-weather dataset as “clean DAWN” would produce an
invalid comparison because scene content and object distributions would differ.

### 3.4 PSNR/SSIM alone cannot establish ADAS benefit

A restoration model can improve visual appearance while erasing a small person,
altering a traffic sign, smoothing a motorcycle, or hallucinating vehicle edges.

The primary scientific metric must therefore be downstream detection mAP. PSNR
and SSIM remain supporting metrics for paired data.

### 3.5 The 256×256 bottleneck is a serious confounder

The historical model resizes every image to 256×256. This can remove small-object
information before YOLO receives the image. A restored image must be compared to
a degraded image passed through the same downsample/upsample operation.

A 256-versus-512 restoration-resolution ablation should be performed on one
representative weather condition after the core result exists.

---

## 4. My technical assessment

### 4.1 The revised v2 scope is the right direction

The original proposal combines restoration, transformers, foundation models,
domain adaptation, adversarial attacks, defensive distillation, diffusion, and
edge deployment. Completing all of that rigorously in one short project would be
unrealistic.

The v2 plan is stronger because it asks one coherent question:

> Does an efficient restoration front-end improve adverse-weather road-object
> detection, and is the improvement still useful after edge optimization?

That can become a defensible paper if the evaluation is rigorous.

### 4.2 Attention gates alone are unlikely to be the final novelty

Attention-gated U-Nets are established technology. Merely adding them to Pix2Pix
and reporting higher PSNR is probably not a sufficiently strong 2027 research
contribution.

The stronger contribution is the combination of:

- Reproducible multi-weather analysis.
- Task-driven detection evaluation.
- Real-versus-synthetic generalization.
- Negative and failure-case analysis.
- FP16/INT8 accuracy-latency-power measurements on real edge hardware.

If time permits, a small task-aware loss or unified conditional model could add
model novelty, but only after the core evidence is complete.

### 4.3 A detector trained on weather may beat restoration

The project must include a detector trained with adverse-weather augmentation
and no restoration front-end. Otherwise, reviewers may reasonably ask why an
extra GAN is needed at all.

The essential comparison is:

1. Clean-trained detector.
2. Weather-augmented detector.
3. Weather-augmented detector receiving restored images.

If ordinary weather augmentation matches or beats restoration, the team should
report it honestly and reconsider the contribution. A learned lightweight
detector-oriented preprocessing module may then be more valuable than a
human-perception-oriented GAN.

### 4.4 Diffusion is valuable future work but risky for the current deadline

Diffusion models are scientifically relevant and often produce strong perceptual
results, but iterative inference, model size, and semantic hallucination conflict
with the current edge goal.

For this cycle, diffusion should be limited to an optional data-augmentation
study. Efficient diffusion restoration such as compact-prior or one-step methods
can be explored after the core submission.

### 4.5 The likely strongest paper outcome

The most defensible paper is not necessarily one claiming a new state-of-the-art
restoration network. A strong outcome would be:

> A controlled study showing when attention-based restoration improves or harms
> adverse-weather detection, how the result transfers to real weather, and what
> accuracy/latency/power trade-off remains after edge quantization.

Even a negative result for one weather type is useful if it is analyzed honestly.

---

## 5. What remains incomplete

The following work cannot be claimed as completed yet:

- Recovering or downloading the datasets.
- Verifying licenses and archive checksums.
- Recovering historical Colab notebooks and checkpoints.
- Building final smoke/rain/fog/snow manifests.
- Verifying the DAWN label mapping and frozen split.
- Installing the GPU training environment.
- Reproducing the smoke result.
- Training vanilla and attention models across three seeds.
- Training rain, fog, or snow models.
- Running YOLOv8 baselines or fine-tuning.
- Calculating new mAP, PSNR, SSIM, confidence intervals, or failure tables.
- Exporting a real trained checkpoint to ONNX/TensorRT.
- Running FP16 or INT8 quantization.
- Measuring Jetson FPS, latency, memory, power, or joules/frame.
- Writing the final IEEE paper.

After removing the unnecessary weather-generation layer, the current code has
passed four lightweight tests. One model-shape test was
skipped because PyTorch is not installed on the current machine. Python syntax
compilation passed for all source files and scripts.

---

## 6. Exact next steps

### Priority 0 — Preserve the current foundation

1. Review the six foundation commits.
2. Protect or tag this point as the pre-experiment foundation.

Suggested tag after review:

```bash
git tag -a v0.1.0-experiment-scaffold -m "Reproducible experiment scaffold"
git push origin --follow-tags
```

### Priority 1 — Recover data and old artifacts

Before training anything:

1. Recover the exact smoke archive used in semester 5.
2. Recover the original Colab notebook and any `.pth` checkpoint.
3. Contact the internship contributors for their notebooks/checkpoints.
4. Record source URL, license, archive SHA-256, image counts, and directory layout.
5. Confirm GPU and Jetson access with the professor.
6. Confirm available storage; the v2 plan estimates roughly 200–500 GB for all
   intended datasets.

### Priority 2 — Build the smoke reproduction baseline

1. Place smoke data under `data/raw/smoke/`.
2. Build the paired manifest with `scripts/build_pair_manifest.py`.
3. Run `scripts/audit_manifest.py`.
4. Inspect the split manually for scene leakage.
5. Train vanilla Pix2Pix with seeds 42, 43, and 44.
6. Train Attention-Pix2Pix with the same seeds.
7. Train the L1-only attention ablation.
8. Evaluate only after checkpoint/model choices are frozen.
9. Compare the new held-out result with the historical result while clearly
   stating that the protocols differ.

This is Gate 1. Do not begin broad architecture exploration until it is complete.

### Priority 3 — Establish the detector problem

Before training new weather-restoration models:

1. Select labelled paired driving datasets with clean and adverse views.
2. Run COCO-pretrained YOLOv8n on their clean and degraded views.
3. Prepare and audit DAWN.
4. Run the zero-shot YOLOv8n DAWN baseline per weather and class.

This produces the motivating result: how much adverse weather actually reduces
detection performance.

### Priority 4 — Extend restoration in a controlled order

Recommended order:

1. Rain using RainCityscapes or another paired driving dataset.
2. Fog using Foggy Cityscapes/RESIDE with a driving-scene primary evaluation.
3. Snow using Snow100K/CSD if data and time remain.
4. Smoke using the recovered semester-5 pairs plus labelled synthetic driving
   scenes for task-driven evaluation.

For the v2 minimum, complete smoke, rain, and one of fog/snow. Run both fog and
snow only if the data and compute are ready without threatening the submission.

### Priority 5 — Run the central detection experiment

For each weather type, evaluate the same detector on:

1. Clean image.
2. Degraded image.
3. Degraded resize-control image.
4. Vanilla-Pix2Pix restored image.
5. Attention-Pix2Pix restored image.

On DAWN, omit the unavailable clean view and compare raw, resize control, and
restored versions of the same real images.

Report per-weather and per-class mAP@50 and mAP@50:95, not only pooled mAP.

### Priority 6 — Compare detector training strategies

Run at minimum:

1. YOLOv8n trained/fine-tuned on clean data.
2. YOLOv8n trained on clean plus available adverse-weather dataset images.
3. The same detector evaluated with restoration.

This determines whether restoration contributes beyond ordinary data
augmentation.

### Priority 7 — Run only focused ablations

After the main result table exists:

1. Attention off versus on.
2. GAN+L1 versus L1 only.
3. One loss-weight experiment on one weather type.
4. Resolution 256 versus 512 on one weather type.
5. Per-weather versus one conditional model, only if time permits.
6. Optional TransWeather/PromptIR or lightweight preprocessing comparison.

Avoid a large hyperparameter grid across every dataset.

### Priority 8 — Edge deployment

Use the best frozen pipeline and evaluate:

1. PyTorch FP32.
2. ONNX Runtime FP32.
3. TensorRT FP16.
4. TensorRT INT8 with a separate calibration set.

Measure the restoration model, detector, and end-to-end pipeline separately.
Record accuracy after each conversion, batch-one p50/p95 latency, FPS, peak
memory, average power, and joules/frame.

### Priority 9 — Paper preparation

Begin the manuscript while experiments run:

1. Finalize the motivation and related-work sections now.
2. Freeze table/figure templates early.
3. Add results as machine-generated tables rather than manual transcription.
4. Include success and failure examples.
5. State dataset and protocol limitations explicitly.
6. Keep diffusion restoration, domain adaptation, and adversarial defence in the
   future-work section unless the core work finishes early.

---

## 7. Recommended immediate team allocation

For three team members:

- Member 1 — recover datasets/artifacts, create dataset cards and manifests, and
  reproduce smoke restoration.
- Member 2 — prepare DAWN and labelled clean/synthetic detection datasets, then
  run YOLOv8n baselines.
- Member 3 — establish the GPU environment, experiment tracking, ONNX export
  checks, and Jetson access; maintain the result tables.

All members should review split manifests and contribute to the paper weekly.

## 8. Definition of the next completed milestone

The next milestone should be considered complete only when the repository has:

- A committed smoke dataset card and frozen split manifest.
- Passing dataset audit output.
- Vanilla, attention, and L1-only runs for seeds 42/43/44.
- Saved best checkpoints and complete run metadata.
- Held-out PSNR/SSIM with per-image results and confidence intervals.
- At least one zero-shot YOLOv8n clean-versus-weather degradation table.
- A short reproduction report stating whether the historical smoke improvement
  was reproduced.

That milestone converts the current scaffold into verified research evidence and
provides a safe foundation for the rain/fog/snow extension.
