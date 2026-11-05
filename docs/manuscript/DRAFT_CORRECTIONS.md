# Corrections for the earlier rain-only working draft

These refer to the eight-page draft titled *Aegis Perception: A Rain Restoration
Baseline and Evaluation Protocol for Weather-Robust Edge Perception*. Preserve
it as a dated draft; use the corrections below when preparing the next version.

| Location | Correction |
|---|---|
| Title, abstract, introduction and conclusion | Broaden to the completed multi-weather benchmark and exploratory downstream evaluation. Keep the rain baseline as historical evidence. |
| Section 3.2 | Distinguish the earlier 20-epoch rain-only run from the 60-epoch four-weather Restormer and its epoch-51 selected checkpoint. |
| Sections 3.3–3.4 and Figure 2 | The old CNN prototype remains in the repo, but trained LiteWeather v2 has low-resolution transformer attention and input-driven prompt routing. Describe that actual implementation and measured results. |
| Section 3.3 named baselines | Official TransWeather, PromptIR and RAMiT runs use pinned source/weight adapters. Do not infer their implementation from preserved legacy stub filenames. |
| Section 3.5 | Detector-feature/task losses are proposed work, not the loss used in the completed six-model comparison. |
| Section 4.1 and Table 1 | Replace the active-data table with the current 18,610/2,011/2,334 protocol. Describe historical leaking smoke splits separately from repaired ones. RESIDE-6K fog-proxy training is complete. |
| Section 4.2 | Supply separate methods for the historical rain experiment, six-model comparison and ten-epoch joint ablations. Use run JSON, not an older planning document, for sampling and loss details. |
| Section 4.3 | Define equal weather/background/stratum aggregation. Preserve per-weather values and uncertainty limits. |
| Sections 5.1–5.2 and Table 2 | The historical rain values are supported; a current paper should lead with the six-model table. Full-precision JSON/CSV evidence now accompanies the reports. |
| Section 5.3 and Table 3 | Populate the available raw/resize-only/Restormer detection rows with the 72-image pilot. Other restoration/detector controls remain unmeasured. |
| Section 5.3 | Include the eight completed Restormer arms as a negative/inconclusive improvement result, with a matched continuation control. |
| Section 6 hardware | Earlier Orin plans and later original-Nano attempts must be versioned separately. Latest operator notes identify original Nano 4GB; do not silently equate it with Orin Nano. |
| Section 6 deployment | Local ONNX parity checks are complete. Limited later Nano build/detector observations exist; sustained end-to-end deployment is still unvalidated. |

## Claims to retain

Reconstruction quality, downstream utility and hardware feasibility are distinct.
Use identical scenes/detector settings for input-view comparisons, keep test out
of checkpoint selection, and distinguish independent backgrounds from frames.
Single-seed evidence and limited smoke diversity remain limitations. Avoid SOTA,
operational safety and algorithmic novelty claims unsupported by matched tests.

## Claims not supported by this evidence

- Universal improvement across weather, objects or images.
- A breakthrough from the spatial/detail Restormer modifications.
- Successful detector-aware training or adverse-weather detector fine-tuning.
- Measured 38–52 FPS Restormer on Orin, 4.3 FPS combined Nano inference,
  or an accepted 4x compressed/accelerated custom quantizer.
- Native-image PSNR/SSIM applying unchanged to resized 128x128/128x224/256x448
  deployment, or random-input detector query rate equalling camera pipeline FPS.
- Bootstrap intervals over six smoke backgrounds demonstrating broad smoke
  generalization, or test resampling accounting for training-seed uncertainty.

The reports are evidence for a controlled empirical study. Whether the final
paper's contribution meets a venue's standards depends on its claims, additional
controls and completed device evaluation; formatting cannot establish novelty.
