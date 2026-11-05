# Completed models and results

## Main comparison

All values below are native-resolution, four-weather balanced test results on
the frozen 2,334-pair protocol. Bold identifies the highest value in that metric
among these six runs. Parameter count alone is not a speed measurement.

| Model | Parameters | Selected epoch / 60 | PSNR (dB) ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|
| Attention Pix2Pix | 29,854,795 | 59 | 24.7281 | 0.820120 |
| **Restormer** | **26,126,644** | **51** | **32.5831** | **0.929690** |
| LiteWeather v2 | 2,189,475 | 59 | 27.2071 | 0.864443 |
| TransWeather | 38,052,078 | 58 | 29.0444 | 0.900040 |
| PromptIR | 35,592,263 | 50 | 31.3899 | 0.918225 |
| RAMiT | 934,651 | 45 | 28.6586 | 0.899470 |

Use [six_models.csv](tables/six_models.csv) for unrounded values, every weather,
raw-input gains and confidence intervals. [six_models.json](evidence/six_models.json)
contains exact run configurations, all epoch histories, regression counts and
metric summaries. [Per-image data](evidence/per_image_restoration.csv.gz) allow
independent reaggregation without downloading images or weights.

Restormer is the current quality-first deployment candidate because it leads
this comparison. It is not the smallest model and has not established an edge
latency advantage. Its rain/fog/snow/smoke scores are respectively
38.7674/31.5053/39.5124/20.5471 dB. Smoke remains the weak condition: 27 of 342
test images lose PSNR and 37 lose SSIM against raw input, with only six test
backgrounds. Aggregate improvement does not imply improvement on every image.

## What was actually trained

- Attention Pix2Pix is the project's attention U-Net generator with a patch
  discriminator, trained from initialization using adversarial loss plus
  100-weighted L1. It is not an exact reproduction of an official attention-GAN paper.
- Restormer uses the official deraining checkpoint topology, base width 48,
  blocks [4,6,6,8], heads [1,2,4,8] and four refinement blocks. The 60-epoch
  run starts from verified official pretrained weights and fine-tunes all layers.
- LiteWeather v2 is a project-specific compact transformer trained from
  initialization. It contains Restormer-style attention at 1/4 and 1/8 scales,
  with input-derived soft prompt routing. The older convolution-only
  `liteweatherformer.py` is preserved; it is not the implementation evaluated here.
- TransWeather, PromptIR and RAMiT use pinned official sources and strict
  checkpoint loading through the project's compatibility adapters. They are
  fine-tuned for 60 epochs on our mixture, not reproductions of the papers'
  original experimental budgets. See the
  [official source/weight lock](../../configs/restoration/official_v1/sources.json).

All six runs use seed 42, global batch four, 4,096 sampled pairs/epoch,
256x256 crops, BF16 execution and validation-selected checkpoints. The five
non-GAN models use AdamW and L1; the GAN uses Adam and adversarial + L1 loss.
Pretraining differs across methods and gives different prior data exposure.
The comparison measures these implemented training recipes on a shared test;
it is not an equal-pretraining causal test of architecture alone or a SOTA claim.

The primary validation selection metric is `macro_scene_psnr_db`. Test results
do not select an epoch. There is only one training seed per model; cluster
bootstrap intervals quantify conditional test uncertainty, not seed variation.

## Which Restormer is used for deployment

The model is the six-model baseline's epoch-51 checkpoint, with compact
`inference.pt` SHA-256:

`5268af32c161413502a8978a839b311152a1dd2bf093ade137e07d58166a8555`

The compact file is 104,701,712 bytes; it contains trained generator weights and
metadata, not the optimizer. This is different from the earlier rain-only model,
the earlier four-weather run, the official initialization and the later joint
research variants. See [model identities](../../artifacts/models/index.json).
Frozen inference uses the trained weights; it does not restore original weights.

## Eight-arm follow-up: report the negative result

The completed joint-v3 suite starts every arm from that trained Restormer and
continues for 10 epochs with two A100s, global batch four, a small learning rate,
and raw/EMA validation. Spatial and gated detail adapters, alone/combined,
and calibrated SSIM/focal-frequency losses are tested against a matched
continuation control. Configurations, calibrated coefficients and histories
are preserved in [joint_experiments.json](evidence/joint_experiments.json).

The control reaches 32.7020 dB; the highest observed test PSNR is 32.7308 dB
for the unchanged architecture with the loss recipe. Architecture-only gains
over control are tiny. None passed the predeclared validation rule requiring
at least +0.10 dB over both initialization and control, nondecreasing macro
SSIM, and bounded per-weather regressions. Selection retained the original
epoch-51 baseline before test. Do not pick a replacement from the test table.

Use the [report](reports/joint_research_report.pdf), [test CSV](tables/joint_test.csv)
and [validation CSV](tables/joint_validation.csv) as an ablation/limitations
section or appendix. These results do not establish architectural innovation.

## Historical rain result

The earlier 20-epoch, seed-42 rain-only run used 784/112/224 pairs and selected
epoch 15. Test scores were 37.2565 dB / 0.972978 SSIM; untouched pretrained
weights scored 23.1265 / 0.794034, and raw inputs 22.2721 / 0.741376.
Those values are supported by [the historical record](evidence/rain_baseline_historical.json).
They do not share the four-weather aggregate or the training protocol above.
Keep the [old report](reports/rain_baseline_historical.pdf) explicitly historical.
