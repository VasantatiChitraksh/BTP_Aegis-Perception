# Dataset and evaluation protocol

## Actual active split

Counts are paired observations. Parentheses give distinct background groups,
not guaranteed independent capture sequences.

| Weather | Source used | Train pairs (groups) | Validation pairs (groups) | Test pairs (groups) |
|---|---|---:|---:|---:|
| Rain | RealRain-1k-H | 773 (755) | 112 (112) | 224 (224) |
| Fog proxy | RESIDE-6K | 5,102 (5,061) | 897 (893) | 1,000 (542) |
| Snow | SnowCityscapes | 5,382 (1,794) | 618 (206) | 768 (256) |
| Smoke | SmokeBench, repaired project split | 7,353 (31) | 384 (6) | 342 (6) |
| **Total pairs** | | **18,610** | **2,011** | **2,334** |

Sources: [protocol](evidence/data_protocol.json), [latest audit](evidence/data_audit.json),
[CSV table](tables/dataset_split.csv). The exact membership is
[active.csv](../../data/manifests/weather_v3/active.csv), SHA-256
`9e7c661f66626761a1641f887f8c8720fc69fc9dffcad5602ef649a944c904ab`.
Do not regenerate the split to prepare a manuscript.

The reserve contains 5,232 snow test pairs, 760 smoke test pairs and 1,123 smoke
validation pairs. These remain held out; they are not extra training examples.
The earlier 6,000-snow test and the historical rain 784/112/224 split are
different protocols, not the current benchmark.

## Where the images came from

| Source | Provenance and meaning of synthetic | Use in this project |
|---|---|---|
| [RealRain-1k](https://github.com/hiker-lw/RealRain-1k) | Authors derive paired images from real rainy videos using rain-density-controlled filtering. These are processed real-video-derived pairs, not independently photographed clean/rainy exposures. | Downloaded high-density H release; SynRain-13k excluded. |
| [RESIDE-6K release](https://github.com/IDKiro/DehazeFormer) | Source-provided synthetic haze over clean imagery; indoor/outdoor haze is a fog proxy. | Downloaded training/test archives from the DehazeFormer release. No claim of real-driving fog equivalence. |
| [SnowCityscapes source recorded in catalog](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet) | Source-provided synthetic snow on Cityscapes backgrounds. | Downloaded pairs; variants grouped by clean background and known sequence cluster. The exact source-generation parameters are not established by our audit. |
| [SmokeBench](https://github.com/QianfengY/SmokeBench) | The authors describe a real-world surveillance desmoking dataset. See the [acquisition paper](https://arxiv.org/html/2509.12701v1). | Downloaded paired images; custom background-disjoint split replaces the leaking historical project manifest. Surveillance smoke is not a road-object detection benchmark. |
| [DAWN](https://data.mendeley.com/datasets/766ygrbt8y/3) | Real adverse-weather images with object annotations; no paired clean restoration targets. | Detection only: 72 pilot images, 24 each fog/rain/snow, drawn from 704 audited image/annotation pairs. |

Primary source links checked on 5 November 2026. The
[source catalog extract](evidence/dataset_sources.json) preserves archive
hashes and provenance from [data/datasets.yaml](../../data/datasets.yaml).
Downloaded synthetic data remain synthetic. We did not generate new training
images with diffusion, GANs or another generative model for these six-model or
joint-v3 runs. Crop/flip augmentation is separate from source weather synthesis.
Optional weather-simulation scripts elsewhere in the repository were not used
to create this frozen training mixture.

O-HAZE, CSD, historical smoke, RTTS, Foggy Driving and other catalog entries are
historical, excluded or alternative sources; their presence in the catalog does
not mean they contributed to the six-model benchmark. Preserve source-specific
data terms when redistributing images; this evidence pack adds no raw datasets.

## Leakage checks and limits

The latest audit decoded 52,140 unique files covering 22,955 active pairs and
7,115 reserve pairs. It found no duplicate decoded pairs, no exact cross-split
overlaps and no near-target cross-split candidates under the specified rule.
Near-copy candidates within a partition already shared a background group.
The rule uses target pHash distance <=2, aspect difference <=0.01 and 16x16 RGB
thumbnail MAE <=2/255.

This fixes the known clean-reference overlap in the rejected smoke manifests.
It does not certify all unknown sequences, semantic scene overlap or pixel-level
alignment. Smoke has only six held-out backgrounds per partition. Scores are
exploratory for smoke, and additional frames do not create new independent scenes.
The test set has been inspected in previous experiments; it is not a new blind
external benchmark.

## Sampling actually used in the reported runs

Use the saved run configurations in [six_models.json](evidence/six_models.json)
and [joint_experiments.json](evidence/joint_experiments.json). The completed runs
use `weighted_seeded` sampling with replacement, weather probabilities
rain/fog/snow = 0.30 each and smoke = 0.10, with background-aware sampling.
These are draw probabilities, not percentages of data assigned to training.
An epoch is 4,096 draws, with global batch four and 1,024 optimizer steps.
Training uses paired random 256x256 crops/flips; evaluation uses native images.

The older [v3 planning document](../weather_protocol_v3.md) also describes a
different balanced-cycle sampler for its earlier follow-up runs. Do not copy
that sampler description into the completed six-model or joint-v3 methods.
Saved run configurations take precedence over generic planning documents.

## How the reported metrics are aggregated

PSNR is computed per image in RGB [0,1], then averaged within a degradation
stratum, across strata within each background, across backgrounds within weather,
and finally equally across the four weathers. SSIM follows the same aggregation,
with uniform 7x7 windows and sample covariance. Mean image PSNR and PSNR from
pooled MSE are not interchangeable. The primary score is not snow-weighted.

The six-model CSV includes 95% cluster-bootstrap intervals (2,000 resamples),
conditional on the fixed checkpoint/test set. They are not confidence intervals
over training seeds. All six runs use seed 42. The original method split rule
and scoring code are linked from [AI_GUIDE.md](AI_GUIDE.md).
