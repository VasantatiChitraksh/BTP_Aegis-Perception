# Dataset collection decision

## Scope

The collection supports the current four-condition pipeline only: **smoke,
rain, fog, and snow**. It is not an indiscriminate adverse-weather dump. Each
condition gets complementary evidence:

| Condition | Paired restoration | Real road or deployment-domain check |
|---|---|---|
| Smoke | ACCV Smoke pairs; SmokeBench | SmokeBench surveillance scenes; no road-object GT |
| Rain | Rain100L; RealRain-1k | DAWN Rain |
| Fog | O-HAZE | RTTS; Foggy Driving; DAWN Fog |
| Snow | CSD; SnowCityscapes | DAWN Snow |

Paired sources support PSNR/SSIM and restoration training. Real unpaired road
sources support the project's actual claim: whether restoration helps or harms
downstream road perception. They must not be reported as paired restoration
benchmarks.

## Kept out of the automatic collection

| Source | Decision | Reason |
|---|---|---|
| SMOKE5K | Optional | Smoke segmentation, not paired desmoking |
| SmokeSeer | Optional | Multi-view RGB/thermal 3D task, unlike the current pipeline |
| SynRain-13k | Excluded | Redundant after RealRain-1k for the initial baseline |
| Rain/Foggy Cityscapes | Gated | Requires separately licensed Cityscapes data |
| CARLA-Haze | Optional | Relevant but about 96 GB |
| RESIDE ITS / Haze4K | Gated | Strong paired fog training sources, but official downloads currently require manual/Baidu access |
| Dense-Haze / NH-HAZE | Optional | Small real paired stress tests; add after the core protocol is fixed |
| Foggy Cityscapes | Gated | 15,000 synthetic fog variants inherit Cityscapes labels; Cityscapes account/license required |
| GoProHazy | Optional | Real paired driving videos, but non-aligned video needs a separate training/evaluation protocol |
| Smoke100K | Optional | 100,000 synthetic pairs plus smoke masks/boxes; no vehicle/person ground truth |
| LaSSoV / PoVSSeg | Gated | Road vehicle-exhaust smoke labels, not paired desmoking or general road-object labels |
| Smoky diesel 6,815 | Unavailable | Has truck/plate/smog labels, but the paper does not publish the image archive |
| Snow100K / RealSnow10K | Optional | Useful later; CSD plus road data cover the initial experiment |
| ACDC / IDD-AW | Gated or optional | Excellent road data, but native task is semantic segmentation |
| Seeing Through Fog | Optional | Large multi-sensor 2D/3D benchmark |
| MUAD | Optional | Synthetic segmentation/depth dataset |

These sources remain in `data/datasets.yaml` so they are discoverable without
silently consuming disk or mixing incompatible tasks.

## Storage and reproducibility rules

1. Archives live in ignored `data/archives/`; extracted originals live in
   ignored `data/raw/<condition>/<dataset_id>/`.
2. Archive names are simple lowercase identifiers. Dataset-native filenames and
   directory structures are preserved during extraction.
3. Source URLs, expected sizes, known hashes, task roles, and rights status live
   in `data/datasets.yaml`; human-readable cautions live in dataset cards.
4. An official download page is preferred. A mirror is explicitly marked as a
   mirror, not silently treated as authoritative.
5. License status `verify` or `unresolved` means do not redistribute the data.
   A repository's code license is not assumed to cover its dataset.
6. Do not merge source-specific official test sets into training. Manifests must
   preserve scene groups and cite their source.

## Recommended first experiments

- Reproduce smoke restoration on the historical pairs, then check generalization
  on SmokeBench.
- Use Rain100L only as a small standard benchmark; use RealRain-1k for broader
  paired evaluation and DAWN Rain for detector impact.
- Use O-HAZE for paired dehazing metrics; use RTTS, Foggy Driving, and DAWN Fog
  for real road-domain checks.
- Use CSD for general desnowing and SnowCityscapes for road scenes; reserve DAWN
  Snow for frozen downstream evaluation.

This separation prevents high synthetic PSNR from being mistaken for evidence
of safer ADAS behavior.

## Smoke and fog annotation clarification

“Annotated smoke” usually means a box or mask around the smoke plume. That is
useful for detecting smoke, but it is not ground truth for cars, people, signs,
and other objects hidden by smoke. For the current pipeline—restore first, then
run an unchanged YOLO model—the latter is needed to report object-detection mAP.
The search found no established, downloadable paired smoke/clear road benchmark
that also supplies those road-object boxes. SmokeBench remains strong for
paired restoration, while a labeled clean road set with reproducible synthetic
smoke is the practical controlled detector test.

Fog is not limited to the 45 O-HAZE pairs. Large synthetic paired training sets
include RESIDE, Haze4K, and Fog/Foggy Cityscapes; real paired stress tests include
O-HAZE, Dense-Haze, and NH-HAZE; GoProHazy supplies paired driving video. RTTS,
Foggy Driving, DAWN Fog, ACDC, and Seeing Through Fog provide real road-domain
labels. The actual constraints are access/license, size, and whether a source is
paired for restoration or annotated for downstream perception—not scarcity.
