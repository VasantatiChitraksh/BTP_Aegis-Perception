# Frozen four-weather protocol v3

This protocol supersedes the proposed v2 follow-up for new runs. It fixes the
sampling and reporting design before further training. Restormer stays unchanged:
we have evidence of imbalanced coverage and weak smoke generalization, not evidence
that changing the backbone will solve either. No new images were synthesized or
new external datasets added. The existing sources are RealRain1k-H, RESIDE6K,
SnowCityscapes, and SmokeBench v2; source-provided synthetic degradations remain
synthetic even though we downloaded them. See the dataset guide for provenance.

## Active data

Counts are paired observations, not independent scenes. Membership and checksums
are frozen in `data/manifests/weather_v3/{active.csv,reserve.csv,protocol.json}`.

| Weather / source | Train pairs (backgrounds) | Validation pairs (backgrounds) | Test pairs (backgrounds / known clusters) |
|---|---:|---:|---:|
| Rain / RealRain1k-H | 773 (755) | 112 (112) | 224 (224 / 224) |
| Fog / RESIDE6K | 5,102 (5,061) | 897 (893) | 1,000 (542 / 542) |
| Snow / SnowCityscapes | 5,382 (1,794) | 618 (206) | 768 (256 / 256) |
| Smoke / SmokeBench v2 | 7,353 (31) | 384 (6) | 342 (6 / 6) |
| Total | **18,610** | **2,011** | **2,334** |

All original cleaned training groups stay in training. Expanded validation uses
previously held-out groups from the full source manifests; none were used to train
the initialization checkpoint. Fifteen exact decoded input/target duplicate pairs
are removed, including eight training pairs. Active and reserve manifests retain
the original cleaned train/val/test assignments. We do not recycle inspected test
images into training. Reserve contains 5,232 snow test, 760 smoke test, and 1,123
smoke validation pairs; it is excluded from training and model selection.

The sample sizes are a practical coverage/computation choice, not a statistical
power guarantee. Keeping all 224 rain and 1,000 fog tests retains their background
diversity. Snow selects 256 known sequence clusters with city-proportional quotas
and at least one sequence per city, then one background per selected sequence and
three degradation-rank strata. Smoke retains all six backgrounds in each held-out
partition, with up to 16 images per stratum (64/background). The smaller test smoke
background has only 22 distinct pairs, giving 342 overall. More correlated frames
would mainly increase evaluation time. Equal weather counts would not create equal
statistical evidence.

Selection uses fixed seed **20261015** and identifiers, never restored-model scores.
Strata are within-background ranks of mean absolute input/target difference in
16x16 RGB thumbnails: four bins, or three for snow. This is a reproducible proxy
for degradation strength, **not measured physical density**, and may reflect motion
or alignment error. Singleton backgrounds have one bin. All strata are frozen;
there is no inference-time use of ground truth.

## Sampling and reproducibility

Each global batch of four contains one rain, fog, snow, and smoke observation.
Within weather, traverse a shuffled permutation of backgrounds, then shuffled
strata within each background, then shuffled images within each stratum. Repeat
only after completing the corresponding cycle. This is sampling without replacement
within cycles and with reuse between cycles; it is not independent sampling with
replacement. Streams carry across epoch boundaries. An epoch is 4,096 draws (1,024
per weather), not a full pass through 18,610 images. At that rate all fog backgrounds
are reached within five epochs rather than restarting coverage every epoch.

Over 20 epochs each weather receives 20,480 draws. Small background pools, especially
smoke's 31, necessarily recur often; equal weighting prevents its thousands of
related frames from dominating. More *independent smoke backgrounds* are the main
data gap. No sampler can invent them. Additional paired smoke data should undergo
licensing, alignment and cross-source overlap checks and get a separate future
protocol version. Do not add unpaired smoke images to a supervised PSNR/SSIM loss.

Crops/flips use a hash of training seed, weather, and global draw index, making
transforms independent of worker count and GPU assignment. Split seed is separate
from fine-tuning seeds **42, 43, 44**. Four A100s process one image each; one/two GPUs
use the same global batch and learning rate. Four GPUs can improve throughput, but
DDP communication and validation overhead prevent a promised 4x speedup. Exact
resume requires the same process count and configuration. Deterministic kernels are
requested; floating-point reductions can still differ across GPU counts/software.

## Matched experiments and selection

Run seed42 L1 and L1+0.05 SSIM pilots, both initialized from the completed shared
seed42 model, with fresh AdamW state, 20 epochs, learning rate 2e-5 decaying to 1e-6,
global batch four, 256 crops, native-resolution validation, and bf16. Everything
except the SSIM loss term is matched. SSIM uses RGB uniform 7x7 windows and sample
covariance, matching the evaluation implementation. L1 is the control; adding SSIM
may trade off PSNR, so it is an experiment, not an assumed improvement. Architecture,
GAN loss and MSE loss are not changed simultaneously.

Before training, measure the initialization on the expanded validation set. A
candidate must retain **every weather's** background/stratum-balanced PSNR and SSIM
relative to this initialization and improve the best macro PSNR by more than 0.02 dB.
This minimum change is a selection threshold, not a significance claim. If nothing
qualifies, `best.pt` is absent and the existing model is retained; `last.pt` still
allows inspection/resume. Guards reduce hidden regressions but cannot guarantee
test gains or prevent all validation overfitting.

Choose the loss using validation only: highest eligible macro PSNR; within 0.02 dB,
prefer higher macro SSIM, then the simpler L1 on a tie. Run that loss with seeds43
and44 without retuning. These are three *fine-tuning seeds from a shared pretrained
initialization*, not three independent pretraining runs. Report every declared seed,
including failed selection/no qualifying checkpoint; do not drop unsuccessful seeds
or substitute the best test seed. If a seed fails, keep the failure explicit and
avoid presenting a success-only three-seed summary.

## Reporting and uncertainty

The primary estimand averages images within stratum, strata within background,
backgrounds within weather, then the four weathers equally. Report each weather,
raw input scores, PSNR/SSIM gains and counts of images made worse. Pooled per-image
metrics remain secondary. Training selection and final evaluation use the same
weighting; score changes due to changed weighting are not model improvements.

`compare_weather_protocol.py` verifies source/result hashes, exact pair identity,
scoring settings and raw metrics, then compares saved baseline and candidate on the
same 2,334 pairs. It produces paired cluster-bootstrap 95% percentile intervals
(2,000 replicates, fixed seed), resampling entire clusters within weather, plus
leave-one-cluster-out ranges. Identical pairs are preserved in baseline/candidate
resampling. Intervals are marginal, not simultaneous tests over all metrics/weathers.
They describe uncertainty conditional on this model and chosen test, not deployment
risk or training-seed variance. `summarize_weather_seeds.py` separately reports mean
and sample standard deviation across the three declared fine-tuning seeds.

The completed baseline on v3 is **32.7306 dB / 0.929878 SSIM**. Its per-weather scores
are rain 38.7153/0.977201, fog 31.8759/0.968112, snow 39.5768/0.981763, and smoke
20.7546/0.792435. These are re-aggregated existing predictions, not new training gains.
The report is `artifacts/restoration/four_weather_restormer_seed42/review/weather_v3_baseline.json`.

Limits: smoke has only six validation and six test backgrounds, so bootstrap
intervals are exploratory. Renamed snow training files and rain files lack recoverable
capture-sequence metadata. Exact/near-copy grouping cannot certify every semantic
scene's independence. These test images have been evaluated previously; v3 is not
an untouched external benchmark. Validate claims on new independent data before
claiming generalization beyond these sources. Group separation follows the
[scikit-learn guidance on dependent observations](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

## DGX commands

Use `artifacts/transfer/aegis_weather_v3_code.tar.gz` and its `.sha256` sidecar.
The archive contains code, configs, manifests and audit/baseline reports; images
and weights are excluded. All 41,398 unique active image paths are already included
in the original `aegis_four_weather.tar` data bundle. Verify the update checksum,
then extract it into the existing DGX checkout, preserving data/checkpoints.
The original frozen bundle predates the quota discovery and contains four-GPU/48000M
launch defaults. After extracting it, apply the resource correction below; the
archive and its checksum remain unchanged:

```bash
sha256sum -c aegis_weather_v3_code.tar.gz.sha256
tar -xzf aegis_weather_v3_code.tar.gz -C /path/to/existing/repository
sed -i -e 's/^#SBATCH --gres=.*/#SBATCH --gres=gpu:a100:1/' \
       -e 's/^#SBATCH --mem=.*/#SBATCH --mem=32000M/' \
       /path/to/existing/repository/scripts/restoration/train_weather_v3_dgx.sbatch
```

Submit from
the DGX repository root. The updated script requests **one full A100 and 32000M host RAM**,
matching the account limits reported on 15 October 2026, and checks the manifest,
archives source and environment, and resumes only a compatible run. It deliberately
does not evaluate test automatically.

```bash
sbatch scripts/restoration/train_weather_v3_dgx.sbatch configs/restoration/weather_v3_l1_seed42.yaml
sbatch scripts/restoration/train_weather_v3_dgx.sbatch configs/restoration/weather_v3_ssim_seed42.yaml
```

Submit the second pilot with an `afterok` dependency on the first to run sequentially.
The account permits only one full A100 at a time. Compare validation
histories and `selection_status.json`, then submit the selected loss's seed43 and
seed44 configs. Full validation is larger than before; check observed epoch time
against the 20-hour allocation and resubmit to resume if needed.

After freezing the loss choice, evaluate each qualifying seed on an allocated GPU
node (replace the example loss/seed as appropriate):

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 scripts/restoration/evaluate.py \
  --config configs/restoration/weather_v3_ssim_seed42.yaml \
  --checkpoint artifacts/restoration/weather_v3_ssim_seed42/best.pt --device cuda
python scripts/restoration/compare_weather_protocol.py \
  --candidate artifacts/restoration/weather_v3_ssim_seed42/metrics_test.json \
  --output artifacts/restoration/weather_v3_ssim_seed42/comparison.json
# Repeat evaluation/comparison for 43 and 44, then:
python scripts/restoration/summarize_weather_seeds.py \
  artifacts/restoration/weather_v3_ssim_seed42/comparison.json \
  artifacts/restoration/weather_v3_ssim_seed43/comparison.json \
  artifacts/restoration/weather_v3_ssim_seed44/comparison.json \
  --output artifacts/restoration/weather_v3_seed_summary.json
```

The frozen preparation script can reproduce membership from the audited local
source/cache into a **new directory**. Never regenerate splits on the DGX or
change manifests mid-run. Existing v2 configurations remain historical; do not use
them for these experiments. CPU tests exercise real two-process DDP, but A100
memory/throughput and model quality must be measured by the submitted jobs.

### Correct already pending pilot jobs after quota discovery

The reported user association limits are `cpu=8,gres/gpu:a100=1,mem=32000M`,
with a separate one-slice `a100_1g.5gb` limit. The slice is not a second full A100.
The four-GPU request was accepted into the queue but exceeded these limits.
Correct pending jobs in place, preserving their dependency and IDs:

```bash
scontrol update JobId=47510 Gres=gpu:a100:1 MinMemoryNode=32000
scontrol update JobId=47511 Gres=gpu:a100:1 MinMemoryNode=32000
squeue -j 47510,47511
```

Changing the on-disk script only affects future submissions; pending job resources
must also be updated. The launcher discovers allocated GPU count, so global batch
four now runs entirely on one GPU with the same sampling/loss/configuration.
Other shared limits or competing jobs can still delay scheduling. Administrator
approval and a quota change are needed for four full A100s on this account.
