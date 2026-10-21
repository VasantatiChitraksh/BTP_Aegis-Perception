# Aegis Perception: Antigravity handoff and command guide

Snapshot: 7 October 2026, India time. This is a repository handoff, not an instruction to rerun all experiments. Read the status first, then use only the command section needed for the next action.

## 1. Starting point and current status

The verified laptop repository is **`/home/bms/Desktop/BTP_Aegis-Perception`**. The user described it approximately as `/desktop/btp_aegs_perception`; Linux paths and capitalization matter. If it has actually been moved, replace the local path below with its real path. Do not create a second checkout just because the spelling differs.

```bash
# LOCAL laptop terminal: establish the working directory and transfer variables.
export AEGIS_LOCAL_REPO='/home/bms/Desktop/BTP_Aegis-Perception'
export AEGIS_SSH='cs23b051@10.23.20.36'
export AEGIS_REMOTE_REPO='/scratch/cs23b051/aegis-perception'
cd "$AEGIS_LOCAL_REPO"
pwd
git status --short
git log -3 --oneline
```

At handoff, local HEAD is `40b02f5` (`Rain-training-completed`); the original training metadata records `3aa4ef4`. The worktree was clean before this handoff document was added. Local and remote versions should be compared before synchronizing; remote state has not been inspected live during this handoff.

**Completed:** one seed-42, weather-specific Restormer rain experiment, pretrained comparison, test metrics, three restored examples, comparison figures, and a six-page PDF report. No rain retraining is needed simply to continue in another IDE.

**Next agreed direction:** extend restoration to additional weather types, establish separate weather baselines, then compare against one jointly trained model. Detection and hardware deployment follow restoration. Choose research changes from measured weaknesses. Do not add a weather classifier/router by default.

**Not completed:** additional seeds, unseen rain datasets, scene near-duplicate audit, fog training, validated snow/smoke Restormer runs, joint-weather training, detection comparisons, quantization, TensorRT, or target-hardware measurements. Workshop name/deadline and deployment hardware are not confirmed in the current handoff.

## 2. What runs where

| Location | Purpose | Terminal prompt / path |
|---|---|---|
| Laptop | Edit code, prepare/inspect data, receive backups, view figures/PDF | `bms@sky`, local repository |
| DGX login node | Environment installation, transfers, audits, SLURM submission/monitoring | `cs23b051@loginnode` |
| DGX compute node | Allocated training/evaluation/inference GPU work | `cs23b051@dgx1`, inside a SLURM allocation |

The working compute device is A100 40 GB. A `a100_3g.20gb` MIG allocation is a slice of an A100, not a separate full GPU. A login-node GPU check once reported an RTX 3050; do not train or run GPU inference on that node outside SLURM.

Dataset images live on disk. The dataloader reads mini-batches/crops into CPU/GPU memory; it does not load the whole dataset into VRAM. The current workflow uses DGX `/scratch`, not S3 or Lightning. `/scratch` is shared cluster storage, not a guaranteed permanent backup; retain experiment backups on the laptop. Large image/weight files are ignored by Git.

## 3. Quick return to the existing working DGX setup

Run SSH separately, then wait for the remote prompt. Commands pasted below `ssh` in a local multi-line shell block can otherwise execute on the laptop only after the SSH session closes.

```bash
# LOCAL. Institute network access is required for this private IP.
ssh cs23b051@10.23.20.36
```

```bash
# REMOTE login node, after SSH has opened.
cd /scratch/"$USER"/aegis-perception
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1
eval "$(conda shell.bash hook)"
conda activate pytorch_12.1
source .venv/bin/activate

python -c 'import sys, torch, numpy; print(sys.executable); print("Torch:", torch.__version__); print("NumPy:", numpy.__version__)'
python -m pip check
```

Expected: project `.venv/bin/python`, **Torch `2.5.1+cu118`**, **NumPy `1.26.4`**. Conda `pytorch_12.1` supplies Python 3.10; its own Torch is `2.5.1+cu121` and is incompatible with the recorded driver. The isolated project virtual environment supplies the working CUDA 11.8 Torch instead.

No passwords are stored here. Use the account's current password through the SSH prompt.

## 4. Fresh remote checkout and environment, only if absent

For an account without a checkout:

```bash
# REMOTE login node.
mkdir -p /scratch/"$USER"
df -h /scratch/"$USER"
quota -s 2>/dev/null || true
sinfo -p longq
cd /scratch/"$USER"

# Run clone only if aegis-perception does not already exist.
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git clone \
  https://github.com/VasantatiChitraksh/BTP_Aegis-Perception.git \
  aegis-perception

cd aegis-perception
```

`mkdir` creates your workspace; `df` shows shared filesystem capacity, not your personal quota; `sinfo` shows partition status. Capacity changes, so inspect it before large transfers. Clearing library variables for this one Git command fixes the observed Conda/libffi HTTPS conflict without changing the whole shell.

Install on **the login node**: the compute node previously could not resolve package-download hosts. A fresh setup downloads several GB of Torch/CUDA packages. These commands install only into `.venv`; do not modify shared `/apps/anaconda3` environments.

```bash
# REMOTE login node, repository root. Skip this block if the working .venv exists.
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1
eval "$(conda shell.bash hook)"
conda activate pytorch_12.1

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel --index-url https://pypi.org/simple

python -m pip install --no-cache-dir 'torch==2.5.1+cu118' \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple

python -m pip install 'numpy==1.26.4' -e '.[metrics,dev]' \
  --index-url https://pypi.org/simple

python -m pip check
python - <<'PY'
import sys, torch, numpy, yaml, tqdm
from PIL import Image
print('Python:', sys.version.split()[0])
print('Interpreter:', sys.executable)
print('Torch:', torch.__version__)
print('NumPy:', numpy.__version__)
assert torch.__version__ == '2.5.1+cu118'
assert numpy.__version__ == '1.26.4'
PY
```

This is for a **new** environment; reuse the existing verified one otherwise. It deliberately includes PyPI for normal dependencies: a previous install using only the PyTorch index failed while resolving `typing_extensions`/`flit_core`.

Recorded compatibility issues:

- Driver `470.161.03` could not initialize Torch cu121; cu118 worked in this cluster. Recheck if administrators change the driver.
- Shared `pytorch_gpu1` had Torch 2.0.1, below the project's `torch>=2.4` requirement.
- NumPy 2 caused warnings with that old Torch; the successful isolated environment uses NumPy 1.26.4.
- `PIL` is supplied by the **Pillow** package; it is not a package named PIL.
- `sbatch --wrap` normally runs `/bin/sh`; using Bash-only `source` there previously failed and selected Conda's wrong Torch. Use the explicit Bash batch example below or the working wrappers.

## 5. Optional laptop Python setup

The existing local data/results can be viewed without recreating a training environment. At handoff no repository-local `.venv` was found on the laptop. Earlier CPU verification used a temporary environment at `/tmp/aegis-validation-check`; do not depend on that temporary directory surviving.

For CPU tests and analysis in a fresh local environment:

```bash
# LOCAL, repository root. Create only if .venv is absent.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel --index-url https://pypi.org/simple
python -m pip install 'torch==2.5.1+cpu' \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
python -m pip install 'numpy==1.26.4' -e '.[metrics,dev,notebook]' \
  --index-url https://pypi.org/simple
python -m pip check
```

Do not copy a laptop `.venv` onto DGX: Python environments are interpreter/platform/path dependent. Native-resolution full Restormer inference can exhaust laptop RAM; run that on an allocated GPU.

## 6. Git, code synchronization, and data transfers

Inspect changes first. For an existing clean remote checkout, fetch the published branch without discarding work:

```bash
# REMOTE repository root.
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git status --short
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git fetch origin
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git log -3 --oneline origin/main
env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/git pull --ff-only
```

If pull reports local modifications/conflicts, inspect them rather than resetting the checkout. `git pull` does not carry ignored images, checkpoints, PDFs, or virtual environments.

If the laptop is the intended source of updated code, this explicitly copies its source/configuration/docs without images, weights, `.git`, or environment folders. Review remote edits before overwriting the same files.

```bash
# LOCAL; establish the variables from section 1 in each new local shell.
cd "$AEGIS_LOCAL_REPO"
rsync -avR --exclude='__pycache__/' --exclude='*.egg-info/' \
  src scripts configs data/manifests data/datasets.yaml docs README.md pyproject.toml \
  "$AEGIS_SSH:$AEGIS_REMOTE_REPO/"
```

Upload the rain data and weights into an existing remote checkout:

```bash
# LOCAL.
ssh "$AEGIS_SSH" \
  'mkdir -p /scratch/cs23b051/aegis-perception/data/raw/rain/realrain1k/RealRain-1k/RealRain-1k-H /scratch/cs23b051/aegis-perception/artifacts/pretrained'

rsync -a --partial --info=progress2 \
  data/raw/rain/realrain1k/RealRain-1k/RealRain-1k-H/ \
  "$AEGIS_SSH:$AEGIS_REMOTE_REPO/data/raw/rain/realrain1k/RealRain-1k/RealRain-1k-H/"

rsync -a --partial --info=progress2 artifacts/pretrained/deraining.pth \
  "$AEGIS_SSH:$AEGIS_REMOTE_REPO/artifacts/pretrained/"
```

The trailing slash on the source data directory copies its **contents**. Rerun the same `rsync` after an interrupted transfer; `--partial` retains partial files. Never add `--delete` to these backup/transfer commands.

For snow, the same pattern applies to an audited source, e.g. CSD:

```bash
# LOCAL. Creates a destination; does not start training.
ssh "$AEGIS_SSH" 'mkdir -p /scratch/cs23b051/aegis-perception/data/raw/snow/csd'
rsync -a --partial --info=progress2 data/raw/snow/csd/ \
  "$AEGIS_SSH:$AEGIS_REMOTE_REPO/data/raw/snow/csd/"
```

CSD occupies approximately 23 GB including masks; the current paired dataloader uses `Snow`/`Gt`, not masks. SnowCityscapes is approximately 3.5 GB. Verify remote space first. Only the rain transfer was confirmed during the completed experiment; presence of other datasets on DGX is not established.

## 7. Dataset preparation and current blockers

All existing local manifests currently resolve to present input/target files. File existence alone does not establish correct alignment or independent splits.

| Source | Local train / val / test pairs | Current status |
|---|---:|---|
| RealRain-1k-H | 784 / 112 / 224 | Completed baseline; byte-duplicate audit passed; near-scene audit still pending |
| Rain100L | 0 / 0 / 100 | Independent test candidate; no training split in this collected copy |
| O-HAZE | 0 / 0 / 45 | Reserved paired fog test benchmark; no fog training source prepared |
| CSD | 7,215 / 785 / 2,000 | Downloaded synthetic snow; full audit and native-resolution GPU preflight still needed |
| SnowCityscapes | 5,382 / 618 / 6,000 | Downloaded synthetic road snow; variants grouped by target ID in builder; verify scene/sequence separation |
| Historical smoke | 98 / 12 / 12 | Audit rejects four clean-target hashes crossing train/val |
| SmokeBench | 8,918 / 957 / 100 | Audit rejects severe shared-background leakage; details below |

**SmokeBench finding from this session:** 9,975 pairs contain only **43 distinct byte-identical clean reference images**. All 43 target hashes appear in both train and validation. Four also appear in test; **all 100 test pairs share a clean reference with training**. This is not 9,975 independent backgrounds. Repair scene/background grouping and establish an independent evaluation before reported training. Preserve the downloaded originals and document changed protocols; do not quietly reshuffle a published test set to improve scores.

**Fog:** `configs/restoration/fog_attention.yaml` points to `data/manifests/fog.csv`, which does not exist. O-HAZE is test-only here. RTTS and Foggy Driving have annotations but no matched clean targets. Cataloged paired options need access/source verification; none was prepared during this session.

**Detection:** DAWN rain/fog/snow, RTTS, and Foggy Driving are local resources, but detector label mapping and fixed views/splits are not prepared. Smoke-plume masks/boxes are not labels for cars/people behind smoke.

List/download/extract only a selected missing source:

```bash
# In an activated environment, repository root. CPU preparation is sufficient.
python scripts/data/download_datasets.py --list
python scripts/data/download_datasets.py --dataset csd
python scripts/data/extract_datasets.py --verify --dataset csd
python scripts/data/extract_datasets.py --extract --dataset csd
```

Archives: `data/archives/`; extracted data: `data/raw/`; derived detector views: `data/processed/`; manifests: `data/manifests/`. Existing data need not be downloaded/extracted again. CSD RAR5 extraction needs current `7zz`/7-Zip; old p7zip may fail. Do not download `--core` merely to resume the project.

Run audits, preserving existing split files:

```bash
python scripts/data/audit_manifest.py data/manifests/realrain1k.csv
python scripts/data/audit_manifest.py data/manifests/csd.csv
python scripts/data/audit_manifest.py data/manifests/snow_cityscapes.csv
python scripts/data/audit_manifest.py data/manifests/smoke_historical.csv
python scripts/data/audit_manifest.py data/manifests/smokebench.csv
```

The last two currently fail for confirmed leakage. This failure is expected and must not be bypassed with `--skip-hashes`. Exact hashes cannot detect all re-encoded or nearby video frames. Manually inspect scene/sequence grouping as well.

`build_dataset_manifests.py --dataset ID` and `build_pair_manifest.py --help` exist for establishing new/repaired manifests. Builders can overwrite an existing CSV, so do not rebuild frozen rain splits just to change a training seed. Seeds 43/44 change training randomness, not dataset membership.

If the original pretrained weight file is missing on a fresh machine, its verified download is also recorded in the notebook. Use the login node or laptop with network access, not a compute allocation:

```bash
# Activated environment, repository root; only necessary when weights are absent.
python -m pip install gdown --index-url https://pypi.org/simple
python - <<'PY'
from pathlib import Path
import hashlib, gdown
weights = Path('artifacts/pretrained/deraining.pth')
weights.parent.mkdir(parents=True, exist_ok=True)
if not weights.exists():
    gdown.download('https://drive.google.com/uc?id=1uuejKpyo0G_5M4DAO2J9_Dijy550tjc5', str(weights), quiet=False, use_cookies=False)
assert hashlib.sha256(weights.read_bytes()).hexdigest() == '225ed9acf80b3527d7080e5182918f642a23a78ef8593ad700b7e924bd660681'
print('Official weights verified')
PY
```

Training reads `deraining.pth` directly. To create the project's epoch-0 checkpoint for baseline evaluation, only when absent:

```bash
# Activated environment with Torch; CPU is sufficient for packaging the weights.
python - <<'PY'
from pathlib import Path
from aegis_perception.config import load_config
from aegis_perception.checkpoints import load_pretrained, save_checkpoint
from aegis_perception.data.catalog import sha256
from aegis_perception.models import build_generator
config = load_config('configs/restoration/rain_restormer.yaml')
output = Path(config['run']['output_dir']) / 'pretrained.pt'
if output.exists():
    raise SystemExit('Existing pretrained.pt retained; no changes made')
model = build_generator(config['model'])
load_pretrained(model, config['model']['pretrained'])
save_checkpoint(output, {
    'generator': model.state_dict(), 'config': config, 'epoch': 0,
    'manifest_sha256': sha256(Path(config['data']['manifest'])),
    'pretrained_sha256': sha256(Path(config['model']['pretrained'])),
    'pretrained_source': 'https://drive.google.com/uc?id=1uuejKpyo0G_5M4DAO2J9_Dijy550tjc5',
})
print(output)
PY
```

## 8. SLURM job submission and GPU checks

A job is an independent command/script plus a resource request, scheduled by SLURM. Closing SSH or the laptop does not cancel a submitted **batch** job. An interactive `srun --pty` session depends on its terminal and is for short checks.

```bash
# REMOTE login node: short interactive allocation, only when a GPU check is needed.
srun --partition=longq --qos=longq --gres=gpu:a100_3g.20gb:1 \
  --cpus-per-task=2 --mem=12000M --time=00:15:00 --pty bash
```

Once the prompt becomes `@dgx1`, activate the environment as in section 3 and check real GPU computation:

```bash
# ALLOCATED compute shell only.
nvidia-smi
python - <<'PY'
import torch
print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
assert torch.__version__ == '2.5.1+cu118'
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(256,256,device='cuda',requires_grad=True)
(x @ x.T).mean().backward()
torch.cuda.synchronize()
print('GPU forward/backward passed')
PY
exit
```

`exit` releases the interactive allocation. A small tensor check does not prove full-image validation will fit.

The known account memory allowance was `32000M`; `--mem=32G` requested 32768 MiB and caused `AssocGrpMemLimit`. The working full job request is `--mem=30000M`. This is **system RAM**, independent of GPU VRAM. Other jobs in the association can also consume limits.

### Existing rain wrapper (already completed)

```bash
# REMOTE repository root. Reference only: no need to rerun the completed seed-42 run.
export AEGIS_CONDA_ENV=pytorch_12.1
export PYTHONUNBUFFERED=1
AEGIS_JOB=$(sbatch --parsable --mem=30000M --gres=gpu:a100:1 \
  scripts/restoration/train_rain_dgx.sbatch)
echo "$AEGIS_JOB"
```

This wrapper hardcodes the rain config and seed-42 `last.pt`. It resumes if that checkpoint exists; for the completed 20-epoch run it prints `Training already complete`. It is not a general fog/snow/seed-43 submission script.

### General training submission for a separately prepared config

First create/review the config and its unique `run.name`/`run.output_dir`, then audit its manifest. The example below uses the existing rain config only to show the interface; a **fresh** attempt in its existing directory will deliberately be rejected. Replace `AEGIS_CONFIG` with the new experiment config before launching a new run.

```bash
# REMOTE repository root. Review the chosen config before submission.
export AEGIS_CONFIG='configs/restoration/rain_restormer.yaml'
export AEGIS_RESUME=''
# For an interrupted SAME run instead:
# export AEGIS_RESUME='artifacts/restoration/rain_restormer_seed42/last.pt'

AEGIS_JOB=$(sbatch --parsable --job-name=aegis-train \
  --partition=longq --qos=longq --gres=gpu:a100:1 \
  --cpus-per-task=4 --mem=30000M --time=24:00:00 \
  --output=/scratch/%u/aegis-train-%j.out \
  --error=/scratch/%u/aegis-train-%j.err <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1
eval "$(conda shell.bash hook)"
conda activate pytorch_12.1
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python -c 'import torch; assert torch.__version__ == "2.5.1+cu118"; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), flush=True)'
if [[ -n "${AEGIS_RESUME:-}" ]]; then
  exec .venv/bin/python scripts/restoration/train.py --config "$AEGIS_CONFIG" --resume "$AEGIS_RESUME"
else
  exec .venv/bin/python scripts/restoration/train.py --config "$AEGIS_CONFIG"
fi
SBATCH
)
echo "Submitted training job: $AEGIS_JOB"
```

This inline script explicitly uses Bash and the project interpreter. It relies on SLURM's normal export of the exported config/resume variables. Native validation runs before optimization for fresh runs. Resume requires matching model/data/train config, seed, and manifest checksum. To change architecture, seed, dataset, or schedule, use a new run, not `--resume` with altered settings.

**Do not use `scripts/train_on_dgx.sh` as-is.** It is stale: activates `pytorch_gpu1`, omits the working `.venv`, and lacks the robust setup used above.

## 9. Monitor, inspect, and stop jobs

```bash
# REMOTE login node. Replace 47223 with the NEW numeric job ID when monitoring.
squeue -u "$USER"
squeue -p longq
squeue -j 47223 -o '%.18i %.10T %.12M %.12l %R'
squeue --start -j 47223
scontrol show job 47223
sacct -j 47223 --starttime=2026-10-06 \
  --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem
```

- `PD`: pending; `R`: allocated/running. An `R` entry alone does not prove model training is progressing; check the logs.
- `Resources`: requested resources unavailable. `Priority`: scheduling priority. `AssocGrpMemLimit`: association memory request/usage exceeds its allowance.
- `squeue --start` is an estimate, can show `N/A`, and is not a promised start time.
- Completed/failed jobs disappear from `squeue`; use `sacct`. An invalid old job ID in `squeue` does not imply lost progress.
- Walltime starts when the job runs; queued time does not consume it.
- Shell variables such as `AEGIS_JOB` disappear in a new SSH session. Record numeric IDs and use those after reconnecting.

```bash
# REMOTE: watch a newly submitted general training job in the SAME shell.
tail -F /scratch/"$USER"/aegis-train-"$AEGIS_JOB".out \
        /scratch/"$USER"/aegis-train-"$AEGIS_JOB".err
```

`Ctrl+C` stops `tail`/`watch`, not the batch job. `tail -F` retries until logs exist. Progress bars normally write to `.err`; a progress bar there is not itself an error. A traceback or nonzero exit status is an actual failure.

To deliberately cancel a specific job: `scancel NUMERIC_JOB_ID`. Cancellation can lose work since the last epoch checkpoint; it does not delete saved checkpoints. Inspect the exact ID before canceling.

## 10. Evaluate a selected or pretrained checkpoint

```bash
# REMOTE repository root. Completed seed-42 results already exist; avoid rerunning without a reason.
export AEGIS_CONDA_ENV=pytorch_12.1
AEGIS_EVAL_JOB=$(sbatch --parsable scripts/evaluate_on_dgx.sh \
  configs/restoration/rain_restormer.yaml \
  artifacts/restoration/rain_restormer_seed42/best.pt \
  --output artifacts/restoration/rain_restormer_seed42/metrics_test_recheck.json)
echo "$AEGIS_EVAL_JOB"
tail -F /scratch/"$USER"/aegis_eval-"$AEGIS_EVAL_JOB".out \
        /scratch/"$USER"/aegis_eval-"$AEGIS_EVAL_JOB".err
```

For pretrained comparison, pass `pretrained.pt` and a distinct output filename. The wrapper defaults to test; append `--split val` for validation. Preserve the original metrics files when rechecking. Pretrained `.pth` files are not the project's checkpoint format: `load_generator` expects `generator` and `config`, as packaged in `pretrained.pt` by the notebook.

Evaluation enforces matching model config and split manifest checksum. RGB `[0,1]`, native dimensions, per-image PSNR/SSIM averaged across the split; MSE floor `1e-12`. The `128 x 128` config size is the training crop, not a resizing rule when `eval_mode: native`.

## 11. Restore an image folder without reevaluating the whole dataset

Prepare a separate input folder of desired images. Outputs must be outside the input tree. The three qualitative inputs already exist on DGX at the path below; if starting a new checkout, copy those images from the source test set first.

```bash
# REMOTE repository root. Same direct-interpreter approach that succeeded for job 47228.
AEGIS_VIS_JOB=$(sbatch --parsable --job-name=aegis-vis \
  --partition=longq --qos=longq --gres=gpu:a100_3g.20gb:1 \
  --cpus-per-task=2 --mem=12000M --time=00:15:00 \
  --output=/scratch/%u/aegis-vis-%j.out \
  --error=/scratch/%u/aegis-vis-%j.err \
  --wrap='. /etc/profile.d/modules.sh
module load anaconda/2023.03-1
cd /scratch/cs23b051/aegis-perception
exec .venv/bin/python scripts/restoration/restore_images.py \
  --config configs/restoration/rain_restormer.yaml \
  --checkpoint artifacts/restoration/rain_restormer_seed42/best.pt \
  --input-root artifacts/restoration/rain_restormer_seed42/qualitative/input \
  --output-root artifacts/restoration/rain_restormer_seed42/qualitative/restored \
  --device cuda')
echo "$AEGIS_VIS_JOB"
tail -F /scratch/"$USER"/aegis-vis-"$AEGIS_VIS_JOB".out \
        /scratch/"$USER"/aegis-vis-"$AEGIS_VIS_JOB".err
```

This 20 GB allocation worked for test images 107, 257, and 338. It does **not** guarantee arbitrary native images fit. All-112 native rain validation peaked at about 22.28 GiB allocated; use a full 40 GB A100 for the proven training/evaluation protocol. Larger images in a new dataset need a preflight before a full run. Do not quietly resize or tile test images and compare against native metrics as if the protocol were unchanged.

## 12. Back up and inspect results

```bash
# LOCAL repository root, with transfer variables from section 1.
mkdir -p artifacts/restoration/rain_restormer_seed42
rsync -a --partial --info=progress2 \
  "$AEGIS_SSH:$AEGIS_REMOTE_REPO/artifacts/restoration/rain_restormer_seed42/" \
  artifacts/restoration/rain_restormer_seed42/

ls -lh artifacts/restoration/rain_restormer_seed42/
python3 -m json.tool artifacts/restoration/rain_restormer_seed42/metrics_test.json
xdg-open artifacts/restoration/rain_restormer_seed42/rain_training_report.pdf
```

The report/figures were generated locally; synchronize direction carefully if the remote also has files with those names. Use `rsync -an` first to preview a planned copy. There is no `--delete` here. To inspect a checkpoint epoch without inference, in an environment with Torch:

```bash
python - <<'PY'
from pathlib import Path
import torch
root = Path('artifacts/restoration/rain_restormer_seed42')
for name in ('best.pt','last.pt','pretrained.pt'):
    ckpt = torch.load(root/name,map_location='cpu',weights_only=False,mmap=True)
    print(name,'epoch:',ckpt.get('epoch'),'best validation:',ckpt.get('best_psnr_db'))
PY
```

## 13. Completed rain experiment: preserve these facts

| Item | Recorded result |
|---|---|
| Model | Official Restormer, 26,126,644 parameters; strict official deraining initialization |
| Training | Seed 42; 20 epochs; batch 4; paired random 128-pixel crops/flip; 3,920 updates |
| Objective / optimizer | RGB-equivalent L1, no GAN; AdamW lr 1e-4, betas 0.9/0.999, decay 1e-4; cosine toward 1e-6; clip 1.0 |
| Device / environment | Full A100 40 GB; Python 3.10.18; Torch 2.5.1+cu118; NumPy 1.26.4; BF16 |
| Selected checkpoint | **best.pt, epoch 15**; validation PSNR 37.2741 dB |
| Latest checkpoint | **last.pt, epoch 20**; final validation PSNR 36.9392 dB |
| Raw test input | PSNR 22.2721 dB; SSIM 0.741376 |
| Pretrained test | PSNR 23.1265 dB; SSIM 0.794034 |
| Fine-tuned test | **PSNR 37.2565 dB; SSIM 0.972978** |
| Mean PSNR improvement | +14.9844 dB over raw; +14.1300 dB over pretrained |
| Per-image findings | All 224 improved PSNR over raw; two lost SSIM vs raw (test/107, test/863); two lost PSNR vs pretrained |
| Runtime / memory | Training job elapsed 39m49s; mean epoch ~114.5s including validation; peak allocated CUDA ~22.28 GiB |
| Job IDs | 47223 training; 47224 fine-tuned test; 47225 pretrained test; 47228 three-image restoration |

All these jobs were reported complete; no current queue state was queried for this handoff. Do not assume an old ID is still running.

Artifact directory on laptop: `artifacts/restoration/rain_restormer_seed42/` (about 1.9 GB before PDF/figures).

- `best.pt`: use for inference and reported comparisons.
- `last.pt`: use for resuming the same run.
- `epoch_005.pt`, `epoch_010.pt`, `epoch_015.pt`, `epoch_020.pt`: periodic full-state snapshots.
- Trained files ~300 MiB include model, optimizer, scheduler/scaler, RNG, config, history, and validation. They are not different architectures. `pretrained.pt` ~100 MiB is the untouched epoch-0 baseline.
- `run.json`, `history.json`, `baseline_validation.json`, both metrics JSONs, `environment.txt`, code archive, and logs retain the evidence.
- `rain_training_report.pdf`: six-page completed-experiment report.
- `qualitative/rain_restoration_comparison.png` and `.pdf`: weak/median/strong gain examples; original restored images in `qualitative/restored/`.
- Original official weights: `artifacts/pretrained/deraining.pth`.

Verified checksums:

```text
Official deraining.pth SHA-256:
225ed9acf80b3527d7080e5182918f642a23a78ef8593ad700b7e924bd660681
Frozen realrain1k.csv SHA-256:
b03a7799863f5e50e79140940fcb329cfbe368267a76a62e5a6e61c8869f31c5
Recorded training source SHA-256:
4314bcb8b0fd729bee1017ecbc730662a9195a57d74f5d00ad4d69c70c9fbdb3
```

Check with `sha256sum artifacts/pretrained/deraining.pth data/manifests/realrain1k.csv`. The training source included working-tree changes beyond the then-recorded Git commit; retain the source archive and run record.

Infrastructure fixes already present: configured BF16 in training/validation/evaluation/folder restoration; full validation preflight; finite-loss checks; depthwise convolutions chunked by independent channels when older CUDA indexing limits are exceeded; strict loading and resumable checkpoints. The convolution fix preserves spatial context and weights; it is not image tiling.

L1 is a legitimate baseline objective; no loss ablation establishes it as superior. Results are a single-dataset/single-seed fine-tuning baseline, not a novel method or evidence of detector/real-time benefit. A validation plateau does not prove a longer schedule cannot improve. No matched FP32 comparison was run.

## 14. IDE context and repository map

Latest active/open file from user IDE context: `artifacts/restoration/rain_restormer_seed42/rain_training_report.pdf`. Earlier tabs included `docs/experiments.md`, `scripts/restoration/train_rain_dgx.sbatch`, `scripts/data/audit_manifest.py`, `src/aegis_perception/training.py`, and `archive/papers/links.txt`.

| Area | Role / important files |
|---|---|
| `src/aegis_perception/models/restormer.py` | Architecture and wrapper; checkpoint weights do not replace source architecture |
| `src/aegis_perception/models/__init__.py` | Factory selects architecture from config |
| `src/aegis_perception/training.py` | Shared optimization, validation, checkpoint, resume logic |
| `src/aegis_perception/inference.py` | Padding, configured precision, prediction, crop-back |
| `src/aegis_perception/data/paired.py` | Paired crops/flips, normalization, native loading |
| `scripts/restoration/` | CLI entry points; thin wrappers around `src` |
| `configs/` | Recipes, data paths, output directory, seed, architecture settings |
| `data/` | Catalog, frozen manifests, ignored archives/raw/processed data |
| `artifacts/` | Ignored run evidence, models, images, report |
| `tests/` | Pipeline/precision/checkpoint and related regression checks |
| `docs/` | Existing dataset, experiment, and literature guidance |
| `archive/` | Historical evidence and paper PDFs; historical claims are not new verified results |

Current repository also contains `transweather`, `promptir`, `ramit`, and `liteweatherformer` source/config files. Their presence is **not** proof of faithful official reproduction, successful weight loading, benchmark performance, or successful training. They were not validated in this session; review before making research claims. Fog/snow/smoke attention configs are legacy Pix2Pix recipes, not prepared Restormer recipes.

The notebook `notebooks/restoration.ipynb` calls the same underlying scripts, with `prepare`, `sanity`, `train`, `evaluate`, and `export` stages. No separate notebook training implementation is needed. Its qualitative cell still calls `predict` without configured precision; review it before full-native GPU use. Re-running its baseline-save cell can overwrite `pretrained.pt`; preserve completed run evidence.

Papers provided by the user:

- `archive/papers/Zhang_WeatherStream_Light_Transport_Automation_of_Single_Image_Deweathering_CVPR_2023_paper.pdf`
- `archive/papers/09447-AAAI24.LiuX.pdf`
- `archive/papers/2609.02434v1.pdf`
- ScienceDirect link in `archive/papers/links.txt`: `https://www.sciencedirect.com/science/article/pii/S095219762601986X`
- AllRestorer: `https://arxiv.org/abs/2411.10708`

ScienceDirect text was also supplied in local attachment files, including `/home/bms/.codex/attachments/90df06cb-d416-4374-9ecc-e3003b80b300/Pasted text.txt` and `/home/bms/.codex/attachments/ba15f4b0-ef56-4982-a23c-1ce1b8636b8d/Pasted text.txt`. These are outside the repo and will not travel with a Git clone; check availability if needed. Read papers for context and ground any proposed changes in verified experiments.

An earlier user request mentioned a Chitraksh commit containing an `int_pipeline`-like folder. Its exact name and incorporation have not been verified in this handoff. Use Git history rather than asserting it was integrated.

User priorities: restoration first, additional weather baselines next, architecture/hyperparameter changes after a working pipeline, then detection and quantization/deployment; minimal files and no unnecessary engineering; concrete commands with clear terminal locations; publication-quality evidence, without acceptance promises.

## 15. Tests, help, and later-stage commands

Run after code changes in an activated environment; these do not start the full GPU training run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m ruff check .
git diff --check
bash -n scripts/restoration/train_rain_dgx.sbatch scripts/evaluate_on_dgx.sh

python scripts/restoration/train.py --help
python scripts/restoration/evaluate.py --help
python scripts/restoration/restore_images.py --help
python scripts/data/build_pair_manifest.py --help
python scripts/detection/train.py --help
python scripts/detection/evaluate.py --help
```

A prior 23-test restoration check passed before subsequent repository additions. The full current checkout has not been declared test-clean by this handoff; run checks and separate pre-existing failures from new changes.

Optional notebook kernel, after installing the notebook extra:

```bash
python -m ipykernel install --user --name aegis-perception --display-name 'Aegis Perception'
```

Choose this kernel in the IDE. It does not allocate a cluster GPU or replace SLURM. Lightning/Colab are alternatives, not the active proven environment; instance permissions, storage persistence, and credits must be checked separately.

Later detection commands require prepared labels and real dataset YAMLs; the current DAWN YAML is a template. In a separate prepared detector environment/allocation:

```bash
# Interface reference only; these custom data paths must be created first.
python scripts/detection/train.py --data data/processed/detection/dataset.yaml \
  --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device 0 --seed 42 --name detector_seed42
python scripts/detection/evaluate.py \
  --model artifacts/detection/training/detector_seed42/weights/best.pt \
  --view raw=data/processed/detection/raw.yaml \
  --view restored=data/processed/detection/restored.yaml --device 0
```

Install `.[detection]` on the login node only after checking that dependency resolution will preserve the working CUDA/NumPy combination, or use a separate detector environment. Detector benefits require same scenes/labels/weights across raw/restored views.

Later ONNX tools exist; install export dependencies in a reviewed environment before using them. This is a CPU numerical check/export at the checkpoint's configured crop size, **128 x 128 for this run**, not a native-resolution deployed engine:

```bash
python scripts/deployment/export_onnx.py \
  --checkpoint artifacts/restoration/rain_restormer_seed42/best.pt \
  --output artifacts/restoration/rain_restormer_seed42/restormer.onnx
python scripts/deployment/validate_onnx.py \
  --checkpoint artifacts/restoration/rain_restormer_seed42/best.pt \
  --onnx artifacts/restoration/rain_restormer_seed42/restormer.onnx
python scripts/deployment/benchmark_onnx.py \
  --model artifacts/restoration/rain_restormer_seed42/restormer.onnx \
  --shape 1 3 128 128 --provider CPUExecutionProvider
```

No successful export/quantization/hardware result is claimed. TensorRT FP16/INT8, representative calibration separate from test, quality checks, and end-to-end latency remain future work.

## 16. Minimal continuation brief for the next agent

Read this file and `docs/datasets.md`, `docs/experiments.md`, and the rain PDF. Preserve completed seed-42 evidence. Verify local/remote revisions and the working `.venv` before GPU work. Audit CSD/SnowCityscapes, prepare one matching Restormer weather baseline, and run native-resolution preflight before full optimization. Acquire a verified paired fog training source. Repair smoke background leakage with a documented protocol. Then train a balanced joint-weather baseline and compare each weather against its specialist. Condition any architectural change on measured weaknesses, and evaluate unseen data/independent scenes before publication claims. Keep code/config changes small and report exactly what was tested.
