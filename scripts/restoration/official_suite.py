#!/usr/bin/env python3
"""Sequential official TransWeather, PromptIR, RAMiT fine-tuning on weather-v3."""

import gc
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from aegis_perception.models.official import source_info
from aegis_perception.official_runtime import activate, build_generator, load_pretrained

activate()
spec = importlib.util.spec_from_file_location(
    "_official_base", ROOT / "scripts/restoration/baseline_suite.py"
)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
suite.__doc__ = __doc__
suite.NAMES = ("transweather", "promptir", "ramit")
suite.CONFIGS = tuple(Path(f"configs/restoration/official_v1/{n}.yaml") for n in suite.NAMES)
suite.OUTPUT = Path("artifacts/restoration/official_v1")
suite.BOUNDARIES = (10, 20, 30, 40, 50, 60)
suite.STAGE_SCRIPT = "scripts/restoration/train_official_stage.sbatch"
suite.LOG_PREFIX = "aegis-official"
suite.TRAIN_COMMAND = (sys.executable, "scripts/restoration/official_entry.py", "train")
suite.EVALUATE_COMMAND = (sys.executable, "scripts/restoration/official_entry.py", "evaluate")
_base_configs, _base_stage, _base_package = suite.checked_configs, suite.run_stage, suite.package

# Present in the official checkpoints but never called by upstream forward().
UNUSED = {
    "transweather": {
        "network.Tenc.mini_patch_embed4.proj.weight",
        "network.Tenc.mini_patch_embed4.proj.bias",
        "network.Tenc.mini_patch_embed4.norm.weight",
        "network.Tenc.mini_patch_embed4.norm.bias",
        "network.convtail.conv_output.conv2d.weight",
        "network.convtail.conv_output.conv2d.bias",
    },
    "promptir": {
        "network.chnl_reduce1.weight",
        "network.chnl_reduce2.weight",
        "network.chnl_reduce3.weight",
        "network.reduce_noise_channel_1.weight",
        "network.reduce_noise_channel_2.weight",
        "network.reduce_noise_channel_3.weight",
    },
    "ramit": set(),
}


def checked_configs():
    configs = _base_configs()
    reference = suite.load_config(ROOT / "configs/restoration/baselines_v1/restormer.yaml")
    for name, config in zip(suite.NAMES, configs, strict=True):
        weights = source_info(name)["weights"]
        if config["model"] != {
            "name": name,
            "implementation": "official_v1",
            "pretrained": weights["path"],
        }:
            raise ValueError("Official suite requires the exact checkpoint-compatible topology")
        if config["train"]["expected_pretrained_sha256"] != weights["sha256"]:
            raise ValueError("Initialization differs from official source lock")
        if config["data"] != reference["data"] or config["run"]["seed"] != reference["run"]["seed"]:
            raise ValueError("Frozen data, sampling and seed must match the completed baselines")
        for key in (
            "epochs",
            "batch_size",
            "learning_rate",
            "optimizer",
            "scheduler",
            "min_learning_rate",
            "betas",
            "weight_decay",
            "precision",
            "grad_clip",
            "l1_weight",
            "ssim_weight",
            "gan_weight",
            "selection_metric",
            "validation_ssim",
        ):
            if config["train"][key] != reference["train"][key]:
                raise ValueError(f"Declared fine-tuning budget/recipe mismatch: {key}")
    return configs


def ensure_initializations(configs):
    # Packaged deployments contain all weights; submission never silently downloads substitutes.
    subprocess.run(
        [sys.executable, "scripts/restoration/fetch_official_models.py", "--offline"], check=True
    )


def preflight(configs):
    import torch
    from PIL import Image

    from aegis_perception.inference import predict

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Allocate exactly one BF16-capable GPU")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stamp = suite.OUTPUT / "preflight.json"
    if stamp.exists():
        return
    suite.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (suite.OUTPUT / "dataset_audit.txt").open("w") as log:
        subprocess.run(
            [sys.executable, "scripts/data/audit_manifest.py", configs[0]["data"]["manifest"]],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    shapes = set()
    for record in suite.load_manifest(configs[0]["data"]["manifest"]):
        if record.split != "train":
            with Image.open(record.input_path) as image:
                shapes.add((image.height, image.width))
    checks = []
    for config in configs:
        name = config["model"]["name"]
        print(
            f"Preflight: strict pretrained load, training batch and native inference: {name}",
            flush=True,
        )
        model = build_generator(config["model"]).cuda().train()
        load_pretrained(model, config["model"]["pretrained"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])
        torch.cuda.reset_peak_memory_stats()
        inputs = torch.zeros(
            config["train"]["batch_size"], 3, *config["data"]["image_size"], device="cuda"
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(inputs)
        loss = (output.float() - 0.1).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training preflight: {name}")
        loss.backward()
        missing = {n for n, p in model.named_parameters() if p.grad is None}
        if missing != UNUSED[name]:
            raise RuntimeError(f"Unexpected inactive parameters: {name}: {missing ^ UNUSED[name]}")
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError(f"Nonfinite gradient: {name}")
        optimizer.step()
        train_memory = torch.cuda.max_memory_allocated() / 2**20
        del optimizer, inputs, output, loss
        model.zero_grad(set_to_none=True)
        model.eval()
        multiple = model.spatial_multiple
        biggest = sorted(
            shapes,
            key=lambda s: ((s[0] + multiple - 1) // multiple) * ((s[1] + multiple - 1) // multiple),
            reverse=True,
        )[:3]
        native = []
        for h, w in biggest:
            print(f"Preflight native shape: {name} {h}x{w}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                inputs = torch.zeros(1, 3, h, w, device="cuda")
                output = predict(model, inputs, precision="bf16")
                if output.shape != inputs.shape or not torch.isfinite(output).all():
                    raise RuntimeError(f"Native inference preflight failed: {name}")
            native.append(
                dict(height=h, width=w, peak_mib=torch.cuda.max_memory_allocated() / 2**20)
            )
            del inputs, output
        checks.append(
            dict(
                model=name,
                parameters=sum(p.numel() for p in model.parameters()),
                training_peak_mib=train_memory,
                native=native,
                unused=sorted(missing),
            )
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    suite.write_json(
        stamp,
        {
            "checks": checks,
            "gpu": torch.cuda.get_device_name(0),
            "vram_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
            "note": "Disposable steps; no test targets or test scores used.",
        },
    )


def report():
    subprocess.run([sys.executable, "scripts/restoration/report_six_models.py"], check=True)


def run_stage(config_path, through_epoch):
    _base_stage(config_path, through_epoch)
    if through_epoch != suite.BOUNDARIES[-1]:
        return
    import torch

    from aegis_perception.checkpoints import save_checkpoint

    config = suite.load_config(config_path)
    directory = Path(config["run"]["output_dir"])
    best = directory / "best.pt"
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    export = {key: checkpoint[key] for key in ("generator", "config", "epoch", "manifest_sha256")}
    export["source_checkpoint_sha256"] = suite.sha256(best)
    export["official_provenance"] = source_info(config["model"]["name"])
    save_checkpoint(directory / "inference.pt", export)
    print(f"Best checkpoint and compact inference export: {directory}", flush=True)


def package(output):
    ensure_initializations(checked_configs())
    names = [
        "pyproject.toml",
        "LICENSE",
        "README.md",
        "docs/official_model_training.txt",
        "scripts/restoration/baseline_suite.py",
        "scripts/restoration/official_suite.py",
        "scripts/restoration/official_entry.py",
        "scripts/restoration/fetch_official_models.py",
        "scripts/restoration/report_six_models.py",
        "scripts/restoration/train.py",
        "scripts/restoration/evaluate.py",
        "scripts/restoration/train_official_stage.sbatch",
        "scripts/restoration/submit_official.sh",
        "scripts/restoration/setup_official.sh",
        "scripts/data/audit_manifest.py",
        "configs/restoration/official_v1/sources.json",
        "configs/restoration/official_v1/references.json",
        "configs/restoration/official_v1/requirements.txt",
        "configs/restoration/liteweatherformer_v2.yaml",
        "configs/restoration/baselines_v1/attention_pix2pix.yaml",
        "configs/restoration/baselines_v1/restormer.yaml",
        "tests/conftest.py",
        "tests/test_official_models.py",
        "artifacts/restoration/official_v1/cpu_checks.json",
        "data/manifests/weather_v3/active.csv",
        "data/manifests/weather_v3/reserve.csv",
        "data/manifests/weather_v3/protocol.json",
    ]
    for name in suite.NAMES:
        info = source_info(name)
        names += [s["path"] for s in info["sources"]] + [info["weights"]["path"]]
    for directory in (
        "baselines_v1/attention_pix2pix_seed42",
        "baselines_v1/restormer_seed42",
        "liteweather_v1/liteweatherformer_v2_seed42",
    ):
        names += [
            f"artifacts/restoration/{directory}/{f}" for f in ("metrics_test.json", "history.json")
        ]
    names += [
        str(p.relative_to(ROOT)) for p in sorted((ROOT / "artifacts/official_wheels").glob("*.whl"))
    ]
    _base_package(output, file_names=names)


suite.checked_configs = checked_configs
suite.ensure_initializations = ensure_initializations
suite.preflight = preflight
suite.report = report
suite.run_stage = run_stage
suite.package = package

if __name__ == "__main__":
    suite.main()
