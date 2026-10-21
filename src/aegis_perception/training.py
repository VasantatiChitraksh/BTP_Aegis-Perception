from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoints import load_pretrained, save_checkpoint
from .config import validate_restoration_config
from .data.catalog import sha256
from .data.paired import PairedImageDataset
from .data.protocol import verify_protocol
from .data.sampling import RestorationBatchSampler, weather_scene_weights
from .distributed import distributed_session, gather_objects
from .evaluation import background_means
from .inference import predict
from .losses import reconstruction_objective, structural_similarity_per_image
from .models import PatchDiscriminator, build_generator, initialize_pix2pix_weights
from .reproducibility import environment_record, seed_everything, write_json


def _validate(
    generator, loader, device: str, *, precision: str = "fp32", compute_ssim: bool = False
) -> dict[str, float]:
    import torch
    from torch.nn import functional as functional

    generator.eval()
    rows = []
    total_l1 = 0.0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            predictions = predict(generator, inputs, precision=precision)
            batch_size = inputs.shape[0]
            conditions = batch.get("weather", ["unknown"] * batch_size)
            groups = batch.get("group_id", [str(len(rows) + i) for i in range(batch_size)])
            total_l1 += functional.l1_loss(predictions, targets).item() * batch_size
            target_01 = targets.add(1).div(2).clamp(0, 1)
            values = {}
            for normalized, prefix in ((predictions, ""), (inputs, "raw_")):
                image_01 = normalized.add(1).div(2).clamp(0, 1)
                mse = (image_01 - target_01).square().flatten(1).mean(1).clamp_min(1e-12)
                values[f"{prefix}psnr_db"] = (-10 * mse.log10()).tolist()
                if compute_ssim:
                    values[f"{prefix}ssim"] = structural_similarity_per_image(
                        target_01, image_01
                    ).tolist()
            for i, (weather, group) in enumerate(zip(conditions, groups, strict=True)):
                rows.append(
                    {
                        "weather": weather,
                        "group_id": group,
                        "stratum": batch.get("stratum", ["all"] * batch_size)[i],
                        **{key: value[i] for key, value in values.items()},
                    }
                )
    gathered = gather_objects((rows, total_l1))
    rows = [row for part, _ in gathered for row in part]
    if not rows:
        raise ValueError("Validation is empty")
    metrics = ["psnr_db", "raw_psnr_db"] + (["ssim", "raw_ssim"] if compute_ssim else [])
    result = {key: float(np.mean([r[key] for r in rows])) for key in metrics}
    result["l1"] = sum(loss for _, loss in gathered) / len(rows)
    result["psnr_gain_db"] = result["psnr_db"] - result["raw_psnr_db"]
    if compute_ssim:
        result["ssim_gain"] = result["ssim"] - result["raw_ssim"]
    groups = background_means(rows, metrics)
    conditions = sorted({row["weather"] for row in rows})
    for weather in conditions:
        weather_groups = [v for (w, _), v in groups.items() if w == weather]
        for metric in metrics:
            prefix = "raw_scene_" if metric.startswith("raw_") else "scene_"
            name = f"{weather}_{prefix}{metric.removeprefix('raw_')}"
            result[name] = float(np.mean([v[metric] for v in weather_groups]))
        result[f"{weather}_scene_psnr_gain_db"] = (
            result[f"{weather}_scene_psnr_db"] - result[f"{weather}_raw_scene_psnr_db"]
        )
        for metric in ("psnr_db", "ssim") if compute_ssim else ("psnr_db",):
            weather_rows = [r for r in rows if r["weather"] == weather]
            result[f"{weather}_{metric}_worse_fraction"] = sum(
                r[metric] < r[f"raw_{metric}"] for r in weather_rows
            ) / len(weather_rows)
    for metric in ("psnr_db", "ssim") if compute_ssim else ("psnr_db",):
        result[f"macro_scene_{metric}"] = float(
            np.mean([result[f"{w}_scene_{metric}"] for w in conditions])
        )
    return result


def passes_validation_guard(current, reference, tolerances):
    """Every weather must retain both metrics relative to initialization."""
    keys = [
        key
        for key in reference
        if any(key.endswith(f"_scene_{metric}") for metric in tolerances)
        and not key.startswith("macro_")
        and "_raw_scene_" not in key
    ]
    if not keys:
        raise ValueError("Validation guard has no per-weather metrics")
    return all(
        current[key] >= reference[key] - tolerances["ssim" if key.endswith("_ssim") else "psnr_db"]
        for key in keys
    )


def train_restoration(config, *, resume=None, stop_after_epochs=None):
    with distributed_session(str(config["run"].get("device", "auto"))) as context:
        return _train_restoration(
            config, resume=resume, stop_after_epochs=stop_after_epochs, distributed_context=context
        )


def _train_restoration(
    config: dict[str, Any],
    *,
    resume: str | Path | None = None,
    stop_after_epochs: int | None = None,
    distributed_context=("cpu", 0, 1),
) -> list[dict[str, float]]:
    """Train and resume at epoch boundaries; the notebook calls this same path."""
    import torch
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from tqdm import tqdm

    validate_restoration_config(config)
    verify_protocol(config["data"])
    if stop_after_epochs is not None and stop_after_epochs <= 0:
        raise ValueError("stop_after_epochs must be positive")
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    if config["run"].get("deterministic", False):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device, rank, world_size = distributed_context
    primary = rank == 0
    precision = config["train"].get("precision", "fp32")
    if precision != "fp32" and not device.startswith("cuda"):
        raise ValueError("Mixed precision requires CUDA; use fp32 for CPU checks")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; select fp16 or fp32")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    output_dir = Path(config["run"]["output_dir"])
    if resume is None and any(
        (output_dir / name).exists() for name in ("last.pt", "best.pt", "history.json")
    ):
        raise ValueError("Run already exists. Use --resume last.pt or choose a new output_dir")
    image_size = tuple(config["data"]["image_size"])
    manifest = config["data"]["manifest"]
    manifest_hash = sha256(Path(manifest))
    train_dataset = PairedImageDataset(
        manifest,
        "train",
        image_size=image_size,
        random_flip=bool(config["data"].get("random_flip", True)),
        spatial_mode=config["data"].get("train_mode", "resize"),
        random_crop=bool(config["data"].get("random_crop", True)),
        max_samples=config["data"].get("max_train_samples"),
    )
    val_dataset = PairedImageDataset(
        manifest,
        "val",
        image_size=image_size,
        random_flip=False,
        spatial_mode=config["data"].get("eval_mode", "resize"),
        random_crop=False,
        max_samples=config["data"].get("max_val_samples"),
    )
    loader_kwargs = {
        "num_workers": int(config["data"].get("num_workers", 4)),
        "pin_memory": device.startswith("cuda"),
    }
    probabilities = config["data"].get("weather_probabilities")
    batch_sampler = None
    global_batch = int(config["train"]["batch_size"])
    sampling = config["data"].get("sampling", "weighted")
    balanced = sampling in {"balanced_batches", "stratified_cycles"}
    if balanced or sampling == "weighted_seeded" or world_size > 1:
        batch_sampler = RestorationBatchSampler(
            train_dataset.records,
            global_batch,
            int(config["data"].get("samples_per_epoch", len(train_dataset))),
            probabilities=probabilities,
            balanced=balanced,
            stratified_cycles=sampling == "stratified_cycles",
            seeded_augmentations=sampling == "weighted_seeded",
            seed=seed,
            rank=rank,
            world_size=world_size,
        )
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        sampler = None
        if probabilities is not None:
            sampler = WeightedRandomSampler(
                weather_scene_weights(train_dataset.records, probabilities),
                int(config["data"].get("samples_per_epoch", len(train_dataset))),
                replacement=True,
            )
        train_loader = DataLoader(
            train_dataset,
            shuffle=sampler is None,
            sampler=sampler,
            batch_size=global_batch,
            **loader_kwargs,
        )
    # No DistributedSampler padding: evaluate every validation image exactly once.
    local_val = Subset(val_dataset, list(range(rank, len(val_dataset), world_size)))
    val_loader = DataLoader(local_val, shuffle=False, batch_size=1, **loader_kwargs)
    compute_ssim = bool(config["train"].get("validation_ssim", False))
    guard = config["train"].get("validation_guard")
    reference_validation = None

    model_config = config["model"]
    generator = build_generator(model_config).to(device)
    is_pix2pix = model_config.get("name", "attention_unet") == "attention_unet"
    gan_weight = float(config["train"].get("gan_weight", 1.0 if is_pix2pix else 0.0))
    if world_size > 1 and gan_weight > 0:
        raise ValueError("Distributed training currently supports reconstruction models only")
    discriminator = None
    if gan_weight > 0:
        discriminator = PatchDiscriminator(features=int(model_config["features"])).to(device)
    if is_pix2pix:
        generator.apply(initialize_pix2pix_weights)
        if discriminator is not None:
            discriminator.apply(initialize_pix2pix_weights)
    pretrained = model_config.get("pretrained")
    pretrained_hash = None
    if pretrained and resume is None:
        if not Path(pretrained).is_file():
            raise FileNotFoundError(
                f"Pretrained weights missing: {pretrained}. Run the notebook download cell"
            )
        pretrained_hash = sha256(Path(pretrained))
        expected_hash = config["train"].get("expected_pretrained_sha256")
        if expected_hash and pretrained_hash != expected_hash:
            raise ValueError("Initialization checkpoint hash differs from the frozen experiment")
        load_pretrained(generator, pretrained)

    optimizer_class = (
        torch.optim.AdamW
        if config["train"].get("optimizer", "adam") == "adamw"
        else torch.optim.Adam
    )
    optimizer_kwargs = {
        "lr": float(config["train"]["learning_rate"]),
        "betas": tuple(float(value) for value in config["train"]["betas"]),
        "weight_decay": float(config["train"].get("weight_decay", 0.0)),
    }
    optimizer_g = optimizer_class(generator.parameters(), **optimizer_kwargs)
    optimizer_d = (
        optimizer_class(discriminator.parameters(), **optimizer_kwargs)
        if discriminator is not None
        else None
    )
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_weight = float(config["train"]["l1_weight"])
    epochs = int(config["train"]["epochs"])
    checkpoint_every = int(config["train"].get("checkpoint_every", 10))
    grad_clip = float(config["train"].get("grad_clip", 0.0))
    schedulers = []
    if config["train"].get("scheduler", "none") == "cosine":
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=float(config["train"].get("min_learning_rate", 1e-6)),
            )
            for optimizer in (optimizer_g, optimizer_d)
            if optimizer is not None
        ]

    history: list[dict[str, float]] = []
    best_psnr, start_epoch = -math.inf, 1
    selection_metric = config["train"].get("selection_metric", "psnr_db")
    resume_checkpoint = None
    if resume:
        resume_checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        for section in ("model", "data", "train"):
            if resume_checkpoint["config"][section] != config[section]:
                raise ValueError(
                    f"Resume {section} config differs. Start a new run for changed experiments"
                )
        if resume_checkpoint["config"]["run"].get("deterministic", False) != config["run"].get(
            "deterministic", False
        ):
            raise ValueError("Resume determinism setting differs")
        if resume_checkpoint["config"]["run"]["seed"] != seed:
            raise ValueError("Resume seed differs")
        if resume_checkpoint.get("manifest_sha256") != manifest_hash:
            raise ValueError("Resume manifest changed; do not change dataset splits mid-run")
        generator.load_state_dict(resume_checkpoint["generator"])
        optimizer_g.load_state_dict(resume_checkpoint["optimizer_g"])
        if discriminator is not None:
            discriminator.load_state_dict(resume_checkpoint["discriminator"])
            optimizer_d.load_state_dict(resume_checkpoint["optimizer_d"])
        for scheduler, state in zip(schedulers, resume_checkpoint["schedulers"], strict=True):
            scheduler.load_state_dict(state)
        scaler.load_state_dict(resume_checkpoint["scaler"])
        history = resume_checkpoint["history"]
        best_psnr = resume_checkpoint.get("best_selection_score", resume_checkpoint["best_psnr_db"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        pretrained_hash = resume_checkpoint.get("pretrained_sha256")
        reference_validation = resume_checkpoint.get("reference_validation")
        if resume_checkpoint.get("world_size", 1) != world_size:
            raise ValueError("Exact resume requires the same GPU/process count")
        rng = resume_checkpoint.get("rng_states_by_rank", [resume_checkpoint["rng_state"]])[rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if device.startswith("cuda") and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
    if start_epoch > epochs:
        print("Training already complete")
        return history

    # DDP wraps only training forwards; validation uses the underlying model.
    training_generator = generator
    if world_size > 1:
        training_generator = DistributedDataParallel(
            generator,
            device_ids=[int(device.split(":")[1])] if device.startswith("cuda") else None,
        )
    # Different crop/flip streams on different ranks, identical global batch schedule.
    if resume is None and world_size > 1:
        seed_everything(seed + rank)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = environment_record(config)
    metadata.update(
        {
            "manifest_sha256": manifest_hash,
            "pretrained_sha256": pretrained_hash,
            "parameter_count": sum(p.numel() for p in generator.parameters()),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "precision": precision,
            "validation_precision": precision,
            "resume_from": str(resume) if resume else None,
            "world_size": world_size,
            "global_batch_size": global_batch,
            "local_batch_size": global_batch // world_size,
            "psnr_protocol": "mean per-image RGB [0,1], MSE floor 1e-12",
        }
    )
    if primary:
        write_json(output_dir / ("resume.json" if resume else "run.json"), metadata)
    tracking_enabled = primary and bool(config["run"].get("wandb", False))
    if tracking_enabled:
        import wandb

        wandb.init(project="aegis-perception", config=config, name=output_dir.name)

    if resume is None:
        print(
            f"Checking all {len(val_dataset)} validation images before training ({precision})",
            flush=True,
        )
        # DataLoader iteration consumes a CPU RNG seed; preserve the training shuffle.
        with torch.random.fork_rng(devices=[]):
            baseline = _validate(
                generator, val_loader, device, precision=precision, compute_ssim=compute_ssim
            )
        reference_validation = baseline
        if guard:
            best_psnr = baseline[selection_metric]
        if primary:
            write_json(
                output_dir / "baseline_validation.json", {"precision": precision, **baseline}
            )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"Validation check passed; pretrained PSNR={baseline['psnr_db']:.2f}dB", flush=True)

    if guard and reference_validation is None:
        raise ValueError("Guarded resume requires reference validation in the checkpoint")

    end_epoch = (
        epochs if stop_after_epochs is None else min(epochs, start_epoch + stop_after_epochs - 1)
    )
    for epoch in range(start_epoch, end_epoch + 1):
        training_generator.train()
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        if discriminator is not None:
            discriminator.train()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        learning_rate = optimizer_g.param_groups[0]["lr"]
        running_g = running_d = 0.0
        samples = 0
        for batch in tqdm(
            train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not primary
        ):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=precision != "fp32"):
                predictions = training_generator(inputs)
            if discriminator is not None:
                discriminator.requires_grad_(True)
                optimizer_d.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=precision != "fp32"):
                    logits_real = discriminator(inputs, targets)
                    logits_fake = discriminator(inputs, predictions.detach())
                    loss_d = 0.5 * (
                        adversarial_loss(logits_real, torch.ones_like(logits_real))
                        + adversarial_loss(logits_fake, torch.zeros_like(logits_fake))
                    )
                scaler.scale(loss_d).backward()
                scaler.step(optimizer_d)
                discriminator.requires_grad_(False)
            else:
                loss_d = torch.zeros((), device=device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=precision != "fp32"):
                loss_g = reconstruction_objective(
                    predictions,
                    targets,
                    l1_weight=l1_weight,
                    pix2pix=is_pix2pix,
                    mse_weight=float(config["train"].get("mse_weight", 0)),
                    ssim_weight=float(config["train"].get("ssim_weight", 0)),
                )
                if discriminator is not None:
                    logits = discriminator(inputs, predictions)
                    loss_g = loss_g + gan_weight * adversarial_loss(logits, torch.ones_like(logits))
            if not torch.isfinite(loss_g) or not torch.isfinite(loss_d):
                raise FloatingPointError("Non-finite training loss; check data and precision")
            scaler.scale(loss_g).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(
                    generator.parameters(), grad_clip, error_if_nonfinite=precision != "fp16"
                )
            scaler.step(optimizer_g)
            scaler.update()
            batch_size = inputs.shape[0]
            samples += batch_size
            running_g += loss_g.item() * batch_size
            running_d += loss_d.item() * batch_size
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started
        # Validation does not need the final training batch's parameter gradients.
        optimizer_g.zero_grad(set_to_none=True)
        if optimizer_d is not None:
            optimizer_d.zero_grad(set_to_none=True)
        print(f"epoch={epoch} validating {len(val_dataset)} images ({precision})", flush=True)
        validation = _validate(
            generator, val_loader, device, precision=precision, compute_ssim=compute_ssim
        )
        totals = gather_objects((running_g, running_d, samples))
        running_g, running_d, samples = map(sum, zip(*totals, strict=True))
        row = {
            "epoch": float(epoch),
            "train_g": running_g / samples,
            "train_d": running_d / samples,
            "val_l1": validation["l1"],
            "val_psnr_db": validation["psnr_db"],
            "val_raw_psnr_db": validation["raw_psnr_db"],
            "val_psnr_gain_db": validation["psnr_gain_db"],
            "learning_rate": float(learning_rate),
            "train_seconds": train_seconds,
            "epoch_seconds": time.perf_counter() - started,
            "train_steps_per_second": len(train_loader) / train_seconds,
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20
            if device.startswith("cuda")
            else 0.0,
        }
        row.update({f"val_{key}": value for key, value in validation.items()})
        history.append(row)
        for scheduler in schedulers:
            scheduler.step()
        selection_score = validation[selection_metric]
        eligible = not guard or passes_validation_guard(validation, reference_validation, guard)
        improved = eligible and selection_score > best_psnr + float(
            config["train"].get("selection_min_delta", 0.0)
        )
        if improved:
            best_psnr = selection_score
        row["validation_guard_passed"] = bool(eligible)
        row["selected_checkpoint"] = bool(improved)
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
        }
        rng_states = gather_objects(rng_state)
        checkpoint = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "config": config,
            "validation": validation,
            "history": history,
            "best_psnr_db": best_psnr,
            "best_selection_score": best_psnr,
            "selection_metric": selection_metric,
            "manifest_sha256": manifest_hash,
            "pretrained_sha256": pretrained_hash,
            "schedulers": [scheduler.state_dict() for scheduler in schedulers],
            "scaler": scaler.state_dict(),
            "rng_state": rng_state,
            "rng_states_by_rank": rng_states,
            "world_size": world_size,
            "reference_validation": reference_validation,
        }

        if discriminator is not None:
            checkpoint.update(
                discriminator=discriminator.state_dict(), optimizer_d=optimizer_d.state_dict()
            )
        if primary:
            save_checkpoint(output_dir / "last.pt", checkpoint)
            if improved:
                save_checkpoint(output_dir / "best.pt", checkpoint)
            if config["train"].get("save_periodic_checkpoints", True) and (
                epoch % checkpoint_every == 0 or epoch == epochs
            ):
                save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", checkpoint)
            write_json(output_dir / "history.json", history)
            write_json(
                output_dir / "selection_status.json",
                {
                    "candidate_available": (output_dir / "best.pt").exists(),
                    "reference": (
                        "initialization evaluated on this run's validation manifest"
                        if guard else "highest validation selection score during this run"
                    ),
                    "guard": guard,
                    "best_selection_score": best_psnr,
                    "latest_epoch": epoch,
                    "latest_guard_passed": bool(eligible),
                },
            )
        if tracking_enabled:
            wandb.log(row)
        ssim_log = (
            f" balanced_ssim={validation['macro_scene_ssim']:.5f}" if compute_ssim else ""
        )
        print(
            f"epoch={epoch} val_psnr={validation['psnr_db']:.2f}dB "
            f"gain={validation['psnr_gain_db']:+.2f}dB "
            f"selected_by={selection_metric} score={selection_score:.2f}dB"
            f"{ssim_log} checkpoint_saved={bool(improved)}",
            flush=True,
        )
    if tracking_enabled:
        wandb.finish()
    return history
