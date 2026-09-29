from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoints import load_pretrained, save_checkpoint, select_device
from .config import validate_restoration_config
from .data.catalog import sha256
from .data.paired import PairedImageDataset
from .inference import predict
from .models import PatchDiscriminator, build_generator, initialize_pix2pix_weights
from .reproducibility import environment_record, seed_everything, write_json


def _validate(generator, loader, device: str) -> dict[str, float]:
    import torch
    from torch.nn import functional as functional

    generator.eval()
    total_l1 = total_psnr = total_raw_psnr = 0.0
    samples = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            predictions = predict(generator, inputs)
            batch_size = inputs.shape[0]
            total_l1 += functional.l1_loss(predictions, targets).item() * batch_size
            target_01 = targets.add(1).div(2).clamp(0, 1)
            for normalized, restored in ((predictions, True), (inputs, False)):
                image_01 = normalized.add(1).div(2).clamp(0, 1)
                mse = (image_01 - target_01).square().flatten(1).mean(1).clamp_min(1e-12)
                value = (-10 * mse.log10()).sum().item()
                if restored:
                    total_psnr += value
                else:
                    total_raw_psnr += value
            samples += batch_size
    return {
        "l1": total_l1 / samples,
        "psnr_db": total_psnr / samples,
        "raw_psnr_db": total_raw_psnr / samples,
        "psnr_gain_db": (total_psnr - total_raw_psnr) / samples,
    }


def train_restoration(
    config: dict[str, Any],
    *,
    resume: str | Path | None = None,
    stop_after_epochs: int | None = None,
) -> list[dict[str, float]]:
    """Train and resume at epoch boundaries; the notebook calls this same path."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    validate_restoration_config(config)
    if stop_after_epochs is not None and stop_after_epochs <= 0:
        raise ValueError("stop_after_epochs must be positive")
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    device = select_device(str(config["run"].get("device", "auto")))
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
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=int(config["train"]["batch_size"]), **loader_kwargs
    )
    # Original-resolution pairs have different sizes and must be batched individually.
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, **loader_kwargs)

    model_config = config["model"]
    generator = build_generator(model_config).to(device)
    is_pix2pix = model_config.get("name", "attention_unet") == "attention_unet"
    gan_weight = float(config["train"].get("gan_weight", 1.0 if is_pix2pix else 0.0))
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
        load_pretrained(generator, pretrained)
        pretrained_hash = sha256(Path(pretrained))

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
    adversarial_loss, reconstruction_loss = nn.BCEWithLogitsLoss(), nn.L1Loss()
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
    resume_checkpoint = None
    if resume:
        resume_checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        for section in ("model", "data", "train"):
            if resume_checkpoint["config"][section] != config[section]:
                raise ValueError(
                    f"Resume {section} config differs. Start a new run for changed experiments"
                )
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
        best_psnr = resume_checkpoint["best_psnr_db"]
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        pretrained_hash = resume_checkpoint.get("pretrained_sha256")
        rng = resume_checkpoint["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if device.startswith("cuda") and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
    if start_epoch > epochs:
        print("Training already complete")
        return history

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
            "resume_from": str(resume) if resume else None,
            "psnr_protocol": "mean per-image RGB [0,1], MSE floor 1e-12",
        }
    )
    write_json(output_dir / ("resume.json" if resume else "run.json"), metadata)
    tracking_enabled = bool(config["run"].get("wandb", False))
    if tracking_enabled:
        import wandb

        wandb.init(project="aegis-perception", config=config, name=output_dir.name)

    end_epoch = (
        epochs if stop_after_epochs is None else min(epochs, start_epoch + stop_after_epochs - 1)
    )
    for epoch in range(start_epoch, end_epoch + 1):
        generator.train()
        if discriminator is not None:
            discriminator.train()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        learning_rate = optimizer_g.param_groups[0]["lr"]
        running_g = running_d = 0.0
        samples = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=precision != "fp32"):
                predictions = generator(inputs)
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
                # Restormer's published L1 loss is on RGB [0,1]; legacy Pix2Pix is [-1,1].
                loss_g = l1_weight * reconstruction_loss(predictions.float(), targets)
                if not is_pix2pix:
                    loss_g = loss_g * 0.5
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
        validation = _validate(generator, val_loader, device)
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
        history.append(row)
        for scheduler in schedulers:
            scheduler.step()
        improved = validation["psnr_db"] > best_psnr
        best_psnr = max(best_psnr, validation["psnr_db"])
        checkpoint = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "config": config,
            "validation": validation,
            "history": history,
            "best_psnr_db": best_psnr,
            "manifest_sha256": manifest_hash,
            "pretrained_sha256": pretrained_hash,
            "schedulers": [scheduler.state_dict() for scheduler in schedulers],
            "scaler": scaler.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
            },
        }
        if discriminator is not None:
            checkpoint.update(
                discriminator=discriminator.state_dict(), optimizer_d=optimizer_d.state_dict()
            )
        save_checkpoint(output_dir / "last.pt", checkpoint)
        if improved:
            save_checkpoint(output_dir / "best.pt", checkpoint)
        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", checkpoint)
        write_json(output_dir / "history.json", history)
        if tracking_enabled:
            wandb.log(row)
        print(
            f"epoch={epoch} val_psnr={validation['psnr_db']:.2f}dB "
            f"gain={validation['psnr_gain_db']:+.2f}dB"
        )
    if tracking_enabled:
        wandb.finish()
    return history
