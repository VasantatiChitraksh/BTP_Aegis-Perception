from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .checkpoints import save_checkpoint
from .config import validate_restoration_config
from .data.paired import PairedImageDataset
from .models import AttentionUNetGenerator, PatchDiscriminator, initialize_pix2pix_weights
from .reproducibility import environment_record, seed_everything, write_json


def _select_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def _validate(generator, loader, device: str) -> dict[str, float]:
    import torch
    from torch.nn import functional as functional

    generator.eval()
    total_l1 = 0.0
    total_mse = 0.0
    samples = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            predictions = generator(inputs)
            batch_size = inputs.shape[0]
            mean_l1 = functional.l1_loss(predictions, targets, reduction="mean").item()
            total_l1 += mean_l1 * batch_size
            pred_01 = predictions.add(1).div(2).clamp(0, 1)
            target_01 = targets.add(1).div(2).clamp(0, 1)
            sample_mse = functional.mse_loss(pred_01, target_01, reduction="none")
            total_mse += sample_mse.flatten(1).mean(1).sum().item()
            samples += batch_size
    mean_mse = total_mse / samples
    return {"l1": total_l1 / samples, "psnr_db": -10.0 * math.log10(max(mean_mse, 1e-12))}


def train_restoration(config: dict[str, Any]) -> list[dict[str, float]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    validate_restoration_config(config)
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    device = _select_device(str(config["run"].get("device", "auto")))
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run.json", environment_record(config))

    image_size = tuple(config["data"]["image_size"])
    manifest = config["data"]["manifest"]
    train_dataset = PairedImageDataset(
        manifest,
        "train",
        image_size=image_size,
        random_flip=bool(config["data"].get("random_flip", True)),
    )
    val_dataset = PairedImageDataset(manifest, "val", image_size=image_size, random_flip=False)
    loader_kwargs = {
        "batch_size": int(config["train"]["batch_size"]),
        "num_workers": int(config["data"].get("num_workers", 4)),
        "pin_memory": device.startswith("cuda"),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    generator = AttentionUNetGenerator(
        features=int(config["model"]["features"]),
        attention=bool(config["model"]["attention"]),
    ).to(device)
    discriminator = PatchDiscriminator(features=int(config["model"]["features"])).to(device)
    generator.apply(initialize_pix2pix_weights)
    discriminator.apply(initialize_pix2pix_weights)

    learning_rate = float(config["train"]["learning_rate"])
    betas = tuple(float(value) for value in config["train"]["betas"])
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)
    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = nn.L1Loss()
    gan_weight = float(config["train"].get("gan_weight", 1.0))
    l1_weight = float(config["train"]["l1_weight"])
    epochs = int(config["train"]["epochs"])
    checkpoint_every = int(config["train"].get("checkpoint_every", 10))

    history: list[dict[str, float]] = []
    best_psnr = -math.inf
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        running_g = 0.0
        running_d = 0.0
        samples = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            predictions = generator(inputs)

            if gan_weight > 0:
                discriminator.requires_grad_(True)
                optimizer_d.zero_grad(set_to_none=True)
                logits_real = discriminator(inputs, targets)
                logits_fake = discriminator(inputs, predictions.detach())
                loss_d = 0.5 * (
                    adversarial_loss(logits_real, torch.ones_like(logits_real))
                    + adversarial_loss(logits_fake, torch.zeros_like(logits_fake))
                )
                loss_d.backward()
                optimizer_d.step()
            else:
                loss_d = torch.zeros((), device=device)

            discriminator.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            loss_g = l1_weight * reconstruction_loss(predictions, targets)
            if gan_weight > 0:
                logits_fake_for_g = discriminator(inputs, predictions)
                loss_g = loss_g + gan_weight * adversarial_loss(
                    logits_fake_for_g, torch.ones_like(logits_fake_for_g)
                )
            loss_g.backward()
            optimizer_g.step()
            discriminator.requires_grad_(True)

            batch_size = inputs.shape[0]
            samples += batch_size
            running_g += loss_g.item() * batch_size
            running_d += loss_d.item() * batch_size
            progress.set_postfix(g=f"{loss_g.item():.3f}", d=f"{loss_d.item():.3f}")

        validation = _validate(generator, val_loader, device)
        row = {
            "epoch": float(epoch),
            "train_g": running_g / samples,
            "train_d": running_d / samples,
            "val_l1": validation["l1"],
            "val_psnr_db": validation["psnr_db"],
        }
        history.append(row)
        write_json(output_dir / "history.json", history)
        checkpoint = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "config": config,
            "validation": validation,
        }
        if epoch % checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", checkpoint)
        if validation["psnr_db"] > best_psnr:
            best_psnr = validation["psnr_db"]
            save_checkpoint(output_dir / "best.pt", checkpoint)
        print(
            f"epoch={epoch} val_l1={validation['l1']:.5f} "
            f"val_psnr={validation['psnr_db']:.2f}dB"
        )
    return history
