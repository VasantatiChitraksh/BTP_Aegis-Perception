"""Differentiable RGB SSIM matching this project's skimage evaluation settings."""

from __future__ import annotations


def structural_similarity_per_image(reference, prediction):
    """Uniform 7x7 window, sample covariance, RGB [0,1], K1=.01/K2=.03.

    Computes only valid windows, matching skimage's cropped mean. Keep this in
    FP32 even when model inference/training uses BF16. No gradient clamp is
    applied to SSIM itself; output pixel clamping belongs to the caller.
    """
    import torch
    from torch.nn import functional as F

    if reference.shape != prediction.shape or reference.ndim != 4:
        raise ValueError("SSIM expects matching NCHW tensors")
    if min(reference.shape[-2:]) < 7:
        raise ValueError("SSIM needs images at least 7x7")
    with torch.autocast(reference.device.type, enabled=False):
        x, y = reference.float(), prediction.float()
        ux, uy = F.avg_pool2d(x, 7, 1), F.avg_pool2d(y, 7, 1)
        vx = (F.avg_pool2d(x * x, 7, 1) - ux.square()) * (49 / 48)
        vy = (F.avg_pool2d(y * y, 7, 1) - uy.square()) * (49 / 48)
        cov = (F.avg_pool2d(x * y, 7, 1) - ux * uy) * (49 / 48)
        numerator = (2 * ux * uy + 0.01**2) * (2 * cov + 0.03**2)
        denominator = (ux.square() + uy.square() + 0.01**2) * (vx + vy + 0.03**2)
        return (numerator / denominator).flatten(1).mean(1)


def reconstruction_objective(
    predictions, targets, *, l1_weight=1.0, mse_weight=0.0, ssim_weight=0.0, pix2pix=False
):
    from torch.nn import functional as F

    # Preserve the historical L1 scale exactly for existing configurations.
    loss = l1_weight * F.l1_loss(predictions.float(), targets.float())
    if not pix2pix:
        loss = loss * 0.5
    x, y = predictions.float().add(1).div(2), targets.float().add(1).div(2)
    if mse_weight:
        loss = loss + mse_weight * F.mse_loss(x, y)
    if ssim_weight:
        loss = (
            loss
            + ssim_weight
            * (1 - structural_similarity_per_image(y.clamp(0, 1), x.clamp(0, 1))).mean()
        )
    return loss
