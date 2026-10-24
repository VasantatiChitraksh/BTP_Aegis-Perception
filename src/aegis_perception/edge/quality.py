"""RGB PSNR/SSIM for small deployment validation panels; no training dependency."""

import cv2
import numpy as np


def quality(reference, actual):
    """Float RGB [0,1], skimage-compatible 7x7 sample-covariance SSIM."""
    x, y = np.asarray(reference, np.float64), np.asarray(actual, np.float64)
    if x.shape != y.shape or x.ndim != 3 or x.shape[2] != 3 or min(x.shape[:2]) < 7:
        raise ValueError("Expected matching RGB images at least 7x7")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite quality inputs")
    psnr = -10 * np.log10(max(float(np.mean((x - y) ** 2)), 1e-12))

    def mean(a):
        return cv2.blur(a, (7, 7), borderType=cv2.BORDER_REFLECT)

    ux, uy = mean(x), mean(y)
    vx, vy = (mean(x * x) - ux * ux) * 49 / 48, (mean(y * y) - uy * uy) * 49 / 48
    cov = (mean(x * y) - ux * uy) * 49 / 48
    score = ((2 * ux * uy + 0.01**2) * (2 * cov + 0.03**2)) / (
        (ux * ux + uy * uy + 0.01**2) * (vx + vy + 0.03**2)
    )
    return {"psnr_db": float(psnr), "ssim": float(score[3:-3, 3:-3].mean())}
