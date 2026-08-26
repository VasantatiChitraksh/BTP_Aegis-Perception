from __future__ import annotations

import math

import numpy as np


def psnr(reference: np.ndarray, prediction: np.ndarray, data_range: float = 1.0) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    mse = np.mean((reference - prediction) ** 2)
    if mse == 0:
        return math.inf
    return 10.0 * math.log10((data_range**2) / mse)


def ssim(reference: np.ndarray, prediction: np.ndarray, data_range: float = 1.0) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError("Install the metrics extra: pip install -e '.[metrics]'") from exc
    return float(
        structural_similarity(reference, prediction, channel_axis=-1, data_range=data_range)
    )
