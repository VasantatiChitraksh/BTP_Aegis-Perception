from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate_yolo(
    *,
    model_path: str | Path,
    data: str | Path,
    image_size: int = 640,
    batch_size: int = 1,
    device: str = "0",
    split: str = "test",
    project: str | Path = "artifacts/detection/evaluation",
    name: str = "evaluation",
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the detection extra: pip install -e '.[detection]'") from exc

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data),
        imgsz=image_size,
        batch=batch_size,
        device=device,
        split=split,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    results = {key: float(value) for key, value in metrics.results_dict.items()}
    results["speed_ms_per_image"] = {
        key: float(value) for key, value in getattr(metrics, "speed", {}).items()
    }
    names = getattr(metrics, "names", {})
    if names:
        results["class_names"] = {str(key): value for key, value in names.items()}
    return results
