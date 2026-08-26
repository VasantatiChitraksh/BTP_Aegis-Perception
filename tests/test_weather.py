from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from aegis_perception.weather import degrade


@pytest.mark.parametrize("condition", ["fog", "rain", "snow", "smoke"])
def test_weather_is_seeded_and_geometry_preserving(condition: str) -> None:
    array = np.zeros((64, 96, 3), dtype=np.uint8)
    array[:, :48] = (40, 80, 120)
    source = Image.fromarray(array)
    first = degrade(source, condition, severity=0.5, seed=123)
    second = degrade(source, condition, severity=0.5, seed=123)
    assert first.size == source.size
    assert first.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(source))


def test_rejects_invalid_severity() -> None:
    with pytest.raises(ValueError, match="severity"):
        degrade(Image.new("RGB", (32, 32)), "rain", severity=1.5, seed=42)
