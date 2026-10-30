"""Essential CPU checks for Nano restoration geometry and acceptance gates."""

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aegis_perception.edge.common import preprocess as reference_preprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/deployment"))
spec = importlib.util.spec_from_file_location(
    "nano_restormer_test", ROOT / "scripts/deployment/nano_restormer.py"
)
rest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rest)


def test_python36_and_frozen_input_geometry():
    ast.parse(Path(rest.__file__).read_text(), feature_version=(3, 6))
    image = np.random.default_rng(2).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    actual, geometry = rest.preprocess(image[:, :, ::-1].copy())
    expected, _ = reference_preprocess(Image.fromarray(image))
    np.testing.assert_array_equal(actual, expected)
    assert geometry == (640, 480, 341, 256)
    assert rest.postprocess(actual, geometry).shape == image.shape


@pytest.mark.parametrize("shift,passed", [(0.0, True), (0.1, False)])
def test_restoration_gate(tmp_path, shift, passed):
    p = tmp_path / "fixtures.npz"
    np.savez(
        p,
        input_fog=np.zeros((1, 3, 256, 448), np.float32),
        expected_fog=np.zeros((1, 3, 256, 448), np.float32),
        target_fog=np.full((1, 3, 256, 448), 0.5, np.float32),
        valid_hw_fog=np.array([256, 341]),
    )

    class Engine:
        def run(self, value):
            return value + shift

    assert rest.validate_restorer(Engine(), p)["passed"] is passed
