"""Deployment contracts: geometry, COCO label IDs, precision metrics and NMS parity."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aegis_perception.edge.common import postprocess, preprocess, select_panel
from aegis_perception.edge.detector import decode, letterbox, nms
from aegis_perception.edge.quality import quality


@pytest.mark.parametrize("size", [(112, 64), (64, 112), (1, 17)])
def test_restoration_preserves_rgb_and_original_geometry(size):
    image = Image.new("RGB", size, (25, 80, 200))
    tensor, geometry = preprocess(image)
    assert tensor.shape == (1, 3, 256, 448)
    assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
    result = postprocess(tensor, geometry)
    assert result.size == size
    np.testing.assert_array_equal(np.asarray(result), np.asarray(image))


def test_nonfinite_restoration_rejected():
    with pytest.raises(ValueError):
        postprocess(np.full((1, 3, 8, 8), np.nan), {"resized_wh": [8, 8], "original_wh": [8, 8]})


def test_ssim_matches_reference():
    from skimage.metrics import structural_similarity

    rng = np.random.default_rng(42)
    reference = rng.random((27, 39, 3))
    output = np.clip(reference + rng.normal(0, 0.05, reference.shape), 0, 1)
    result = quality(reference, output)
    expected = structural_similarity(reference, output, channel_axis=-1, data_range=1)
    assert result["ssim"] == pytest.approx(expected, abs=1e-12)
    assert quality(reference, reference)["ssim"] == pytest.approx(1)


@pytest.mark.parametrize("shape", [(91, 231), (401, 129), (640, 640)])
def test_letterbox_matches_ultralytics_pixels(shape):
    from ultralytics.data.augment import LetterBox

    image = np.random.default_rng(2).integers(0, 256, (*shape, 3), dtype=np.uint8)
    ours, _ = letterbox(image)
    reference = LetterBox((640, 640), auto=False)(image=image)
    np.testing.assert_allclose(ours[0].transpose(1, 2, 0), reference / 255, atol=1e-7)


def test_nms_class_awareness_and_inverse_geometry():
    output = np.zeros((1, 84, 8400), np.float32)
    output[0, :4, :3] = np.array([[320, 320, 320], [320, 320, 320], [100, 100, 100], [60, 60, 60]])
    output[0, 6, :2] = [0.9, 0.8]  # COCO car, overlapping same class: retain one.
    output[0, 4, 2] = 0.7  # COCO person, same box: retain different class.
    _, geometry = letterbox(np.zeros((320, 640, 3), np.uint8))
    result = decode(output, geometry)
    assert result.shape == (2, 6)
    assert set(result[:, 5]) == {0, 2}
    np.testing.assert_allclose(result[:, :4], [[270, 130, 370, 190]] * 2)
    assert nms(np.empty((0, 4)), np.empty(0), 0.7).size == 0


def test_invalid_detector_output_rejected():
    with pytest.raises(ValueError):
        decode(np.zeros((1, 85, 8400)), {})


def test_calibration_does_not_use_validation_or_test():
    records = select_panel("train", 16)
    assert len(records) == 64
    assert all(r.split == "train" for r in records)
    assert {r.weather for r in records} == {"fog", "rain", "snow", "smoke"}
    assert len({r.sample_id for r in records}) == 64


def test_dawn_requires_xml_class_agreement(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/deployment/edge_dawn.py"
    spec = importlib.util.spec_from_file_location("edge_dawn_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = tmp_path / "road.jpg"
    Image.new("RGB", (100, 50)).save(image)
    (tmp_path / "road.xml").write_text(
        "<annotation><size><width>100</width><height>50</height></size>"
        "<object><name>car</name><bndbox><xmin>10</xmin><ymin>5</ymin><xmax>90</xmax><ymax>45</ymax>"
        "</bndbox></object></annotation>"
    )
    (tmp_path / "road.txt").write_text("3 0.5 0.5 0.8 0.8\n")
    labels, counts, _ = module.annotations(image)
    assert labels.startswith("2 ") and counts["car"] == 1
    (tmp_path / "road.txt").write_text("2 0.5 0.5 0.8 0.8\n")
    with pytest.raises(ValueError, match="class mapping"):
        module.annotations(image)


@pytest.mark.parametrize("shift, passes", [(0.0, True), (0.2, False)])
def test_precision_gate_rejects_changed_restoration(tmp_path, monkeypatch, shift, passes):
    path = Path(__file__).resolve().parents[1] / "scripts/deployment/edge_build.py"
    spec = importlib.util.spec_from_file_location("edge_build_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = np.random.default_rng(11).uniform(-0.5, 0.5, (1, 3, 16, 16)).astype(np.float32)
    fixture = tmp_path / "fixtures.npz"
    np.savez(
        fixture, input_fog=data, expected_fog=data, target_fog=data, valid_hw_fog=np.array([16, 16])
    )

    class FakeSession:
        def __init__(self, _):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def run(self, x):
            return x + shift

    monkeypatch.setattr(module, "TrtSession", FakeSession)
    result = module.validate(tmp_path / "unused.engine", fixture, "restormer", "fp16")
    assert result["passed"] is passes
