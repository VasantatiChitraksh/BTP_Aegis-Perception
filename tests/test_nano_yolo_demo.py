"""CPU checks for the legacy Nano adapter; no claim of device execution."""

import ast
import importlib.util
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/deployment/nano_yolo_demo.py"
spec = importlib.util.spec_from_file_location("nano_demo", PATH)
nano = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nano)


def test_python36_syntax():
    ast.parse(PATH.read_text(), feature_version=(3, 6))


def test_engine_context_is_selected_and_released_on_error():
    events = []

    class Context:
        def push(self):
            events.append("push")

        def pop(self):
            events.append("pop")

    engine = nano.Engine.__new__(nano.Engine)
    engine.device_context = Context()

    def fail(_):
        raise ValueError("execution failed")

    engine._run = fail
    with pytest.raises(ValueError, match="execution failed"):
        engine.run(None)
    assert events == ["push", "pop"]


def test_bgr_to_rgb_and_inverse_letterbox():
    image = np.zeros((320, 640, 3), np.uint8)
    image[:] = (10, 20, 30)
    value, geometry = nano.letterbox(image)
    assert value.shape == (1, 3, 640, 640)
    np.testing.assert_allclose(value[0, :, 160, 0], np.array([30, 20, 10]) / 255)
    np.testing.assert_allclose(value[0, :, 0, 0], np.repeat(114 / 255, 3))
    output = np.zeros((1, 84, 8400), np.float32)
    output[0, :4, :3] = [[320] * 3, [320] * 3, [100] * 3, [60] * 3]
    output[0, 6, :2] = [0.9, 0.8]
    output[0, 4, 2] = 0.7
    boxes = nano.decode(output, geometry)
    assert boxes.shape == (2, 6)
    assert set(boxes[:, 5]) == {0, 2}
    np.testing.assert_allclose(boxes[:, :4], [[270, 130, 370, 190]] * 2)
    assert nano.decode(np.zeros_like(output), geometry).shape == (0, 6)


@pytest.mark.parametrize(
    "box_error,score_error,passed", [(0, 0, True), (1, 0, False), (0, 0.03, False)]
)
def test_reference_gate(tmp_path, box_error, score_error, passed):
    path = tmp_path / "fixtures.npz"
    np.savez(
        path,
        input_test=np.zeros((1, 3, 640, 640), np.float32),
        expected_test=np.zeros((1, 84, 8400), np.float32),
    )

    class Engine:
        def run(self, _):
            value = np.zeros((1, 84, 8400), np.float32)
            value[:, :4] = box_error
            value[:, 4:] = score_error
            return value

    assert nano.validate(Engine(), path)["passed"] is passed


def test_preview_localhost_and_latest_frame():
    preview = nano.Preview(0)
    try:
        host, port = preview.server.server_address
        assert host == "127.0.0.1"
        base = "http://{}:{}".format(host, port)
        with urlopen(base, timeout=3) as response:
            assert b"No Restormer" in response.read()
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/frame.jpg", timeout=3)
        assert error.value.code == 503
        preview.jpeg = b"test frame bytes"
        with urlopen(base + "/frame.jpg?t=1", timeout=3) as response:
            assert response.read() == preview.jpeg
            assert response.headers["Cache-Control"] == "no-store"
    finally:
        preview.close()
