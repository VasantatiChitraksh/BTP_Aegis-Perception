"""Check the ONNX adapter's real weight changes and rejection behavior."""

import importlib.util
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "custom_candidate", ROOT / "scripts/deployment/edge_custom_candidate.py"
)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def model():
    weights = [
        numpy_helper.from_array(
            np.array([0.123, 0.931, -0.387], dtype=np.float32).reshape(3, 1, 1, 1), name
        )
        for name in ("network.patch_embed.proj.weight", "middle.weight", "network.output.weight")
    ]
    nodes = [
        helper.make_node("Conv", ["x" if i == 0 else f"v{i}", w.name], [f"v{i + 1}"], group=3)
        for i, w in enumerate(weights)
    ]
    graph = helper.make_graph(
        nodes,
        "test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("v3", TensorProto.FLOAT, [1, 3, 8, 8])],
        weights,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)


def test_preserves_boundaries_and_float_storage():
    graph = model()
    # Multiple values per channel are needed to exercise rounding loss.
    weight = np.array([0.123, 0.931, -0.387] * 3, dtype=np.float32).reshape(3, 1, 1, 3)
    graph.graph.initializer[1].CopyFrom(numpy_helper.from_array(weight, "middle.weight"))
    before = [numpy_helper.to_array(t).copy() for t in graph.graph.initializer]
    stats = candidate.transform(graph)
    after = [numpy_helper.to_array(t) for t in graph.graph.initializer]
    assert len(stats["preserved_weights"]) == 2
    assert stats["converted_weights"] == ["middle.weight"]
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[2], after[2])
    assert not np.array_equal(before[1], after[1])
    assert after[1].dtype == np.float32


def test_screen_rejects_wrong_reference(tmp_path):
    path = tmp_path / "model.onnx"
    onnx.save(model(), path)
    fixtures = tmp_path / "fixtures.npz"
    np.savez(
        fixtures,
        input_0=np.ones((1, 3, 8, 8), np.float32),
        expected_0=np.zeros((1, 3, 8, 8), np.float32),
        target_0=np.zeros((1, 3, 8, 8), np.float32),
        valid_hw_0=np.array([8, 8]),
    )
    assert candidate.screen(path, fixtures)["passed"] is False


@pytest.mark.parametrize("bits,percentile", [(3, 99), (8, 0), (8, 101)])
def test_invalid_quantization_parameters(bits, percentile):
    with pytest.raises(ValueError):
        candidate.transform(model(), bits, percentile)
