"""Verify hardware-specific pruning layout, exclusions and original pruning reuse."""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compression_screen", ROOT / "scripts/deployment/edge_compression_screen.py"
)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def test_two_four_prunes_input_channel_axis_and_preserves_other_layers():
    model = nn.ModuleDict(
        {
            "eligible": nn.Conv2d(8, 3, 3, bias=False),
            "depthwise": nn.Conv2d(8, 8, 3, groups=8, bias=False),
            "rgb": nn.Conv2d(3, 8, 3, bias=False),
        }
    )
    original = {k: v.clone() for k, v in model.state_dict().items()}
    # Every spatial position has the same strictly increasing channel weights.
    with torch.no_grad():
        model["eligible"].weight.copy_(torch.arange(1, 9).reshape(1, 8, 1, 1))
    assert screen.prune_two_four(model) == ["eligible"]
    expected = torch.tensor([0, 0, 3, 4, 0, 0, 7, 8]).reshape(1, 8, 1, 1)
    torch.testing.assert_close(model["eligible"].weight, expected.expand(3, 8, 3, 3).float())
    for name in ("depthwise", "rgb"):
        torch.testing.assert_close(model[name].weight, original[name + ".weight"])


def test_two_four_idempotent_with_tied_magnitudes():
    model = nn.Sequential(nn.Conv2d(8, 2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(1)
    screen.prune_two_four(model)
    once = model[0].weight.clone()
    screen.prune_two_four(model)
    torch.testing.assert_close(model[0].weight, once)
    assert int((once == 0).sum()) == 8


@pytest.mark.parametrize("failed_precision", ["", "fp32", "int8"])
def test_device_comparison_continues_after_build_failure(tmp_path, failed_precision):
    """Simulate orchestration only: never import TensorRT or pretend to measure GPU."""
    scripts = tmp_path / "scripts/deployment"
    scripts.mkdir(parents=True)
    driver = scripts / "edge_compression_device.sh"
    shutil.copy2(ROOT / "scripts/deployment/edge_compression_device.sh", driver)
    fake = tmp_path / "fake_python"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "with open('calls.txt', 'a') as f: f.write(' '.join(args) + '\\n')\n"
        "if args[0].endswith('edge_build.py') and 'restormer' in args:\n"
        "    precision = args[args.index('--precision') + 1]\n"
        "    if precision == os.environ.get('FAIL_PRECISION'): sys.exit(1)\n"
    )
    fake.chmod(0o755)
    result = subprocess.run(
        ["bash", str(driver)],
        env={**os.environ, "AEGIS_PYTHON": str(fake), "FAIL_PRECISION": failed_precision},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == (1 if failed_precision else 0), result.stderr
    status = next((tmp_path / "results").glob("*/status.tsv")).read_text()
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    for precision in ("fp32", "fp16", "int8"):
        assert f"{precision}\t" in status
        measured = any(
            "edge_run.py" in call and f"comparison_restormer_{precision}.engine" in call
            for call in calls
        )
        assert measured == (precision != failed_precision)
    assert any("--calibration" in call and "--precision int8" in call for call in calls)
