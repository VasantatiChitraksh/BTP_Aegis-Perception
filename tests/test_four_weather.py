from __future__ import annotations

import runpy
from collections import defaultdict
from pathlib import Path

import pytest
from PIL import Image

from aegis_perception.data.manifest import PairRecord, write_manifest
from aegis_perception.data.sampling import weather_scene_weights

ROOT = Path(__file__).resolve().parents[1]


def record(name, weather, group, split="train"):
    return PairRecord(name, Path(name), Path(name), split, weather, "fixture", group)


def test_weather_and_scene_sampling_does_not_count_variants_as_scenes():
    rows = [record(str(i), "rain", "a") for i in range(20)]
    rows += [record("b", "rain", "b"), record("c", "smoke", "c")]
    weights = weather_scene_weights(rows, {"rain": 0.8, "smoke": 0.2})
    totals = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True):
        totals[row.weather, row.group_id] += weight
    assert totals == pytest.approx({("rain", "a"): 0.4, ("rain", "b"): 0.4, ("smoke", "c"): 0.2})
    with pytest.raises(ValueError, match="group_id"):
        weather_scene_weights([record("x", "rain", "")], {"rain": 1})


def test_repair_preserves_heldout_groups_and_deterministic_assignment():
    module = runpy.run_path(str(ROOT / "scripts/data/prepare_four_weather.py"))
    kept, excluded = module["remove_split_overlap"](
        [
            record("a", "rain", "scene", "train"),
            record("b", "rain", "scene", "val"),
            record("c", "rain", "scene", "test"),
            record("d", "rain", "other", "train"),
        ]
    )
    assert [row.sample_id for row in kept] == ["c", "d"]
    assert len(excluded) == 2
    groups = {str(i) for i in range(40)}
    assignment = module["split_groups"](groups, 42, 0.15)
    assert assignment == module["split_groups"](set(reversed(sorted(groups))), 42, 0.15)
    assert set(assignment.values()) == {"train", "val", "test"}


def test_audit_catches_identical_pixels_with_different_encoding(tmp_path, monkeypatch, capsys):
    first, second = tmp_path / "a.png", tmp_path / "b.bmp"
    image = Image.new("RGB", (16, 16), "red")
    image.save(first)
    image.save(second)
    manifest = tmp_path / "pairs.csv"
    write_manifest(
        manifest,
        [
            PairRecord("a", first, first, "train", "rain", "fixture"),
            PairRecord("b", second, second, "test", "rain", "fixture"),
        ],
    )
    monkeypatch.setattr("sys.argv", ["audit_manifest.py", str(manifest)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(ROOT / "scripts/data/audit_manifest.py"), run_name="__main__")
    assert exc.value.code == 1
    assert "decoded clean target(s)" in capsys.readouterr().out


def test_validation_macro_gives_equal_weight_to_weather_and_scene():
    import torch

    from aegis_perception.training import _validate

    def batch(error, weather, group):
        return {
            "input": torch.full((1, 3, 8, 8), error),
            "target": torch.zeros(1, 3, 8, 8),
            "weather": [weather],
            "group_id": [group],
        }

    rows = [batch(0.1, "rain", "a"), batch(0.2, "rain", "b"), batch(0.4, "smoke", "c")]
    expected = _validate(torch.nn.Identity(), rows, "cpu")
    duplicated = _validate(torch.nn.Identity(), rows + [rows[0]] * 20, "cpu")
    assert duplicated["macro_scene_psnr_db"] == pytest.approx(expected["macro_scene_psnr_db"])
    assert duplicated["psnr_db"] != pytest.approx(expected["psnr_db"])
    assert expected["macro_scene_psnr_db"] == pytest.approx(
        (expected["rain_scene_psnr_db"] + expected["smoke_scene_psnr_db"]) / 2
    )
