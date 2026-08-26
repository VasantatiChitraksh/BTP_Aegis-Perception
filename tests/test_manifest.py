from __future__ import annotations

from aegis_perception.data.manifest import deterministic_split


def test_split_is_deterministic() -> None:
    first = deterministic_split("scene-17", seed=42, train_fraction=0.8, val_fraction=0.1)
    second = deterministic_split("scene-17", seed=42, train_fraction=0.8, val_fraction=0.1)
    assert first == second


def test_same_group_stays_in_one_split() -> None:
    split = deterministic_split("drive-04", seed=42, train_fraction=0.8, val_fraction=0.1)
    assert split in {"train", "val", "test"}
    assert split == deterministic_split(
        "drive-04", seed=42, train_fraction=0.8, val_fraction=0.1
    )
