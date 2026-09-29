from __future__ import annotations

import hashlib
import runpy
import zipfile
from pathlib import Path

import pytest

from aegis_perception.data.catalog import select_datasets, validate_archive
from aegis_perception.data.manifest import image_index

REPOSITORY = Path(__file__).resolve().parents[1]


def test_selection_excludes_optional_downloads_and_keeps_requested_order():
    datasets = [
        {"id": "smoke", "selection": "core", "auto_download": True, "conditions": ["smoke"]},
        {"id": "fog", "selection": "core", "auto_download": True, "conditions": ["fog"]},
        {"id": "gated", "selection": "gated", "auto_download": False, "conditions": ["fog"]},
    ]
    assert select_datasets(datasets, condition="fog") == [datasets[1]]
    assert select_datasets(datasets, ids=["fog", "smoke", "fog"]) == [datasets[1], datasets[0]]
    with pytest.raises(ValueError, match="Unknown dataset"):
        select_datasets(datasets, ids=["missing"])


def test_archive_rejects_incomplete_and_corrupt_downloads(tmp_path):
    archive = tmp_path / "sample.zip"
    archive.write_bytes(b"archive")
    digest = hashlib.sha256(b"archive").hexdigest()
    metadata = {"expected_bytes": 7, "sha256": digest}
    assert validate_archive(archive, metadata) == digest
    marker = tmp_path / "sample.zip.aria2"
    marker.touch()
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_archive(archive, metadata)
    marker.unlink()
    archive.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate_archive(archive, metadata)
    archive.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="expected 7 bytes"):
        validate_archive(archive, metadata)


def test_extraction_rejects_zip_path_traversal(tmp_path):
    module = runpy.run_path(str(REPOSITORY / "scripts/data/extract_datasets.py"))
    extract_zip = module["extract_zip"]
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escaped.txt", "should not be extracted")
    with pytest.raises(RuntimeError, match="unsafe ZIP member"):
        extract_zip(archive, tmp_path / "output")
    assert not (tmp_path / "escaped.txt").exists()


def test_image_index_rejects_ambiguous_pair_keys(tmp_path):
    (tmp_path / "scene.png").touch()
    (tmp_path / "scene.jpg").touch()
    with pytest.raises(ValueError, match="duplicate pair key"):
        image_index(tmp_path)
