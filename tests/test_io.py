import hashlib
import json

import pandas as pd
import pytest

from moto_study.io import load_prepared, project_root, read_bytes, save_prepared, source_path


def test_corrupt_raw_object_fails_before_decode(tmp_path):
    path = tmp_path / "data/raw/example.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_bytes(
            tmp_path,
            {
                "path": "data/raw/example.json",
                "sha256": hashlib.sha256(b"expected").hexdigest(),
                "source_id": "test",
            },
        )


def test_source_path_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        source_path(tmp_path, {"path": "../outside.json"})


def test_missing_preparation_explains_next_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="notebook 01"):
        load_prepared(tmp_path)


def test_changed_prepared_file_rejected(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/source_manifest.json").write_text(json.dumps({"sources": []}))
    frame = pd.DataFrame({"example": [1]})
    save_prepared(tmp_path, frame, frame)
    load_prepared(tmp_path)
    (tmp_path / "data/processed/trade.parquet").write_bytes(b"broken")
    with pytest.raises(ValueError, match="Prepared file changed"):
        load_prepared(tmp_path)


def test_project_root_is_portable(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/source_manifest.json").write_text("{}")
    (tmp_path / "pyproject.toml").write_text("")
    nested = tmp_path / "notebooks"
    nested.mkdir()
    assert project_root(nested) == tmp_path
