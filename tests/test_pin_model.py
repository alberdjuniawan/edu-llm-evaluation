import importlib.util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pin_model.py"
    spec = importlib.util.spec_from_file_location("pin_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_hash_is_stable_and_content_sensitive(tmp_path):
    mod = _load()
    (tmp_path / "model.safetensors").write_bytes(b"abc")
    first, listing = mod.hash_local_weights(tmp_path)
    second, _ = mod.hash_local_weights(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"abd")
    third, _ = mod.hash_local_weights(tmp_path)

    assert first == second != third
    assert listing == [("model.safetensors", 3)]


def test_no_weights_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load().hash_local_weights(tmp_path)
