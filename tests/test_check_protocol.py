import importlib.util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_protocol.py"
    spec = importlib.util.spec_from_file_location("check_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


check_consistency = _load().check_consistency


def _metadata(**overrides):
    base = {
        "model_id": "m",
        "model_revision": "r1",
        "git_commit": "c1",
        "seed": 42,
        "generation": {
            "prompt_version": "cg_v1",
            "max_new_tokens": 256,
            "do_sample": False,
            "use_cache": True,
        },
        "dataset": {"dataset_hash": "h1"},
    }
    base.update(overrides)

    return base


def test_matching_runs_pass():
    errors, warnings = check_consistency(
        [("a", _metadata()), ("b", _metadata(model_revision="r2"))]
    )

    assert errors == []
    assert warnings == []


def test_mismatched_protocol_fails():
    other = _metadata()
    other["generation"] = {**other["generation"], "max_new_tokens": 512}

    errors, _ = check_consistency([("a", _metadata()), ("b", other)])

    assert len(errors) == 1
    assert "max_new_tokens" in errors[0]


def test_missing_key_fails():
    other = _metadata()
    del other["dataset"]

    errors, _ = check_consistency([("a", _metadata()), ("b", other)])

    assert any("dataset_hash" in error for error in errors)


def test_commit_drift_warns():
    errors, warnings = check_consistency(
        [("a", _metadata()), ("b", _metadata(git_commit="c2"))]
    )

    assert errors == []
    assert len(warnings) == 1


def test_empty_rejected():
    with pytest.raises(ValueError):
        check_consistency([])
