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
            "enable_thinking": False,
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


def _knowledge(**overrides):
    base = {
        "model_id": "m",
        "model_revision": "r1",
        "seed": 42,
        "scoring": {
            "score_mode": "letter",
            "score_span": None,
            "letter_style": "natural",
            "prompt_format": "plain",
            "exclude_fewshot": True,
            "limit": None,
            "batch_size": 8,
        },
        "dataset": {"primary_questions": 10, "retention_questions": 5},
    }
    base.update(overrides)

    return base


def test_knowledge_runs_are_checked_on_their_own_track():
    errors, _ = check_consistency(
        [("a", _knowledge()), ("b", _knowledge(model_revision="r2"))]
    )

    assert errors == []


def test_knowledge_scoring_mismatch_fails():
    other = _knowledge()
    other["scoring"] = {**other["scoring"], "letter_style": "official"}

    errors, _ = check_consistency([("a", _knowledge()), ("b", other)])

    assert any("letter_style" in e for e in errors)


def test_mixed_tracks_fail():
    errors, _ = check_consistency([("a", _knowledge()), ("b", _metadata())])

    assert any("track" in e for e in errors)


def test_unpinned_model_fails():
    errors, _ = check_consistency(
        [("a", _knowledge(model_revision=None)), ("b", _knowledge())]
    )

    assert any("di-pin" in e for e in errors)


def test_local_weights_hash_counts_as_pinned():
    errors, _ = check_consistency(
        [
            ("a", _knowledge(model_revision=None, model_weights_sha256="abc")),
            ("b", _knowledge()),
        ]
    )

    assert errors == []
