import pytest

from edu_eval.generation.cases import (
    dataset_fingerprint,
    generation_config_hash,
    load_cases,
    require_research_ready,
    select_by_subject,
    select_full,
    select_pilot,
    validate_resume_identity,
    validation_report,
)
from edu_eval.generation.config import GenerationConfig, require_deterministic
from edu_eval.generation.schema import ControlledGenerationCase


def _evidence(phase, grade, scope="phase_only", locator=None):
    return {
        "curriculum_phase": phase,
        "grade": grade,
        "evidence_source": "s",
        "evidence_locator": locator,
        "evidence_text": "t",
        "evidence_scope": scope,
    }


def _reference(kind="capsule"):
    if kind == "capsule":
        return {
            "source_id": "r",
            "source_title": "t",
            "source_url": None,
            "source_type": "project_capsule",
            "source_version": None,
            "locator": None,
            "text": "x",
        }

    return {
        "source_id": "r",
        "source_title": "t",
        "source_url": "https://example.com/x",
        "source_type": "official_textbook",
        "source_version": "v1",
        "locator": "p.1",
        "text": "x",
    }


def _case(case_id, subject, scope="phase_only", ref="capsule"):
    locator = "ATP X" if scope == "grade_specific" else None

    return ControlledGenerationCase.model_validate(
        {
            "case_id": case_id,
            "subject": subject,
            "concept": "c",
            "reference": _reference(ref),
            "target_evidence": {
                "SD6": _evidence("C", 6, scope, locator),
                "SMP7": _evidence("D", 7, scope, locator),
                "SMP9": _evidence("D", 9, scope, locator),
                "SMA10": _evidence("E", 10, scope, locator),
            },
        }
    )


def _cases():
    subjects = ["IPA"] * 3 + ["IPS"] * 3

    return [
        _case(f"CG-{index:03d}", subject)
        for index, subject in enumerate(subjects, start=1)
    ]


def _eligible_cases():
    subjects = ["IPA"] * 3 + ["IPS"] * 3

    return [
        _case(f"CG-{index:03d}", subject, "grade_specific", "official")
        for index, subject in enumerate(subjects, start=1)
    ]


def _config():
    return GenerationConfig(
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
        prompt_version="cg_v1",
        target_grades=["SD6", "SMP7", "SMP9", "SMA10"],
        subjects=["IPA", "IPS"],
        pilot_cases_per_subject=1,
        full_cases_per_subject=3,
    )


def test_pilot_selection_deterministic():
    cases = _cases()
    config = _config()

    first = [c.case_id for c in select_pilot(cases, config)]
    second = [c.case_id for c in select_pilot(cases, config)]

    assert len(first) == 2
    assert first == second


def test_full_selection_counts():
    assert len(select_full(_cases(), _config())) == 6


def test_selection_shortfall_raises():
    with pytest.raises(ValueError):
        select_by_subject(_cases()[:2], 3)


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "does_not_exist.jsonl")


def test_draft_dataset_is_not_research_ready():
    report = validation_report(_cases(), _config())

    assert report["full_ready"] == "PASS"
    assert report["research_ready"].startswith("NOT_READY")
    assert "project_capsule" in report["research_ready"]
    assert "phase_only" in report["research_ready"]


def test_eligible_dataset_is_research_ready():
    report = validation_report(_eligible_cases(), _config())

    assert report["research_ready"] == "PASS"
    assert report["provenance"] == "PASS"


def test_subject_mismatch_blocks_research_ready():
    config = _config().model_copy(update={"subjects": ["IPA", "IPS", "MTK"]})
    report = validation_report(_cases(), config)

    assert report["subjects_match"] == "FAIL"
    assert report["research_ready"].startswith("NOT_READY")


def test_duplicates_block_research_ready():
    cases = _cases() + [_cases()[0]]
    report = validation_report(cases, _config())

    assert report["duplicate_case_ids"].startswith("FAIL")
    assert report["research_ready"].startswith("NOT_READY")


def test_require_research_ready_gate():
    with pytest.raises(RuntimeError):
        require_research_ready(validation_report(_cases(), _config()))

    require_research_ready(validation_report(_eligible_cases(), _config()))


def test_validate_resume_identity():
    validate_resume_identity({}, {"a": 1})
    validate_resume_identity({"a": 1}, {"a": 1})
    validate_resume_identity({"a": 1}, {"a": 1, "b": 2})

    with pytest.raises(RuntimeError):
        validate_resume_identity({"a": 1}, {"a": 2})


def test_require_deterministic_gate():
    require_deterministic(_config(), allow_sampling=False)

    sampling = _config().model_copy(update={"do_sample": True})

    with pytest.raises(RuntimeError):
        require_deterministic(sampling, allow_sampling=False)

    require_deterministic(sampling, allow_sampling=True)


def test_fingerprints_stable_and_sensitive(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")

    first = dataset_fingerprint(path)
    second = dataset_fingerprint(path)

    assert first == second

    path.write_text('{"a": 2}\n', encoding="utf-8")

    assert dataset_fingerprint(path) != first
    assert generation_config_hash(_config()) == generation_config_hash(_config())
