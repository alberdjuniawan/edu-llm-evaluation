import pytest
from pydantic import ValidationError

from edu_eval.generation.schema import (
    ControlledGenerationCase,
    GradeEvidence,
    SourceReference,
)


def _reference(**overrides):
    payload = {
        "source_id": "REF-X",
        "source_title": "Kapsul referensi project-authored - X (Y) [draf]",
        "source_url": None,
        "source_type": "project_capsule",
        "source_version": "capsule-v1-draft",
        "locator": None,
        "text": "Teks acuan.",
    }
    payload.update(overrides)

    return payload


def _evidence(phase, grade, **overrides):
    payload = {
        "curriculum_phase": phase,
        "grade": grade,
        "evidence_source": "rational",
        "evidence_locator": None,
        "evidence_text": "Bukti.",
        "evidence_scope": "phase_only",
    }
    payload.update(overrides)

    return payload


def _case(**overrides):
    payload = {
        "case_id": "CG-001",
        "subject": "IPA",
        "concept": "Fotosintesis",
        "task_type": "materi",
        "reference": _reference(),
        "target_evidence": {
            "SD6": _evidence("C", 6),
            "SMP7": _evidence("D", 7),
            "SMP9": _evidence("D", 9),
            "SMA10": _evidence("E", 10),
        },
    }
    payload.update(overrides)

    return payload


def test_valid_case_passes():
    case = ControlledGenerationCase.model_validate(_case())

    assert case.case_id == "CG-001"
    assert case.reference.source_id == "REF-X"
    assert set(case.target_evidence) == {"SD6", "SMP7", "SMP9", "SMA10"}


def test_missing_grade_evidence_fails():
    payload = _case()
    del payload["target_evidence"]["SMP9"]

    with pytest.raises(ValidationError):
        ControlledGenerationCase.model_validate(payload)


def test_wrong_phase_mapping_fails():
    payload = _case()
    payload["target_evidence"]["SD6"] = _evidence("D", 6)

    with pytest.raises(ValidationError):
        ControlledGenerationCase.model_validate(payload)


def test_wrong_grade_number_fails():
    payload = _case()
    payload["target_evidence"]["SMA10"] = _evidence("E", 11)

    with pytest.raises(ValidationError):
        ControlledGenerationCase.model_validate(payload)


def test_official_source_requires_url_and_locator():
    with pytest.raises(ValidationError):
        SourceReference.model_validate(_reference(source_type="official_textbook"))


def test_capsule_must_not_carry_url():
    with pytest.raises(ValidationError):
        SourceReference.model_validate(
            _reference(source_url="https://example.com/fake")
        )


def test_grade_specific_claim_requires_locator():
    with pytest.raises(ValidationError):
        GradeEvidence.model_validate(_evidence("D", 7, evidence_scope="grade_specific"))


def test_grade_specific_claim_with_locator_passes():
    evidence = GradeEvidence.model_validate(
        _evidence(
            "D",
            7,
            evidence_scope="grade_specific",
            evidence_locator="ATP IPA Fase D",
        )
    )

    assert evidence.evidence_scope == "grade_specific"
