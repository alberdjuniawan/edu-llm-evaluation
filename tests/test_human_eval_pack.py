import pytest

from edu_eval.human_eval.pack import build_pack


def _records():
    return [
        {
            "model_id": "m1",
            "case_id": "CG-001",
            "target_grade": "SD6",
            "output_text": "materi satu",
        },
        {
            "model_id": "m0",
            "case_id": "CG-002",
            "target_grade": "SMP7",
            "output_text": "materi dua",
        },
    ]


def _references():
    return {
        "CG-001": {
            "subject": "IPA",
            "concept": "Fotosintesis",
            "reference_text": "acuan satu",
        },
        "CG-002": {
            "subject": "IPS",
            "concept": "Peta",
            "reference_text": "acuan dua",
        },
    }


def test_build_pack_deterministic():
    first, mapping_first = build_pack(_records(), _references(), seed=42)
    second, mapping_second = build_pack(_records(), _references(), seed=42)

    assert first == second
    assert mapping_first == mapping_second
    assert len(first) == 2


def test_build_pack_hides_model_identity():
    pack, mapping = build_pack(_records(), _references(), seed=42)

    assert set(mapping.values()) == {"A", "B"}

    for item in pack:
        assert "model_id" not in item
        assert "source" not in item
        assert item["blind_model_label"] in {"A", "B"}


def test_build_pack_carries_rating_context():
    pack, _ = build_pack(_records(), _references(), seed=42)
    by_case = {item["case_id"]: item for item in pack}

    assert by_case["CG-001"]["subject"] == "IPA"
    assert by_case["CG-001"]["reference_text"] == "acuan satu"
    assert by_case["CG-002"]["concept"] == "Peta"


def test_build_pack_rejects_empty_records():
    with pytest.raises(ValueError):
        build_pack([], _references(), seed=42)
