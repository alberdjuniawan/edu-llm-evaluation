from pathlib import Path

import pytest

from edu_eval.generation.cases import (
    load_cases,
    select_by_subject,
    select_full,
    select_pilot,
    validation_report,
)
from edu_eval.generation.config import GenerationConfig

CASES_PATH = Path("data/processed/controlled_generation_cases.jsonl")
CONFIG_PATH = Path("configs/generation.yaml")


def _config():
    return GenerationConfig.from_yaml(CONFIG_PATH)


def test_canonical_dataset_loads_60_cases():
    cases = load_cases(CASES_PATH)

    assert len(cases) == 60


def test_subject_distribution_is_12_each():
    cases = load_cases(CASES_PATH)
    report = validation_report(cases, _config())

    assert report["subjects"] == {
        "Bahasa Indonesia": 12,
        "IPA": 12,
        "IPS": 12,
        "Informatika": 12,
        "Pendidikan Pancasila": 12,
    }
    assert report["schema"] == "PASS"
    assert report["duplicate_case_ids"] == "PASS"
    assert report["pilot_ready"] == "PASS"
    assert report["full_ready"] == "PASS"


def test_pilot_selection_is_10_cases_deterministic():
    cases = load_cases(CASES_PATH)
    config = _config()

    first = [c.case_id for c in select_pilot(cases, config)]
    second = [c.case_id for c in select_pilot(cases, config)]

    assert len(first) == 10
    assert first == second
    assert len(select_pilot(cases, config)) * 4 == 40


def test_full_selection_is_60_cases():
    cases = load_cases(CASES_PATH)
    config = _config()

    selected = select_full(cases, config)

    assert len(selected) == 60
    assert len(selected) * 4 == 240


def test_selection_rejects_unknown_subject_shortfall():
    cases = load_cases(CASES_PATH)[:5]

    with pytest.raises(ValueError):
        select_by_subject(cases, 12)


def test_missing_dataset_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "does_not_exist.jsonl")
