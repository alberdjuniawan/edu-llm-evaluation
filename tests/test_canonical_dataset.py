from pathlib import Path

import pytest

from edu_eval.generation.cases import load_cases, validation_report
from edu_eval.generation.config import GenerationConfig

pytestmark = pytest.mark.integration

CASES_PATH = Path("data/processed/controlled_generation_cases.jsonl")
CONFIG_PATH = Path("configs/generation.yaml")


def _load():
    if not CASES_PATH.exists():
        pytest.skip("canonical dataset absent")

    return load_cases(CASES_PATH)


def test_canonical_dataset_loads_60_cases():
    assert len(_load()) == 60


def test_canonical_subject_distribution():
    report = validation_report(_load(), GenerationConfig.from_yaml(CONFIG_PATH))

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


def test_canonical_dataset_draft_status_documented():
    report = validation_report(_load(), GenerationConfig.from_yaml(CONFIG_PATH))

    # Update after SME review + grade-specific evidence land.
    assert report["research_ready"].startswith("NOT_READY")
    assert report["phase_only_count"] == 240
