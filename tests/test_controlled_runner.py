import pytest

from edu_eval.generation.config import GenerationConfig
from edu_eval.generation.controlled import ControlledGenerationRunner
from edu_eval.generation.schema import ControlledGenerationCase

CASE_PAYLOAD = {
    "case_id": "CG-001",
    "subject": "IPA",
    "concept": "Fotosintesis",
    "task_type": "materi",
    "reference": {
        "source_id": "REF-CG-001",
        "source_title": "Kapsul referensi project-authored",
        "source_url": None,
        "source_type": "project_capsule",
        "source_version": "capsule-v1-draft",
        "locator": None,
        "text": "Tumbuhan membuat makanan melalui fotosintesis.",
    },
    "target_evidence": {
        "SD6": {
            "curriculum_phase": "C",
            "grade": 6,
            "evidence_source": "r",
            "evidence_locator": None,
            "evidence_text": "bukti",
            "evidence_scope": "phase_only",
        },
        "SMP7": {
            "curriculum_phase": "D",
            "grade": 7,
            "evidence_source": "r",
            "evidence_locator": None,
            "evidence_text": "bukti",
            "evidence_scope": "phase_only",
        },
        "SMP9": {
            "curriculum_phase": "D",
            "grade": 9,
            "evidence_source": "r",
            "evidence_locator": None,
            "evidence_text": "bukti",
            "evidence_scope": "phase_only",
        },
        "SMA10": {
            "curriculum_phase": "E",
            "grade": 10,
            "evidence_source": "r",
            "evidence_locator": None,
            "evidence_text": "bukti",
            "evidence_scope": "phase_only",
        },
    },
}


class FakeGenerationRunner:
    def __init__(self):
        self.prompts: list[str] = []
        self.config = GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            prompt_version="cg_v1",
            target_grades=["SD6", "SMP7", "SMP9", "SMA10"],
            pilot_cases_per_subject=2,
            full_cases_per_subject=12,
        )

    def generate(self, prompt: str):
        self.prompts.append(prompt)

        return {
            "text": "materi",
            "input_tokens": 10,
            "output_tokens": 5,
            "latency_seconds": 0.1,
            "hit_max_new_tokens": False,
        }


def _case():
    return ControlledGenerationCase.model_validate(CASE_PAYLOAD)


def _runner():
    return ControlledGenerationRunner(
        generation_runner=FakeGenerationRunner(),  # type: ignore[arg-type]
        model_id="smoke_qwen",
        target_grades=["SD6", "SMP7", "SMP9", "SMA10"],
        prompt_version="cg_v1",
    )


def test_build_prompt_holds_reference_constant():
    runner = _runner()
    case = _case()

    prompt_sd = runner.build_prompt(case, "SD6")
    prompt_sma = runner.build_prompt(case, "SMA10")

    assert case.reference.text in prompt_sd
    assert case.reference.text in prompt_sma
    assert "IPA" in prompt_sd and "Fotosintesis" in prompt_sd


def test_build_prompt_has_no_hardcoded_complexity_constraints():
    runner = _runner()

    prompt = runner.build_prompt(_case(), "SD6").lower()

    assert "kata per kalimat" not in prompt
    assert "readability" not in prompt
    assert "fkgl" not in prompt


def test_prompt_id_is_deterministic():
    runner = _runner()
    case = _case()

    first = runner.generate(case, "SMP7")["prompt_id"]
    second = runner.generate(case, "SMP7")["prompt_id"]

    assert first == second
    assert len(first) == 16


def test_generate_preserves_lineage():
    runner = _runner()

    result = runner.generate(_case(), "SMP9")

    assert result["case_id"] == "CG-001"
    assert result["source_id"] == "REF-CG-001"
    assert result["target_grade"] == "SMP9"
    assert result["target_level"] == "SMP"
    assert result["output_text"] == "materi"
    assert result["model_id"] == "smoke_qwen"
    assert result["prompt_version"] == "cg_v1"
    assert result["generation_config"] == {
        "max_new_tokens": 256,
        "do_sample": False,
        "use_cache": True,
    }


def test_generate_case_covers_all_target_grades():
    runner = _runner()

    results = runner.generate_case(_case())

    assert [r["target_grade"] for r in results] == [
        "SD6",
        "SMP7",
        "SMP9",
        "SMA10",
    ]


def test_unknown_target_grade_rejected():
    with pytest.raises(ValueError):
        ControlledGenerationRunner(
            generation_runner=FakeGenerationRunner(),  # type: ignore[arg-type]
            model_id="smoke_qwen",
            target_grades=["SD6", "UNIVERSITY"],  # type: ignore[list-item]
        )


def test_empty_model_id_rejected():
    with pytest.raises(ValueError):
        ControlledGenerationRunner(
            generation_runner=FakeGenerationRunner(),  # type: ignore[arg-type]
            model_id="  ",
            target_grades=["SD6"],
        )


def test_generation_config_matches_spec():
    config = GenerationConfig.from_yaml("configs/generation.yaml")

    assert config.max_new_tokens == 256
    assert config.do_sample is False
    assert config.use_cache is True
    assert config.prompt_version == "cg_v1"
    assert list(config.target_grades) == ["SD6", "SMP7", "SMP9", "SMA10"]
    assert config.pilot_cases_per_subject == 2
    assert config.full_cases_per_subject == 12
