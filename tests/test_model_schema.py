import pytest
from pydantic import ValidationError

from edu_eval.models.schema import ModelSpec


def test_valid_research_model():
    model = ModelSpec(
        model_id="sft_v2",
        model_role="research",
        stage="sft_v2",
        source="example/sft-v2",
        architecture="Qwen3_5ForConditionalGeneration",
        parameter_count=9_000_000_000,
        dtype="bfloat16",
        context_length=262_144,
        chat_template_available=True,
        documentation_status="partial",
    )

    assert model.model_id == "sft_v2"
    assert model.model_role == "research"
    assert model.stage == "sft_v2"
    assert model.parameter_count == 9_000_000_000


def test_valid_smoke_model():
    model = ModelSpec(
        model_id="smoke_qwen",
        model_role="smoke_test",
        stage="smoke",
        source="Qwen/Qwen2.5-0.5B-Instruct",
    )

    assert model.model_role == "smoke_test"
    assert model.stage == "smoke"


def test_invalid_stage():
    payload = {
        "model_id": "bad",
        "model_role": "research",
        "stage": "something_else",
        "source": "example/model",
    }

    with pytest.raises(ValidationError):
        ModelSpec.model_validate(payload)


def test_invalid_parameter_count():
    payload = {
        "model_id": "bad",
        "model_role": "research",
        "stage": "pretrained",
        "source": "example/model",
        "parameter_count": -1,
    }

    with pytest.raises(ValidationError):
        ModelSpec.model_validate(payload)


def test_base_and_final_stages_valid():
    for stage in ("base", "final"):
        model = ModelSpec(
            model_id="m",
            model_role="research",
            stage=stage,
            source="org/model",
        )

        assert model.stage == stage
