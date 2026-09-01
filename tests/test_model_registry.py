from pathlib import Path

import pytest

from edu_eval.models.registry import ModelRegistry

CONFIG_PATH = Path("configs/models.yaml")


def test_load_model_registry():
    registry = ModelRegistry.from_yaml(CONFIG_PATH)

    assert len(registry.all()) == 1


def test_get_smoke_model():
    registry = ModelRegistry.from_yaml(CONFIG_PATH)

    model = registry.get("smoke_qwen")

    assert model.model_role == "smoke_test"
    assert model.stage == "smoke"
    assert model.source == "Qwen/Qwen2.5-0.5B-Instruct"


def test_unknown_model_raises():
    registry = ModelRegistry.from_yaml(CONFIG_PATH)

    with pytest.raises(KeyError):
        registry.get("does_not_exist")
