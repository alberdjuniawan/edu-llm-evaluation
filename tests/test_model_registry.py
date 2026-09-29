from pathlib import Path

import pytest

from edu_eval.models.registry import ModelRegistry

CONFIG_PATH = Path("configs/models.yaml")


def test_load_model_registry():
    registry = ModelRegistry.from_yaml(CONFIG_PATH)

    assert {model.model_id for model in registry.all()} == {
        "smoke_qwen",
        "qwen35_9b_base",
        "sr02_cpt_final",
        "sr_all",
    }


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


def test_research_lineup_unresolved_until_audit():
    registry = ModelRegistry.from_yaml(CONFIG_PATH)

    for model_id in ("qwen35_9b_base", "sr02_cpt_final", "sr_all"):
        model = registry.get(model_id)

        assert model.model_role == "research"
        assert model.revision is None
        assert model.lineage_status == "unresolved"


def test_cpt_parent_declared():
    registry = ModelRegistry.from_yaml(CONFIG_PATH)
    model = registry.get("sr02_cpt_final")

    assert model.stage == "cpt"
    assert model.parent_model_id == "qwen35_9b_base"
    assert model.lineage_group == "sr-02"


def test_sr_all_lineage_not_invented():
    registry = ModelRegistry.from_yaml(CONFIG_PATH)
    model = registry.get("sr_all")

    assert model.stage == "final"
    assert model.parent_model_id is None
    assert model.lineage_group is None
