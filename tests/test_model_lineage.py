from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from edu_eval.knowledge.indommlu import load_indommlu_csv
from edu_eval.models.loader import ModelLoader
from edu_eval.models.schema import ModelSpec

DATA_PATH = Path("data/raw/IndoMMLU.csv")


def _load():
    if not DATA_PATH.exists():
        pytest.skip("IndoMMLU data absent")

    return load_indommlu_csv(DATA_PATH)


def test_zero_shot_slices_exclude_fewshot():
    questions = _load()
    zero = [q for q in questions if not q.is_for_fewshot]

    assert len(questions) == 14979
    assert len(zero) == 14904

    primary = [q for q in zero if q.level in ("SD", "SMP")]
    retention = [q for q in zero if q.level == "SMA"]

    assert len(primary) == 8118
    assert len(retention) == 4742
    assert all(not q.is_for_fewshot for q in primary + retention)


def test_model_lineage_defaults_unresolved():
    spec = ModelSpec(
        model_id="m",
        model_role="research",
        stage="cpt",
        source="org/model",
    )

    assert spec.lineage_status == "unresolved"
    assert spec.precision is None


def test_model_lineage_verified_accepts_all_fields():
    spec = ModelSpec(
        model_id="m1",
        model_role="research",
        stage="cpt",
        source="org/model",
        revision="abc123",
        lineage_group="sr-02",
        parent_model_id="m0",
        provenance_url="https://example.com/m1",
        lineage_status="verified",
        precision="bf16",
    )

    assert spec.lineage_status == "verified"
    assert spec.precision == "bf16"


def test_invalid_precision_rejected():
    with pytest.raises(ValidationError):
        ModelSpec(
            model_id="m",
            model_role="research",
            stage="cpt",
            source="org/model",
            precision="int4",
        )


def test_resolve_dtype_mapping():
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")

    assert ModelLoader.resolve_dtype(cpu, None) == torch.float32
    assert ModelLoader.resolve_dtype(cuda, None) == torch.bfloat16
    assert ModelLoader.resolve_dtype(cuda, "fp16") == torch.float16
    assert ModelLoader.resolve_dtype(cpu, "bf16") == torch.bfloat16
