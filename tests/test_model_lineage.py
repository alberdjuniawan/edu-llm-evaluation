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


def test_trust_remote_code_defaults_off():
    spec = ModelSpec(
        model_id="m",
        model_role="research",
        stage="final",
        source="org/model",
    )

    assert spec.trust_remote_code is False


def test_resolve_dtype_mapping():
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")

    assert ModelLoader.resolve_dtype(cpu, None) == torch.float32
    assert ModelLoader.resolve_dtype(cuda, None) == torch.bfloat16
    assert ModelLoader.resolve_dtype(cuda, "fp16") == torch.float16
    assert ModelLoader.resolve_dtype(cpu, "bf16") == torch.bfloat16


def test_loader_uses_declared_generative_class():
    from unittest.mock import MagicMock, patch

    from edu_eval.models import loader as loader_module

    spec = ModelSpec(
        model_id="m",
        model_role="research",
        stage="final",
        source="org/model",
        trust_remote_code=True,
    )
    fake_model = MagicMock()
    info = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}

    with (
        patch.object(
            loader_module.AutoTokenizer,
            "from_pretrained",
            return_value=MagicMock(),
        ),
        patch.object(
            loader_module.AutoConfig,
            "from_pretrained",
            return_value=MagicMock(architectures=["Qwen2ForCausalLM"]),
        ),
        patch.object(
            loader_module.transformers.Qwen2ForCausalLM,
            "from_pretrained",
            return_value=(fake_model, info),
        ) as model_fn,
    ):
        _, model, device = ModelLoader.load(spec)

    assert model is fake_model
    _, kwargs = model_fn.call_args
    assert kwargs["trust_remote_code"] is True
    assert kwargs["dtype"] == ModelLoader.resolve_dtype(device, None)
    fake_model.eval.assert_called_once_with()


def test_loader_rejects_undeclared_architecture():
    from edu_eval.models import loader as loader_module

    spec = ModelSpec(
        model_id="m",
        model_role="research",
        stage="final",
        source="org/model",
    )

    with pytest.raises(ValueError, match="no architectures"):
        loader_module.ModelLoader.resolve_model_class(spec, [])

    with pytest.raises(TypeError, match="not a generative model"):
        loader_module.ModelLoader.resolve_model_class(spec, ["Qwen2Model"])

    with pytest.raises(ValueError, match="trust_remote_code"):
        loader_module.ModelLoader.resolve_model_class(spec, ["NoSuchClass123"])


def test_loader_rejects_mismatched_weights():
    from edu_eval.models import loader as loader_module

    with pytest.raises(RuntimeError, match="did not match"):
        loader_module.ModelLoader.verify_loading_info(
            {"missing_keys": ["a"], "unexpected_keys": [], "mismatched_keys": []}
        )

    loader_module.ModelLoader.verify_loading_info(
        {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}
    )
