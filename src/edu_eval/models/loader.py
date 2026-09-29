from typing import Any, cast

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.generation.utils import GenerationMixin

from edu_eval.models.schema import ModelSpec, Precision

_PRECISION_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


class ModelLoader:
    @staticmethod
    def resolve_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    @staticmethod
    def resolve_dtype(device: torch.device, precision: Precision | None) -> torch.dtype:
        if precision is not None:
            return _PRECISION_DTYPES[precision]

        return torch.bfloat16 if device.type == "cuda" else torch.float32

    @staticmethod
    def resolve_model_class(
        spec: ModelSpec, architectures: list[str] | None
    ) -> type[PreTrainedModel]:
        if not architectures:
            raise ValueError(f"Checkpoint {spec.source!r} declares no architectures.")

        name = architectures[0]
        native = getattr(transformers, name, None)

        if isinstance(native, type) and issubclass(native, GenerationMixin):
            if not issubclass(native, PreTrainedModel):
                raise TypeError(
                    f"Declared architecture {name!r} is not a transformers model."
                )

            return cast("type[PreTrainedModel]", native)

        if isinstance(native, type):
            raise TypeError(
                f"Declared architecture {name!r} is not a generative model."
            )

        if not spec.trust_remote_code:
            raise ValueError(
                f"Architecture {name!r} is not natively supported; "
                "enable trust_remote_code consciously to load remote code."
            )

        remote = get_class_from_dynamic_module(
            name, spec.source, revision=spec.revision
        )

        if isinstance(remote, type) and issubclass(remote, GenerationMixin):
            if not issubclass(remote, PreTrainedModel):
                raise TypeError(
                    f"Remote architecture {name!r} is not a transformers model."
                )

            return cast("type[PreTrainedModel]", remote)

        raise TypeError(f"Remote architecture {name!r} is not a generative model.")

    @staticmethod
    def verify_loading_info(loading_info: dict) -> None:
        missing = loading_info.get("missing_keys", [])
        unexpected = loading_info.get("unexpected_keys", [])
        mismatched = loading_info.get("mismatched_keys", [])

        if missing or unexpected or mismatched:
            raise RuntimeError(
                "Checkpoint weights did not match the model class "
                f"(missing={missing}, unexpected={unexpected}, "
                f"mismatched={mismatched}). Stopping instead of "
                "evaluating randomly initialized weights."
            )

    @classmethod
    def load(cls, spec: ModelSpec):
        device = cls.resolve_device()

        tokenizer = AutoTokenizer.from_pretrained(
            spec.source,
            revision=spec.revision,
        )

        config = AutoConfig.from_pretrained(
            spec.source,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
        )
        model_class = cls.resolve_model_class(
            spec, list(getattr(config, "architectures", None) or [])
        )

        dtype = cls.resolve_dtype(device, spec.precision)
        device_map: str | dict[str, str] = (
            "auto" if device.type == "cuda" else {"": str(device)}
        )

        model: PreTrainedModel
        loading_info: dict
        loaded: Any = model_class.from_pretrained(
            spec.source,
            revision=spec.revision or "main",
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=spec.trust_remote_code,
            output_loading_info=True,
        )
        model, loading_info = loaded
        cls.verify_loading_info(loading_info)

        model.eval()

        return tokenizer, model, device
