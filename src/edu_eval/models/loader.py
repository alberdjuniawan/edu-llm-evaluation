import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

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

    @classmethod
    def load(cls, spec: ModelSpec):
        device = cls.resolve_device()

        tokenizer = AutoTokenizer.from_pretrained(
            spec.source,
            revision=spec.revision,
        )

        dtype = cls.resolve_dtype(device, spec.precision)
        device_map: str | dict[str, str] = (
            "auto" if device.type == "cuda" else {"": str(device)}
        )

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            spec.source,
            revision=spec.revision,
            dtype=dtype,
            device_map=device_map,
        )

        model.eval()

        return tokenizer, model, device
