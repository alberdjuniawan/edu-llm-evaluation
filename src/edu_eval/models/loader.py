import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from edu_eval.models.schema import ModelSpec


class ModelLoader:
    @staticmethod
    def resolve_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    @classmethod
    def load(cls, spec: ModelSpec):
        device = cls.resolve_device()

        tokenizer = AutoTokenizer.from_pretrained(
            spec.source,
            revision=spec.revision,
        )

        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
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
