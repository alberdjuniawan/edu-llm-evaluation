from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ModelRuntimeInfo(BaseModel):
    architecture: str
    model_type: str | None
    architectures: list[str] | None
    parameter_count: int
    dtype: str
    device: str
    context_length: int | None
    vocab_size: int | None
    chat_template_available: bool


def inspect_model(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> ModelRuntimeInfo:
    config = model.config
    parameter = next(model.parameters())

    architectures = getattr(config, "architectures", None)

    if architectures is not None:
        architectures = list(architectures)

    return ModelRuntimeInfo(
        architecture=type(model).__name__,
        model_type=getattr(config, "model_type", None),
        architectures=architectures,
        parameter_count=model.num_parameters(),
        dtype=str(parameter.dtype).replace("torch.", ""),
        device=str(parameter.device),
        context_length=getattr(config, "max_position_embeddings", None),
        vocab_size=getattr(config, "vocab_size", None),
        chat_template_available=tokenizer.chat_template is not None,
    )
