import time
from collections.abc import Callable
from typing import Protocol, TypedDict

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.generation.utils import GenerateOutput

from edu_eval.generation.config import GenerationConfig


class GenerativeModel(Protocol):
    generate: Callable[..., torch.LongTensor | GenerateOutput]


class GenerationResult(TypedDict):
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    hit_max_new_tokens: bool


class GenerationRunner:
    def __init__(
        self,
        model: GenerativeModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        config: GenerationConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

    def generate(self, user_prompt: str) -> GenerationResult:
        if not user_prompt.strip():
            raise ValueError("user_prompt must not be empty.")

        messages = [{"role": "user", "content": user_prompt}]
        template_kwargs: dict[str, bool] = {}

        if self.config.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.config.enable_thinking

        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **template_kwargs,
        )

        if not isinstance(encoded, BatchEncoding):
            raise TypeError(
                "apply_chat_template must return a BatchEncoding when called "
                "with tokenize=True + return_tensors='pt'."
            )

        inputs: BatchEncoding = encoded
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "GenerationRunner expects a single prompt "
                f"(input_ids shape [1, seq_len]), got {tuple(input_ids.shape)}."
            )

        input_tokens = input_ids.shape[1]
        start = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                use_cache=self.config.use_cache,
                return_dict_in_generate=False,
            )

        latency = time.perf_counter() - start
        sequences = (
            outputs.sequences if isinstance(outputs, GenerateOutput) else outputs
        )

        if not isinstance(sequences, torch.Tensor):
            raise TypeError(
                "model.generate() must return a Tensor or GenerateOutput, "
                f"got {type(outputs).__name__}."
            )

        generated_tokens = sequences[0, input_tokens:]
        output_tokens = generated_tokens.shape[0]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if not isinstance(decoded, str):
            raise TypeError(
                f"tokenizer.decode must return str, got {type(decoded).__name__}."
            )

        return {
            "text": decoded.lstrip(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_seconds": latency,
            "hit_max_new_tokens": output_tokens >= self.config.max_new_tokens,
        }
