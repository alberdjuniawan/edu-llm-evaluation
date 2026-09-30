"""IndoMMLU letter scoring (the method used by the official evaluation).

The official repo (fajri91/IndoMMLU) reports every leaderboard number with
``evaluate.py --by_letter``: one forward pass over the prompt, then the next-token
logits of the letters A..E are compared (argmax over the first len(options) letters).
This is the primary metric here so results are comparable to the paper/leaderboard.

Only deviation (``style="natural"``, the default): the official code leaves a trailing
space after ``Jawaban:`` and then scores the bare letter token. That was written for
SentencePiece tokenizers; on byte-level BPE (Qwen) the space becomes its own token, so
we end the prompt at ``Jawaban:`` and score `` A`` (space + letter, one token), which is
how BPE tokenizes "Jawaban: A" naturally. ``style="official"`` reproduces the original
byte-for-byte input and is meant for a small sensitivity check.

Batching uses RIGHT padding and reads the logits at each row's last real token. Real
tokens never see padding that comes after them, so this is exact for plain attention and
for recurrent / linear-attention layers alike (left padding would let pad tokens flow
into recurrent state). Full logits are computed; lower batch_size if you hit OOM.
"""

import math
from collections.abc import Sequence
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from edu_eval.generation.template_check import require_thinking_off
from edu_eval.knowledge.indommlu import format_indommlu_prompt
from edu_eval.knowledge.schema import KnowledgeQuestion

LETTERS = ("A", "B", "C", "D", "E")
PROBE = "Jawaban:"
ANSWER_HEADER = "Jawaban: "
STYLES = ("natural", "official")
PROMPT_FORMATS = ("plain", "chat")


def letter_token_ids(tokenizer: PreTrainedTokenizerBase, style: str) -> list[int]:
    """One candidate token id per letter; raises instead of guessing."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}.")

    ids: list[int] = []

    if style == "natural":
        base = tokenizer(PROBE, add_special_tokens=False)["input_ids"]

        for letter in LETTERS:
            full = tokenizer(f"{PROBE} {letter}", add_special_tokens=False)["input_ids"]

            if full[: len(base)] != base or len(full) != len(base) + 1:
                raise ValueError(
                    f"Huruf {letter!r} bukan tepat satu token setelah {PROBE!r}: "
                    f"base={base} full={full}. Tokenizer ini butuh style='official' "
                    "atau penanganan khusus."
                )

            ids.append(int(full[-1]))
    else:
        for letter in LETTERS:
            encoded = tokenizer(letter, add_special_tokens=False)["input_ids"]

            if not encoded:
                raise ValueError(f"Huruf {letter!r} menghasilkan 0 token.")

            ids.append(int(encoded[-1]))

    if len(set(ids)) != len(ids):
        raise ValueError(f"Token huruf tidak unik: {ids}")

    return ids


def build_input_text(
    question: KnowledgeQuestion,
    style: str,
    prompt_format: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
    enable_thinking: bool | None = False,
) -> str:
    prompt = format_indommlu_prompt(question)

    if prompt_format == "plain":
        return prompt if style == "official" else prompt.rstrip(" ")

    if prompt_format == "chat":
        if tokenizer is None:
            raise ValueError("chat format needs the tokenizer.")

        body = prompt.removesuffix(ANSWER_HEADER).rstrip()
        kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

        return rendered + PROBE

    raise ValueError(f"prompt_format must be one of {PROMPT_FORMATS}.")


class LetterScorer:
    score_mode = "letter"

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        style: str = "natural",
        prompt_format: str = "plain",
        enable_thinking: bool | None = False,
    ) -> None:
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}.")

        if prompt_format not in PROMPT_FORMATS:
            raise ValueError(f"prompt_format must be one of {PROMPT_FORMATS}.")

        if prompt_format == "chat" and style != "natural":
            raise ValueError("chat format only supports style='natural'.")

        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id for batched scoring.")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.style = style
        self.prompt_format = prompt_format
        self.enable_thinking = enable_thinking
        self.model.eval()
        self.tokenizer.padding_side = "right"
        self.candidate_ids = letter_token_ids(tokenizer, style)
        self.template_info: dict[str, Any] | None = None

        if prompt_format == "chat":
            self.template_info = require_thinking_off(tokenizer, enable_thinking)

    @property
    def signature(self) -> tuple:
        return ("letter", self.style, self.prompt_format)

    def describe(self) -> dict[str, Any]:
        return {
            "score_mode": "letter",
            "letter_style": self.style,
            "prompt_format": self.prompt_format,
            "letter_token_ids": self.candidate_ids,
            "letter_tokens": self.tokenizer.convert_ids_to_tokens(self.candidate_ids),
            "template_check": self.template_info,
        }

    @torch.inference_mode()
    def _letter_log_probs(self, texts: Sequence[str]) -> list[list[float]]:
        self.tokenizer.padding_side = "right"
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=(self.prompt_format == "plain"),
            return_attention_mask=True,
        ).to(self.device)

        mask = encoded["attention_mask"]

        if not bool(mask[:, 0].all()):
            raise RuntimeError("Padding is not on the right; last-token index invalid.")

        outputs = self.model(
            input_ids=encoded["input_ids"], attention_mask=mask, use_cache=False
        )
        rows = torch.arange(mask.size(0), device=mask.device)
        last = outputs.logits[rows, mask.sum(dim=1) - 1].float()
        del outputs
        log_probs = torch.log_softmax(last, dim=-1)[:, self.candidate_ids]

        return log_probs.cpu().tolist()

    def _build_result(
        self, question: KnowledgeQuestion, row: Sequence[float]
    ) -> dict[str, Any]:
        n_choices = len(question.choices)
        values = list(row[:n_choices])
        peak = max(values)
        normalizer = peak + math.log(sum(math.exp(v - peak) for v in values))
        probs = [math.exp(v - normalizer) for v in values]
        ordered = sorted(range(n_choices), key=lambda i: values[i], reverse=True)
        predicted = ordered[0]

        return {
            "question_id": question.question_id,
            "level": question.level,
            "grade": question.grade,
            "subject": question.subject,
            "gold_index": question.answer_index,
            "score_mode": "letter",
            "letter_style": self.style,
            "prompt_format": self.prompt_format,
            "choice_scores": [
                {
                    "choice_index": i,
                    "letter": LETTERS[i],
                    "log_prob": values[i],
                    "prob": probs[i],
                }
                for i in range(n_choices)
            ],
            "predictions": {
                "letter": {
                    "predicted_index": predicted,
                    "correct": predicted == question.answer_index,
                    "gold_score": values[question.answer_index],
                    "margin": values[ordered[0]] - values[ordered[1]],
                    "letter_mass": math.exp(normalizer),
                }
            },
        }

    def predict_many(
        self, questions: Sequence[KnowledgeQuestion], batch_size: int = 8
    ) -> list[dict[str, Any]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        texts = [
            build_input_text(
                q, self.style, self.prompt_format, self.tokenizer, self.enable_thinking
            )
            for q in questions
        ]
        order = sorted(range(len(questions)), key=lambda i: len(texts[i]))
        results: list[dict[str, Any] | None] = [None] * len(questions)

        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            rows = self._letter_log_probs([texts[i] for i in indices])

            for row, index in zip(rows, indices, strict=True):
                results[index] = self._build_result(questions[index], row)

        return [r for r in results if r is not None]

    def predict(self, question: KnowledgeQuestion) -> dict[str, Any]:
        return self.predict_many([question], batch_size=1)[0]
