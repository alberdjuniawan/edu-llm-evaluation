from typing import TypedDict

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from edu_eval.knowledge.indommlu import format_indommlu_prompt
from edu_eval.knowledge.schema import KnowledgeQuestion

VOCAB_CHUNK_SIZE = 16_384


class ChoiceScore(TypedDict):
    choice_index: int
    log_likelihood: float
    mean_log_likelihood: float
    token_count: int


class ModePrediction(TypedDict):
    predicted_index: int
    correct: bool
    gold_score: float
    margin: float


class KnowledgePrediction(TypedDict):
    question_id: str
    level: str
    grade: str
    subject: str
    gold_index: int
    score_mode: str
    score_span: str
    choice_scores: list[ChoiceScore]
    predictions: dict[str, ModePrediction]


class KnowledgeScorer:
    score_mode = "mean_log_likelihood"

    @property
    def signature(self) -> tuple:
        return ("mean_log_likelihood", self.span)

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        span: str = "full",
    ) -> None:
        if span not in ("full", "answer"):
            raise ValueError("span must be 'full' or 'answer'.")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.span = span
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id for batched scoring.")

        self.tokenizer.padding_side = "right"

    @torch.inference_mode()
    def _score_batch(
        self, prompts: list[str], choices: list[str], prompt_lens: list[int]
    ) -> list[ChoiceScore]:
        if len(prompts) != len(choices):
            raise ValueError("Prompts and choices must have equal length.")

        if len(prompts) != len(prompt_lens):
            raise ValueError("Prompts and prompt_lens must have equal length.")

        full_texts = [
            prompt + choice for prompt, choice in zip(prompts, choices, strict=True)
        ]

        encoded = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        ).to(self.device)

        outputs = self.model(**encoded, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        del outputs

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        target_ids = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:].bool()
        target_logits = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).float()

        logsumexp = None

        for start in range(0, logits.size(-1), VOCAB_CHUNK_SIZE):
            end = min(start + VOCAB_CHUNK_SIZE, logits.size(-1))
            chunk = logits[:, :, start:end].float()
            chunk_logsumexp = torch.logsumexp(chunk, dim=-1)

            if logsumexp is None:
                logsumexp = chunk_logsumexp
            else:
                logsumexp = torch.logaddexp(logsumexp, chunk_logsumexp)

        token_log_probs = target_logits - logsumexp

        if self.span == "answer":
            positions = torch.arange(
                token_log_probs.size(1), device=token_log_probs.device
            ).unsqueeze(0)
            starts = (
                torch.tensor(prompt_lens, device=token_log_probs.device).unsqueeze(1)
                - 1
            )
            target_mask = target_mask & (positions >= starts)

        token_log_probs = token_log_probs.masked_fill(~target_mask, 0.0)

        token_counts = target_mask.sum(dim=1).clamp_min(1)
        total_log_likelihood = token_log_probs.sum(dim=1)
        mean_log_likelihood = total_log_likelihood / token_counts

        return [
            {
                "choice_index": index,
                "log_likelihood": total_log_likelihood[index].item(),
                "mean_log_likelihood": mean_log_likelihood[index].item(),
                "token_count": int(token_counts[index].item()),
            }
            for index in range(len(full_texts))
        ]

    @staticmethod
    def _predict_from_scores(
        scores: list[ChoiceScore], gold_index: int
    ) -> ModePrediction:
        ordered = sorted(
            scores,
            key=lambda score: score["mean_log_likelihood"],
            reverse=True,
        )
        predicted_index = ordered[0]["choice_index"]
        gold_score = next(
            score["mean_log_likelihood"]
            for score in scores
            if score["choice_index"] == gold_index
        )
        margin = ordered[0]["mean_log_likelihood"] - ordered[1]["mean_log_likelihood"]

        return {
            "predicted_index": predicted_index,
            "correct": predicted_index == gold_index,
            "gold_score": gold_score,
            "margin": margin,
        }

    def _prompt_len(self, prompt: str) -> int:
        encoded = self.tokenizer(prompt, return_tensors="pt")

        return int(encoded["input_ids"].shape[1])

    def predict(self, question: KnowledgeQuestion) -> KnowledgePrediction:
        prompt = format_indommlu_prompt(question)
        prompt_len = self._prompt_len(prompt)
        scores = self._score_batch(
            prompts=[prompt] * len(question.choices),
            choices=question.choices,
            prompt_lens=[prompt_len] * len(question.choices),
        )
        prediction = self._predict_from_scores(scores, question.answer_index)

        return {
            "question_id": question.question_id,
            "level": question.level,
            "grade": question.grade,
            "subject": question.subject,
            "gold_index": question.answer_index,
            "score_mode": "mean_log_likelihood",
            "score_span": self.span,
            "choice_scores": scores,
            "predictions": {"mean_log_likelihood": prediction},
        }

    def predict_many(
        self, questions: list[KnowledgeQuestion], batch_size: int = 1
    ) -> list[KnowledgePrediction]:
        results: list[KnowledgePrediction] = []

        for start in range(0, len(questions), batch_size):
            batch = questions[start : start + batch_size]
            flattened_prompts: list[str] = []
            flattened_choices: list[str] = []
            flattened_lens: list[int] = []
            boundaries: list[tuple[int, int]] = []

            for question in batch:
                prompt = format_indommlu_prompt(question)
                begin = len(flattened_prompts)
                flattened_prompts.extend([prompt] * len(question.choices))
                flattened_choices.extend(question.choices)
                flattened_lens.extend(
                    [self._prompt_len(prompt)] * len(question.choices)
                )
                end = len(flattened_prompts)
                boundaries.append((begin, end))

            flattened_scores = self._score_batch(
                prompts=flattened_prompts,
                choices=flattened_choices,
                prompt_lens=flattened_lens,
            )

            for question, (begin, end) in zip(batch, boundaries, strict=True):
                scores = [
                    ChoiceScore(
                        choice_index=index,
                        log_likelihood=score["log_likelihood"],
                        mean_log_likelihood=score["mean_log_likelihood"],
                        token_count=score["token_count"],
                    )
                    for index, score in enumerate(flattened_scores[begin:end])
                ]
                prediction = self._predict_from_scores(scores, question.answer_index)
                results.append(
                    {
                        "question_id": question.question_id,
                        "level": question.level,
                        "grade": question.grade,
                        "subject": question.subject,
                        "gold_index": question.answer_index,
                        "score_mode": "mean_log_likelihood",
                        "score_span": self.span,
                        "choice_scores": scores,
                        "predictions": {"mean_log_likelihood": prediction},
                    }
                )

        return results
