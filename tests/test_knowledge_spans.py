import math

import pytest
import torch
from transformers import BatchEncoding

from edu_eval.knowledge.schema import KnowledgeQuestion
from edu_eval.knowledge.scorer import KnowledgeScorer


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.padding_side = "right"
        self._ids = {"x": 40, "y": 41, "a": 5, "b": 6, "c": 7}

    def _encode(self, text):
        return [self._ids.get(ch, (ord(ch) % 40) + 10) for ch in text]

    def __call__(self, texts, return_tensors=None, **kwargs):
        single = isinstance(texts, str)

        if single:
            texts = [texts]

        seqs = [self._encode(text) for text in texts]
        width = max(len(seq) for seq in seqs)
        mask = [[1] * len(seq) + [0] * (width - len(seq)) for seq in seqs]
        seqs = [seq + [0] * (width - len(seq)) for seq in seqs]

        return BatchEncoding(
            {
                "input_ids": torch.tensor(seqs),
                "attention_mask": torch.tensor(mask),
            }
        )


class FakeModel(torch.nn.Module):
    def __init__(self, vocab=50):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None, use_cache=True, **kwargs):
        batch, length = input_ids.shape
        logits = (
            torch.arange(self.vocab, dtype=torch.float32)
            .expand(batch, length, self.vocab)
            .clone()
        )

        return type("Outputs", (), {"logits": logits})()


LOG_NORM = math.log(sum(math.exp(k) for k in range(50)))


def _question(choices, answer_index=1, text="xx"):
    return KnowledgeQuestion(
        question_id="q",
        source="test",
        level="SD",
        grade="6",
        subject="IPA",
        subject_group="STEM",
        question=text,
        choices=choices,
        answer_index=answer_index,
    )


def _scorer(span="full"):
    return KnowledgeScorer(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device=torch.device("cpu"),
        span=span,
    )


def test_invalid_span_rejected():
    with pytest.raises(ValueError):
        _scorer(span="top_k")


def test_full_span_scores_all_tokens():
    scores = _scorer("full")._score_batch(["xx"], ["a"], [2])

    assert scores[0]["token_count"] == 2
    assert scores[0]["log_likelihood"] == pytest.approx(45 - 2 * LOG_NORM)
    assert scores[0]["mean_log_likelihood"] == pytest.approx(22.5 - LOG_NORM)


def test_answer_span_scores_choice_only():
    scores = _scorer("answer")._score_batch(["xx"], ["a"], [2])

    assert scores[0]["token_count"] == 1
    assert scores[0]["log_likelihood"] == pytest.approx(5 - LOG_NORM)
    assert scores[0]["mean_log_likelihood"] == pytest.approx(5 - LOG_NORM)


def test_predict_records_span_and_picks_best_answer():
    result = _scorer("answer").predict(_question(["a", "b", "c"], answer_index=2))

    assert result["score_span"] == "answer"
    assert result["predictions"]["mean_log_likelihood"]["predicted_index"] == 2
    assert result["predictions"]["mean_log_likelihood"]["correct"] is True
    assert result["predictions"]["mean_log_likelihood"]["margin"] == pytest.approx(1.0)


def test_batch_matches_single_with_padding():
    one = _question(["a", "b", "c"], answer_index=2, text="xx")
    two = _question(["a", "b", "c"], answer_index=0, text="yyyy")
    scorer = _scorer("answer")

    batched = scorer.predict_many([one, two], batch_size=2)
    singles = [scorer.predict(one), scorer.predict(two)]

    for got, want in zip(batched, singles):
        assert got["choice_scores"] == want["choice_scores"]
        assert got["predictions"] == want["predictions"]
