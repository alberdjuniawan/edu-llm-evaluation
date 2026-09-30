import importlib.util
import json
from pathlib import Path

import pytest

from edu_eval.knowledge.schema import KnowledgeQuestion


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_knowledge.py"
    spec = importlib.util.spec_from_file_location("run_knowledge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


rk = _load()


def _q(i):
    return KnowledgeQuestion(
        question_id=str(i),
        source="t",
        level="SD",
        grade="6",
        subject="IPA",
        subject_group="STEM",
        question="q",
        choices=["A. a", "B. b", "C. c"],
        answer_index=0,
    )


class FakeLetterScorer:
    signature = ("letter", "natural", "plain")

    def predict_many(self, questions, batch_size=1):
        return [
            {
                "question_id": q.question_id,
                "gold_index": 0,
                "score_mode": "letter",
                "letter_style": "natural",
                "prompt_format": "plain",
                "choice_scores": [],
                "predictions": {"letter": {"predicted_index": 0, "correct": True}},
            }
            for q in questions
        ]


def test_run_level_writes_then_resumes(tmp_path):
    questions = [_q(i) for i in range(5)]
    path = tmp_path / "primary.jsonl"

    first = rk.run_level(FakeLetterScorer(), questions, "SD+SMP", path, batch_size=2)
    second = rk.run_level(FakeLetterScorer(), questions, "SD+SMP", path, batch_size=2)

    assert first["newly_processed"] == 5 and first["accuracy"] == 1.0
    assert second["newly_processed"] == 0 and second["correct"] == 5


def test_resume_refuses_to_mix_scoring_methods(tmp_path):
    legacy = {
        "question_id": "0",
        "gold_index": 0,
        "score_span": "answer",
        "predictions": {"mean_log_likelihood": {"predicted_index": 0, "correct": True}},
    }
    path = tmp_path / "primary.jsonl"
    path.write_text(json.dumps(legacy) + "\n")

    with pytest.raises(RuntimeError, match="was scored with"):
        rk.read_existing_results(path, [_q(0)], ("letter", "natural", "plain"))


def test_resume_accepts_legacy_records_for_legacy_signature(tmp_path):
    legacy = {
        "question_id": "0",
        "gold_index": 0,
        "score_span": "answer",
        "predictions": {"mean_log_likelihood": {"predicted_index": 0, "correct": True}},
    }
    path = tmp_path / "primary.jsonl"
    path.write_text(json.dumps(legacy) + "\n")

    assert rk.read_existing_results(
        path, [_q(0)], ("mean_log_likelihood", "answer")
    ) == (1, 1)


def test_result_tag():
    assert rk.result_tag(("letter", "natural", "plain"), None) == "letter-natural-plain"
    assert (
        rk.result_tag(("letter", "natural", "plain"), 300)
        == "letter-natural-plain-n300"
    )
    assert (
        rk.result_tag(("mean_log_likelihood", "answer"), None)
        == "mean_log_likelihood-answer"
    )
