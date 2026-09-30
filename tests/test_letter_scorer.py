import math

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from edu_eval.knowledge.diagnostics import (
    chance_baseline,
    correctness,
    position_bias_report,
    position_debiased_accuracy,
)
from edu_eval.knowledge.indommlu import format_indommlu_prompt
from edu_eval.knowledge.letter_scorer import (
    LETTERS,
    LetterScorer,
    build_input_text,
    letter_token_ids,
)
from edu_eval.knowledge.preflight import (
    check_batch_equivalence,
    evaluate_knowledge_preflight,
)
from edu_eval.knowledge.schema import KnowledgeQuestion

CORPUS = [
    "Ini adalah soal IPA untuk 6 SD. Pilihlah salah satu jawaban yang dianggap benar!",
    "A. tumbuhan\nB. hewan\nC. batu\nD. air\nE. udara",
    "Jawaban: A",
    "Jawaban: B",
    "Jawaban: C",
    "Jawaban: D",
    "Jawaban: E",
    "Jawaban: A. tumbuhan",
    " A B C D E",
] * 40


def _make_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer)

    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>")


def _question(qid: str, question: str, n_choices: int = 4, answer: int = 0):
    choices = [f"{LETTERS[i]}. opsi {i} untuk {qid}" for i in range(n_choices)]

    return KnowledgeQuestion(
        question_id=qid,
        source="t",
        level="SD",
        grade="6",
        subject="IPA",
        subject_group="STEM",
        question=question,
        choices=choices,
        answer_index=answer,
    )


@pytest.fixture(scope="module")
def rig():
    torch.manual_seed(0)
    tokenizer = _make_tokenizer()
    config = GPT2Config(
        vocab_size=len(tokenizer), n_layer=2, n_head=2, n_embd=32, n_positions=512
    )
    model = GPT2LMHeadModel(config).eval()

    return tokenizer, model


QUESTIONS = [
    _question("1", "Apa itu fotosintesis?", 4, 1),
    _question(
        "2",
        "Sebutkan benda hidup yang bernapas dan tumbuh besar setiap hari di kebun?",
        5,
        3,
    ),
    _question("3", "Air?", 3, 2),
    _question(
        "4",
        "Manakah yang termasuk tumbuhan berbiji terbuka dan hidup di daerah dingin sekali?",
        4,
        0,
    ),
]


def test_natural_letter_ids_are_single_space_letter_tokens(rig):
    tokenizer, _ = rig
    ids = letter_token_ids(tokenizer, "natural")
    tokens = tokenizer.convert_ids_to_tokens(ids)

    assert len(set(ids)) == 5
    assert [t.lstrip("Ġ") for t in tokens] == list(LETTERS)
    assert all(t.startswith("Ġ") for t in tokens)


def test_natural_ids_match_how_the_text_really_tokenizes(rig):
    tokenizer, _ = rig
    ids = letter_token_ids(tokenizer, "natural")

    for letter, expected in zip(LETTERS, ids, strict=True):
        tokens = tokenizer(f"x Jawaban: {letter}", add_special_tokens=False)[
            "input_ids"
        ]
        assert tokens[-1] == expected


def test_prompt_text_matches_official_up_to_the_trailing_space():
    q = QUESTIONS[0]
    official = format_indommlu_prompt(q)

    assert official.endswith("Jawaban: ")
    assert build_input_text(q, "official", "plain") == official
    assert build_input_text(q, "natural", "plain") == official.rstrip(" ")
    assert build_input_text(q, "natural", "plain").endswith("Jawaban:")


def test_batched_equals_single_with_right_padding(rig):
    tokenizer, model = rig
    scorer = LetterScorer(model, tokenizer, torch.device("cpu"))
    ok, worst = check_batch_equivalence(scorer, QUESTIONS, tolerance=1e-4)

    assert ok, f"batched vs single differ by {worst}"
    assert scorer.tokenizer.padding_side == "right"


def test_result_schema_and_normalisation(rig):
    tokenizer, model = rig
    scorer = LetterScorer(model, tokenizer, torch.device("cpu"))
    results = scorer.predict_many(QUESTIONS, batch_size=2)

    assert [r["question_id"] for r in results] == ["1", "2", "3", "4"]
    assert [len(r["choice_scores"]) for r in results] == [4, 5, 3, 4]

    for r, q in zip(results, QUESTIONS, strict=True):
        assert math.isclose(
            sum(c["prob"] for c in r["choice_scores"]), 1.0, rel_tol=1e-6
        )
        entry = r["predictions"]["letter"]
        assert entry["correct"] == (entry["predicted_index"] == q.answer_index)
        assert 0.0 < entry["letter_mass"] <= 1.0
        assert correctness(r) == entry["correct"]
        assert r["score_mode"] == "letter"


def test_padding_uses_only_valid_letters(rig):
    tokenizer, model = rig
    scorer = LetterScorer(model, tokenizer, torch.device("cpu"))
    three_choice = scorer.predict_many([QUESTIONS[2]], batch_size=1)[0]

    assert three_choice["predictions"]["letter"]["predicted_index"] in (0, 1, 2)


def test_left_padding_is_rejected_by_the_guard(rig):
    tokenizer, model = rig
    scorer = LetterScorer(model, tokenizer, torch.device("cpu"))

    class LeftPadder:
        """Tokenizer proxy that forces left padding, as if the setting were flipped."""

        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __call__(self, *args, **kwargs):
            self.inner.padding_side = "left"

            return self.inner(*args, **kwargs)

    scorer.tokenizer = LeftPadder(tokenizer)

    with pytest.raises(RuntimeError):
        scorer._letter_log_probs(["a much longer prompt Jawaban:", "Jawaban:"])

    tokenizer.padding_side = "right"


def test_padded_batch_matches_single_for_unequal_lengths(rig):
    tokenizer, model = rig
    scorer = LetterScorer(model, tokenizer, torch.device("cpu"))
    short, long = QUESTIONS[2], QUESTIONS[3]
    together = scorer.predict_many([short, long], batch_size=2)
    alone = [scorer.predict_many([q], batch_size=1)[0] for q in (short, long)]

    for a, b in zip(together, alone, strict=True):
        for x, y in zip(a["choice_scores"], b["choice_scores"], strict=True):
            assert x["log_prob"] == pytest.approx(y["log_prob"], abs=1e-4)


def test_chat_format_disables_thinking_and_ends_at_probe():
    class ChatTok:
        chat_template = "{% if enable_thinking is defined %}x{% endif %}"
        pad_token_id = 0

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
            think = (
                "<think>\n\n</think>\n\n"
                if kw.get("enable_thinking") is False
                else "<think>\n"
            )
            return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n{think}"

    text = build_input_text(
        QUESTIONS[0], "natural", "chat", ChatTok(), enable_thinking=False
    )

    assert text.endswith("</think>\n\nJawaban:")
    assert "Jawaban: " not in text.split("assistant")[0]


def test_chat_format_rejects_open_think_block():
    from edu_eval.generation.template_check import require_thinking_off

    class Bad:
        chat_template = ""

        def apply_chat_template(self, *a, **k):
            return "<|im_start|>assistant\n<think>\n"

    with pytest.raises(RuntimeError):
        require_thinking_off(Bad(), False)


def _fake_result(pred, gold, mass=0.9, n=4):
    return {
        "question_id": "q",
        "gold_index": gold,
        "choice_scores": [
            {"choice_index": i, "log_prob": -float(i == pred)} for i in range(n)
        ],
        "predictions": {
            "letter": {
                "predicted_index": pred,
                "correct": pred == gold,
                "letter_mass": mass,
            }
        },
    }


def test_preflight_fails_on_constant_answer():
    results = [_fake_result(0, i % 4) for i in range(40)]
    ok, _, failures, _ = evaluate_knowledge_preflight(results)

    assert not ok and any("dipilih" in f for f in failures)


def test_preflight_fails_when_no_probability_on_letters():
    results = [_fake_result(i % 4, i % 4, mass=0.001) for i in range(40)]
    ok, _, failures, _ = evaluate_knowledge_preflight(results)

    assert not ok and any("huruf" in f for f in failures)


def test_preflight_passes_and_warns_at_chance():
    results = [_fake_result(i % 4, (i + 1) % 4) for i in range(40)]
    ok, stats, failures, warnings = evaluate_knowledge_preflight(results)

    assert ok and failures == [] and warnings
    assert stats["chance"] == pytest.approx(0.25)


def test_chance_and_debias():
    results = [_fake_result(i % 4, i % 4) for i in range(40)]

    assert chance_baseline(results) == pytest.approx(0.25)
    assert 0.0 <= position_debiased_accuracy(results) <= 1.0
    assert position_bias_report(results)["n"] == 40
