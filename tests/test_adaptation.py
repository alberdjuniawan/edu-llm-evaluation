import pytest

from edu_eval.linguistic.adaptation import (
    GRADE_ORDER,
    clean_text,
    features,
    per_case_tau,
    summarize_adaptation,
)


def _rows(values_by_case):
    return [
        {"case_id": c, "target_grade": g, "features": {"words_per_sentence": v}}
        for c, vals in values_by_case.items()
        for g, v in zip(GRADE_ORDER, vals, strict=True)
    ]


def test_clean_text_turns_bullets_into_sentences():
    cleaned = clean_text("## Judul\n- satu\n- dua\n**tiga**")

    assert "#" not in cleaned and "*" not in cleaned
    assert cleaned.count(".") == 4


def test_bullets_do_not_inflate_words_per_sentence():
    text = "\n".join(f"- poin nomor {chr(97 + i)} tentang tumbuhan" for i in range(6))

    assert features(text)["words_per_sentence"] == pytest.approx(5.0)


def test_tau_is_one_for_monotone_and_minus_one_for_reverse():
    taus = per_case_tau(
        _rows({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]}), "words_per_sentence"
    )

    assert taus == {"a": pytest.approx(1.0), "b": pytest.approx(-1.0)}


def test_incomplete_cases_are_skipped():
    rows = _rows({"a": [1, 2, 3, 4]})[:3] + _rows({"b": [1, 2, 3, 4]})

    assert set(per_case_tau(rows, "words_per_sentence")) == {"b"}


def test_summary_counts_and_direction():
    rows = _rows({f"c{i}": [5, 6, 8, 12] for i in range(6)} | {"x": [9, 8, 7, 6]})
    summary = summarize_adaptation(rows, "words_per_sentence")

    assert summary["n_cases"] == 7
    assert summary["cases_positive"] == 6
    assert summary["cases_sma10_gt_sd6"] == 6
    assert summary["mean_sma10_minus_sd6"] > 0


def test_summary_needs_complete_cases():
    with pytest.raises(ValueError):
        summarize_adaptation(_rows({"a": [1, 2, 3, 4]})[:3], "words_per_sentence")
