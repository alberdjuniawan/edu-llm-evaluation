"""Does a model adapt text complexity to the target grade? (same case, 4 grades)"""

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from edu_eval.linguistic.analyzer import analyze_text, tokenize_words
from edu_eval.statistics.metrics import bootstrap_ci, kendall_tau_b

GRADE_ORDER = ("SD6", "SMP7", "SMP9", "SMA10")
METRICS = (
    "words_per_sentence",
    "avg_word_length_chars",
    "lexical_density",
    "fkgl",
    "ari",
    "word_count",
    "ttr_first_n",
)

_MARKDOWN_RE = re.compile(r"[#*_`>|]+")
_BULLET_RE = re.compile(r"^\s*(?:[-+]|\d+[.)])\s+", re.MULTILINE)


def clean_text(text: str) -> str:
    """Strip markdown and make bullets/lines count as sentences.

    The analyzer only splits sentences on . ! ? so bullet lists without punctuation
    would look like one giant sentence and inflate words-per-sentence.
    """
    text = _BULLET_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    lines = [line if line[-1] in ".!?…" else line + "." for line in lines]

    return " ".join(lines)


def features(text: str, n_words: int = 150) -> dict[str, float]:
    cleaned = clean_text(text)
    base = analyze_text(cleaned)
    words = tokenize_words(cleaned)[:n_words]

    return {
        "words_per_sentence": base["words_per_sentence"],
        "avg_word_length_chars": base["avg_word_length_chars"],
        "lexical_density": base["lexical_density"],
        "fkgl": base["fkgl"],
        "ari": base["ari"],
        "word_count": float(base["word_count"]),
        "ttr_first_n": len(set(words)) / len(words) if words else 0.0,
    }


def per_case_tau(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    """Kendall tau-b between grade order and the metric, one value per case."""
    by_case: dict[str, dict[str, float]] = defaultdict(dict)

    for row in rows:
        by_case[row["case_id"]][row["target_grade"]] = row["features"][metric]

    taus: dict[str, float] = {}

    for case_id, values in by_case.items():
        if all(grade in values for grade in GRADE_ORDER):
            taus[case_id] = kendall_tau_b(
                list(range(len(GRADE_ORDER))), [values[g] for g in GRADE_ORDER]
            )

    return taus


def summarize_adaptation(
    rows: Sequence[Mapping[str, Any]], metric: str, seed: int = 42
) -> dict[str, Any]:
    taus = per_case_tau(rows, metric)

    if not taus:
        raise ValueError("No case has all four grades.")

    values = list(taus.values())
    by_case: dict[str, dict[str, float]] = defaultdict(dict)

    for row in rows:
        by_case[row["case_id"]][row["target_grade"]] = row["features"][metric]

    diffs = [by_case[c]["SMA10"] - by_case[c]["SD6"] for c in taus]

    return {
        "metric": metric,
        "n_cases": len(values),
        "mean_tau": sum(values) / len(values),
        "tau_ci": bootstrap_ci(values, seed=seed),
        "cases_positive": sum(v > 0 for v in values),
        "mean_sma10_minus_sd6": sum(diffs) / len(diffs),
        "sma10_minus_sd6_ci": bootstrap_ci(diffs, seed=seed),
        "cases_sma10_gt_sd6": sum(d > 0 for d in diffs),
    }
