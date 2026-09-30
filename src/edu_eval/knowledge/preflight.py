"""Cheap go/no-go checks run BEFORE a long knowledge run, so a broken setup never
burns a full benchmark worth of compute (results here are discarded, not written)."""

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from edu_eval.knowledge.diagnostics import prediction_entry

MAX_SINGLE_LETTER_SHARE = 0.70
MIN_LETTER_MASS_FAIL = 0.05
MIN_LETTER_MASS_WARN = 0.30
CHANCE_MARGIN_WARN = 0.03
MAX_BATCH_LOGPROB_DIFF = 0.5


def evaluate_knowledge_preflight(
    results: Sequence[Mapping[str, Any]],
) -> tuple[bool, dict[str, Any], list[str], list[str]]:
    if not results:
        raise ValueError("results must not be empty.")

    entries = [prediction_entry(r) for r in results]
    n = len(results)
    counts = Counter(int(e["predicted_index"]) for e in entries)
    max_share = max(counts.values()) / n
    accuracy = sum(bool(e["correct"]) for e in entries) / n
    chance = sum(1 / len(r["choice_scores"]) for r in results) / n
    masses = [e["letter_mass"] for e in entries if "letter_mass" in e]
    median_mass = statistics.median(masses) if masses else None
    failures: list[str] = []
    warnings: list[str] = []

    if max_share > MAX_SINGLE_LETTER_SHARE:
        failures.append(
            f"satu posisi jawaban dipilih di {max_share:.0%} soal "
            f"(batas {MAX_SINGLE_LETTER_SHARE:.0%}): prompt/tokenisasi kemungkinan rusak"
        )

    if median_mass is not None and median_mass < MIN_LETTER_MASS_FAIL:
        failures.append(
            f"probabilitas huruf A-E hanya {median_mass:.1%} (median): model tidak "
            "memprediksi huruf di posisi itu. Coba --prompt-format chat"
        )
    elif median_mass is not None and median_mass < MIN_LETTER_MASS_WARN:
        warnings.append(f"probabilitas huruf A-E rendah ({median_mass:.1%} median)")

    if accuracy < chance + CHANCE_MARGIN_WARN:
        warnings.append(
            f"akurasi {accuracy:.1%} tidak jauh di atas chance {chance:.1%} "
            f"(n={n}, noise besar bila n kecil)"
        )

    stats = {
        "n": n,
        "accuracy": accuracy,
        "chance": chance,
        "predicted_counts": dict(sorted(counts.items())),
        "max_single_position_share": max_share,
        "median_letter_mass": median_mass,
    }

    return (not failures, stats, failures, warnings)


def check_batch_equivalence(
    scorer: Any,
    questions: Sequence[Any],
    tolerance: float = MAX_BATCH_LOGPROB_DIFF,
) -> tuple[bool, float]:
    """Batched (padded) scoring must agree with one-at-a-time scoring."""
    batched = scorer.predict_many(questions, batch_size=len(questions))
    single = scorer.predict_many(questions, batch_size=1)
    worst = 0.0

    for first, second in zip(batched, single, strict=True):
        for a, b in zip(first["choice_scores"], second["choice_scores"], strict=True):
            worst = max(worst, abs(a["log_prob"] - b["log_prob"]))

    return worst <= tolerance, worst
