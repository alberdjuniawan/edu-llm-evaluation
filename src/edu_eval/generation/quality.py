import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

THINKING_PATTERNS = (
    re.compile(r"</?think>", re.IGNORECASE),
    re.compile(r"^\s*hm+\b", re.IGNORECASE),
    re.compile(r"\bpengguna\s+(meminta|ingin|minta|menginginkan)\b", re.IGNORECASE),
    re.compile(r"\bthe user\b", re.IGNORECASE),
)

META_PREAMBLE = re.compile(
    r"^\s*(baik|oke|okay),?\s+(saya|aku)\s+akan\b", re.IGNORECASE
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "empty": 0.0,
    "thinking_leak": 0.0,
    "dash_spam": 0.0,
    "repetitive": 0.10,
    "truncated": 0.10,
}


def repetition_ratio(text: str, n: int = 4) -> float:
    words = text.split()
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

    if not grams:
        return 0.0

    return 1 - len(set(grams)) / len(grams)


def is_dash_spam(text: str) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if len(lines) < 5:
        return False

    return sum(line in ("-", "--", "*") for line in lines) / len(lines) > 0.5


def output_flags(text: str, hit_max_new_tokens: bool) -> dict[str, bool]:
    return {
        "empty": not text.strip(),
        "truncated": bool(hit_max_new_tokens),
        "thinking_leak": any(p.search(text) for p in THINKING_PATTERNS),
        "dash_spam": is_dash_spam(text),
        "repetitive": repetition_ratio(text) > 0.30,
        "meta_preamble": bool(META_PREAMBLE.search(text)),  # informational only
    }


def flag_rates(results: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not results:
        raise ValueError("results must not be empty.")

    all_flags = [
        output_flags(r["output_text"], r["hit_max_new_tokens"]) for r in results
    ]

    return {
        key: sum(flags[key] for flags in all_flags) / len(all_flags)
        for key in all_flags[0]
    }


def evaluate_preflight(
    results: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float] | None = None,
) -> tuple[bool, dict[str, float], list[str]]:
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rates = flag_rates(results)
    reasons = [
        f"{key}={rates[key]:.0%} > batas {limit:.0%}"
        for key, limit in limits.items()
        if rates[key] > limit
    ]

    return (not reasons, rates, reasons)


def run_preflight(
    generate: Callable[[Any, str], Mapping[str, Any]],
    cases: Sequence[Any],
    grades: Sequence[str],
    done: set[tuple[str, str]],
    thresholds: Mapping[str, float] | None = None,
) -> tuple[list[Mapping[str, Any]], bool, dict[str, float], list[str]]:
    """Generate a few outputs and judge them BEFORE committing to the full run.

    Nothing is written here; the caller persists the results only when ok.
    """
    results = [
        generate(case, grade)
        for case in cases
        for grade in grades
        if (case.case_id, grade) not in done
    ]

    if not results:
        return [], True, {}, []

    ok, rates, reasons = evaluate_preflight(results, thresholds)

    return results, ok, rates, reasons
