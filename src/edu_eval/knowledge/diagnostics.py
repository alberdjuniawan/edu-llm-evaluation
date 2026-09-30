from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

ENTRY_KEYS = ("letter", "mean_log_likelihood", "official")


def prediction_entry(record: Mapping[str, Any]) -> Mapping[str, Any]:
    predictions = record["predictions"]

    for key in ENTRY_KEYS:
        if key in predictions:
            return predictions[key]

    raise KeyError(f"No known prediction key in {list(predictions)}.")


def correctness(record: Mapping[str, Any]) -> bool:
    return bool(prediction_entry(record)["correct"])


def position_bias_report(records: list[dict]) -> dict:
    predicted_counts: dict[int, int] = {}
    gold_counts: dict[int, int] = {}
    gold_correct: dict[int, int] = {}
    gold_total: dict[int, int] = {}

    for record in records:
        predicted = int(prediction_entry(record)["predicted_index"])
        gold = int(record["gold_index"])
        predicted_counts[predicted] = predicted_counts.get(predicted, 0) + 1
        gold_counts[gold] = gold_counts.get(gold, 0) + 1
        gold_total[gold] = gold_total.get(gold, 0) + 1

        if correctness(record):
            gold_correct[gold] = gold_correct.get(gold, 0) + 1

    return {
        "n": len(records),
        "predicted_counts": predicted_counts,
        "gold_counts": gold_counts,
        "accuracy_by_gold_position": {
            position: gold_correct.get(position, 0) / total
            for position, total in gold_total.items()
        },
    }


def chance_baseline(records: Sequence[Mapping[str, Any]]) -> float:
    if not records:
        raise ValueError("records must not be empty.")

    return sum(1 / len(r["choice_scores"]) for r in records) / len(records)


def _score(choice: Mapping[str, Any]) -> float:
    for key in ("log_prob", "mean_log_likelihood"):
        if key in choice:
            return float(choice[key])

    raise KeyError(f"No score key in {list(choice)}.")


def position_debiased_accuracy(records: Sequence[Mapping[str, Any]]) -> float:
    """Label-free calibration: subtract each position's mean score, then re-argmax.

    Diagnostic only. If it barely moves accuracy, the model has real signal beyond a
    positional prior; if it jumps, the raw number is inflated/deflated by position bias.
    """
    sums: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int], int] = defaultdict(int)

    for record in records:
        n = len(record["choice_scores"])

        for choice in record["choice_scores"]:
            key = (n, int(choice["choice_index"]))
            sums[key] += _score(choice)
            counts[key] += 1

    correct = 0

    for record in records:
        n = len(record["choice_scores"])
        adjusted = [
            _score(c)
            - sums[(n, int(c["choice_index"]))] / counts[(n, int(c["choice_index"]))]
            for c in record["choice_scores"]
        ]
        predicted = max(range(n), key=lambda i: adjusted[i])
        correct += predicted == int(record["gold_index"])

    return correct / len(records)
