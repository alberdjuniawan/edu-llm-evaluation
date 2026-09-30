import argparse
import json
from collections import defaultdict
from pathlib import Path

from edu_eval.knowledge.diagnostics import (
    chance_baseline,
    correctness,
    position_bias_report,
    position_debiased_accuracy,
    prediction_entry,
)
from edu_eval.statistics.metrics import bootstrap_ci, cohens_h, paired_bootstrap_ci

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "knowledge"


def load_predictions(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}.") from exc

    return records


def mean_margin(records: list[dict]) -> float:
    margins = [prediction_entry(record).get("margin") for record in records]
    margins = [margin for margin in margins if isinstance(margin, (int, float))]

    return sum(margins) / len(margins) if margins else 0.0


def summarize(records: list[dict], label: str) -> None:
    flags = [1.0 if correctness(record) else 0.0 for record in records]
    low, high = bootstrap_ci(flags, seed=42)
    print(
        f"{label}: n={len(flags)} acc={sum(flags) / len(flags):.4f} "
        f"[{low:.4f}, {high:.4f}] margin={mean_margin(records):.3f}"
    )


def summarize_slice(records: list[dict], label: str, by_level: bool = False) -> None:
    print(label)
    summarize(records, "all")

    if by_level:
        levels: dict[str, list[dict]] = defaultdict(list)

        for record in records:
            levels[record["level"]].append(record)

        for level in sorted(levels):
            summarize(levels[level], f"level {level}")

    grades: dict[str, list[dict]] = defaultdict(list)
    subjects: dict[str, list[dict]] = defaultdict(list)

    for record in records:
        grades[record["grade"]].append(record)
        subjects[record["subject"]].append(record)

    for grade in sorted(grades):
        summarize(grades[grade], f"grade {grade}")

    for subject in sorted(subjects):
        summarize(subjects[subject], f"subject {subject}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize knowledge results.")
    parser.add_argument("--model-id", nargs="+", required=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--tag",
        default="letter-natural-plain",
        help="Results subfolder, e.g. letter-natural-plain or mean_log_likelihood-answer.",
    )
    args = parser.parse_args()

    if not args.model_id:
        raise ValueError("Provide at least one --model-id.")

    by_model: dict[str, tuple[list[dict], list[dict]]] = {}

    for model_id in args.model_id:
        model_dir = args.results_dir / model_id / args.tag
        by_model[model_id] = (
            load_predictions(model_dir / "primary_sd_smp.jsonl"),
            load_predictions(model_dir / "retention_sma.jsonl"),
        )

    for model_id, (primary, retention) in by_model.items():
        print(model_id)
        summarize_slice(primary, "primary SD+SMP", by_level=True)
        summarize_slice(retention, "retention SMA")
        bias = position_bias_report(primary + retention)
        print(f"predicted_positions={bias['predicted_counts']}")
        print(f"gold_positions={bias['gold_counts']}")
        print(f"acc_by_gold={bias['accuracy_by_gold_position']}")
        both = primary + retention
        print(
            f"chance={chance_baseline(both):.4f} "
            f"position_debiased_acc={position_debiased_accuracy(both):.4f}"
        )

    if len(by_model) >= 2:
        reference_id = args.model_id[0]
        reference = by_model[reference_id]

        for other_id in args.model_id[1:]:
            other = by_model[other_id]

            for label, first, second in (
                ("primary", reference[0], other[0]),
                ("retention", reference[1], other[1]),
            ):
                first_map = {r["question_id"]: correctness(r) for r in first}
                second_map = {r["question_id"]: correctness(r) for r in second}
                shared = sorted(set(first_map) & set(second_map))

                if not shared:
                    raise ValueError(f"No shared question_ids in {label}.")

                low, high = paired_bootstrap_ci(
                    [float(first_map[q]) for q in shared],
                    [float(second_map[q]) for q in shared],
                    seed=42,
                )
                first_acc = sum(first_map[q] for q in shared) / len(shared)
                second_acc = sum(second_map[q] for q in shared) / len(shared)
                print(
                    f"paired {label} {reference_id}-{other_id}: n={len(shared)} "
                    f"[{low:.4f}, {high:.4f}] h={cohens_h(first_acc, second_acc):.3f}"
                )


if __name__ == "__main__":
    main()
