import argparse
import json
from collections import defaultdict
from pathlib import Path

from edu_eval.knowledge.diagnostics import correctness, position_bias_report
from edu_eval.statistics.metrics import bootstrap_ci, paired_bootstrap_ci

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


def summarize(records: list[dict], label: str) -> None:
    flags = [1.0 if correctness(record) else 0.0 for record in records]
    low, high = bootstrap_ci(flags, seed=42)
    print(
        f"{label}: n={len(flags)} acc={sum(flags) / len(flags):.4f} [{low:.4f}, {high:.4f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize knowledge results.")
    parser.add_argument("--model-id", nargs="+", required=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    if len(args.model_id) > 2:
        raise ValueError("Compare at most two models at a time.")

    by_model: dict[str, list[dict]] = {}

    for model_id in args.model_id:
        model_dir = args.results_dir / model_id
        by_model[model_id] = load_predictions(
            model_dir / "primary_sd_smp.jsonl"
        ) + load_predictions(model_dir / "retention_sma.jsonl")

    for model_id, records in by_model.items():
        print(model_id)
        summarize(records, "overall")
        bias = position_bias_report(records)
        print(f"predicted_positions={bias['predicted_counts']}")
        print(f"gold_positions={bias['gold_counts']}")
        print(f"acc_by_gold={bias['accuracy_by_gold_position']}")
        levels: dict[str, list[dict]] = defaultdict(list)
        subjects: dict[str, list[dict]] = defaultdict(list)

        for record in records:
            levels[record["level"]].append(record)
            subjects[record["subject"]].append(record)

        for level in sorted(levels):
            summarize(levels[level], f"level {level}")

        for subject in sorted(subjects):
            summarize(subjects[subject], f"subject {subject}")

    if len(by_model) == 2:
        first, second = (by_model[mid] for mid in args.model_id)
        first_map = {r["question_id"]: correctness(r) for r in first}
        second_map = {r["question_id"]: correctness(r) for r in second}
        shared = sorted(set(first_map) & set(second_map))

        if not shared:
            raise ValueError("No shared question_ids for paired comparison.")

        low, high = paired_bootstrap_ci(
            [float(first_map[q]) for q in shared],
            [float(second_map[q]) for q in shared],
            seed=42,
        )
        print(
            f"paired {args.model_id[0]}-{args.model_id[1]}: n={len(shared)} [{low:.4f}, {high:.4f}]"
        )


if __name__ == "__main__":
    main()
