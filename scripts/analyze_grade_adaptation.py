import argparse
import json
from collections import defaultdict
from pathlib import Path

from edu_eval.generation.quality import flag_rates
from edu_eval.linguistic.adaptation import METRICS, features, summarize_adaptation
from edu_eval.linguistic.mentions import classify_mentions

VALID_TRUNCATION = 0.10


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-case grade-adaptation analysis of controlled generation."
    )
    parser.add_argument(
        "outputs", nargs="+", type=Path, help="controlled_outputs.jsonl"
    )
    parser.add_argument("--n-words", type=int, default=150)
    args = parser.parse_args()

    by_model: dict[str, list[dict]] = defaultdict(list)

    for path in args.outputs:
        for record in load(path):
            by_model[record["model_id"]].append(record)

    for model_id, records in by_model.items():
        rates = flag_rates(records)
        mentions = sum(
            classify_mentions(r["output_text"])["grade_mention"] for r in records
        ) / len(records)
        print(f"\n=== {model_id} (n={len(records)})")
        print(
            f"truncated={rates['truncated']:.0%} thinking_leak={rates['thinking_leak']:.0%} "
            f"repetitive={rates['repetitive']:.0%} grade_mention={mentions:.0%}"
        )

        if rates["thinking_leak"] > 0 or rates["truncated"] > VALID_TRUNCATION:
            print(
                "!! TIDAK VALID untuk klaim adaptasi: output berisi penalaran atau "
                "terpotong. Perbaiki generation dulu."
            )

        rows = [
            {
                "case_id": r["case_id"],
                "target_grade": r["target_grade"],
                "features": features(r["output_text"], args.n_words),
            }
            for r in records
        ]

        for metric in METRICS:
            s = summarize_adaptation(rows, metric)
            print(
                f"{metric:22s} tau={s['mean_tau']:+.2f} "
                f"[{s['tau_ci'][0]:+.2f},{s['tau_ci'][1]:+.2f}] "
                f"pos={s['cases_positive']}/{s['n_cases']}  "
                f"SMA10-SD6={s['mean_sma10_minus_sd6']:+.2f} "
                f"[{s['sma10_minus_sd6_ci'][0]:+.2f},{s['sma10_minus_sd6_ci'][1]:+.2f}] "
                f"({s['cases_sma10_gt_sd6']}/{s['n_cases']} SMA>SD)"
            )


if __name__ == "__main__":
    main()
