import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from edu_eval.linguistic.analyzer import LinguisticFeatures, analyze_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "linguistic"


def load_linguistic_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or not isinstance(config.get("linguistic"), dict):
        raise TypeError("linguistic config must map 'linguistic' to a mapping.")

    return config["linguistic"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Linguistic diagnostics.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "linguistic.yaml"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--name", default="features")
    args = parser.parse_args()

    config = load_linguistic_config(args.config)
    min_chars = int(config.get("min_text_chars", 1))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.results_dir / f"{args.name}.jsonl"
    by_grade: dict[str, list[LinguisticFeatures]] = defaultdict(list)

    with output_path.open("w", encoding="utf-8") as out:
        for input_path in args.inputs:
            with Path(input_path).open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    text = record.get("output_text", "")

                    if len(text) < min_chars:
                        continue

                    features = analyze_text(text)
                    by_grade[record.get("target_grade", "?")].append(features)
                    out.write(
                        json.dumps(
                            {
                                "model_id": record.get("model_id"),
                                "case_id": record.get("case_id"),
                                "target_grade": record.get("target_grade"),
                                "features": features,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    total = sum(len(v) for v in by_grade.values())
    print(f"{total} records -> {output_path}")

    for grade in sorted(by_grade):
        feats = by_grade[grade]
        mean_wps = sum(f["words_per_sentence"] for f in feats) / len(feats)
        mean_ttr = sum(f["lexical_diversity_ttr"] for f in feats) / len(feats)
        print(f"{grade}: n={len(feats)} wps={mean_wps:.1f} ttr={mean_ttr:.3f}")


if __name__ == "__main__":
    main()
