"""Move pre-tag knowledge results (results/knowledge/<model>/*.jsonl) into their
scoring-method subfolder so they are never mixed with new runs. Idempotent."""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILES = ("primary_sd_smp.jsonl", "retention_sma.jsonl", "run_metadata.json")


def tag_from_metadata(metadata: dict) -> str:
    scoring = metadata["scoring"]
    mode = scoring["score_mode"]

    if mode == "letter":
        return f"letter-{scoring['letter_style']}-{scoring['prompt_format']}"

    return f"{mode}-{scoring['score_span']}"


def migrate(results_dir: Path) -> list[str]:
    moved: list[str] = []

    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        meta_path = model_dir / "run_metadata.json"

        if not meta_path.exists():
            continue

        tag = tag_from_metadata(json.loads(meta_path.read_text(encoding="utf-8")))
        target = model_dir / tag
        target.mkdir(exist_ok=True)

        for name in FILES:
            source = model_dir / name

            if source.exists():
                if (target / name).exists():
                    raise FileExistsError(f"{target / name} already exists.")

                shutil.move(str(source), str(target / name))
                moved.append(f"{model_dir.name}/{name} -> {tag}/")

    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "results" / "knowledge"
    )
    args = parser.parse_args()

    for line in migrate(args.results_dir) or ["nothing to migrate"]:
        print(line)


if __name__ == "__main__":
    main()
