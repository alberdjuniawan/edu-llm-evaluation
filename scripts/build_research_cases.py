import argparse
import json
from pathlib import Path

from edu_eval.generation.cases import load_cases, validation_report
from edu_eval.generation.config import GenerationConfig
from edu_eval.generation.schema import ControlledGenerationCase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "curated_cases.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "controlled_generation_cases.jsonl"
)
GENERATION_CONFIG = PROJECT_ROOT / "configs" / "generation.yaml"


def check_research_record(payload: dict, line_number: int) -> ControlledGenerationCase:
    try:
        case = ControlledGenerationCase.model_validate(payload)
    except ValueError as exc:
        raise ValueError(f"line {line_number}: schema rejected: {exc}") from exc

    if case.reference.source_type == "project_capsule":
        raise ValueError(f"line {line_number}: project_capsule is pilot-only.")

    for target, evidence in case.target_evidence.items():
        if evidence.evidence_scope != "grade_specific":
            raise ValueError(f"line {line_number}: {target} is not grade_specific.")

    return case


def main() -> None:
    parser = argparse.ArgumentParser(description="Build research case dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-per-subject", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Curated input not found: {args.input}.")

    if args.output.exists() and args.output.stat().st_size > 0 and not args.force:
        raise RuntimeError(f"Refusing to overwrite {args.output} without --force.")

    payloads = []

    with args.input.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON.") from exc

            if not isinstance(payload, dict):
                raise TypeError(f"line {line_number}: record must be an object.")

            payloads.append(check_research_record(payload, line_number))

    if not payloads:
        raise ValueError("No records found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        for case in payloads:
            file.write(
                json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n"
            )

    print(f"{len(payloads)} cases -> {args.output}")

    cases = load_cases(args.output)
    config = GenerationConfig.from_yaml(GENERATION_CONFIG)

    if args.min_per_subject is not None:
        config = config.model_copy(
            update={"full_cases_per_subject": args.min_per_subject}
        )

    report = validation_report(cases, config)
    print(f"research={report['research_ready']}")

    for note in report["notes"]:
        print(f"note: {note}")

    if report["research_ready"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
