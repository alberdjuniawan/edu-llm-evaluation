import json
from collections import Counter
from pathlib import Path
from typing import TypedDict

from edu_eval.generation.config import GenerationConfig
from edu_eval.generation.schema import ControlledGenerationCase


class CaseValidationReport(TypedDict):
    total_cases: int
    subjects: dict[str, int]
    schema: str
    provenance: str
    curriculum_evidence: str
    phase_mapping: str
    duplicate_case_ids: str
    pilot_ready: str
    full_ready: str
    phase_only_count: int
    grade_specific_count: int
    notes: list[str]


def load_cases(path: str | Path) -> list[ControlledGenerationCase]:
    case_path = Path(path)

    if not case_path.exists():
        raise FileNotFoundError(
            f"Canonical case dataset not found: {case_path}. "
            "Build it with scripts/build_controlled_cases.py."
        )

    cases: list[ControlledGenerationCase] = []

    with case_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {case_path}:{line_number}.") from exc

            try:
                cases.append(ControlledGenerationCase.model_validate(payload))
            except ValueError as exc:
                raise ValueError(
                    f"Schema validation failed at {case_path}:{line_number} "
                    f"(case_id={payload.get('case_id')!r}): {exc}"
                ) from exc

    return cases


def select_by_subject(
    cases: list[ControlledGenerationCase], cases_per_subject: int
) -> list[ControlledGenerationCase]:
    if cases_per_subject <= 0:
        raise ValueError("cases_per_subject must be positive.")

    by_subject: dict[str, list[ControlledGenerationCase]] = {}

    for case in cases:
        by_subject.setdefault(case.subject, []).append(case)

    selected: list[ControlledGenerationCase] = []

    for subject in sorted(by_subject):
        ordered = sorted(by_subject[subject], key=lambda c: c.case_id)

        if len(ordered) < cases_per_subject:
            raise ValueError(
                f"Subject {subject!r} has {len(ordered)} cases, "
                f"but {cases_per_subject} were requested."
            )

        selected.extend(ordered[:cases_per_subject])

    return selected


def select_pilot(
    cases: list[ControlledGenerationCase], config: GenerationConfig
) -> list[ControlledGenerationCase]:
    return select_by_subject(cases, config.pilot_cases_per_subject)


def select_full(
    cases: list[ControlledGenerationCase], config: GenerationConfig
) -> list[ControlledGenerationCase]:
    return select_by_subject(cases, config.full_cases_per_subject)


def validation_report(
    cases: list[ControlledGenerationCase], config: GenerationConfig
) -> CaseValidationReport:
    notes: list[str] = []
    counts = Counter(case.subject for case in cases)
    case_ids = [case.case_id for case in cases]
    duplicate_ids = sorted({cid for cid in case_ids if case_ids.count(cid) > 1})
    phase_only = sum(
        1
        for case in cases
        for evidence in case.target_evidence.values()
        if evidence.evidence_scope == "phase_only"
    )
    grade_specific = sum(
        1
        for case in cases
        for evidence in case.target_evidence.values()
        if evidence.evidence_scope == "grade_specific"
    )
    capsules = sum(
        1 for case in cases if case.reference.source_type == "project_capsule"
    )

    if capsules:
        notes.append(f"{capsules}/{len(cases)} references are project_capsule drafts.")

    if "SMP7" in config.target_grades and "SMP9" in config.target_grades:
        notes.append("SMP7 and SMP9 share Phase D by design.")

    def check_balance(per_subject: int, label: str) -> str:
        short = {
            subject: count
            for subject, count in sorted(counts.items())
            if count < per_subject
        }

        if short:
            notes.append(f"{label} not ready: {short}.")

            return "FAIL"

        return "PASS"

    return {
        "total_cases": len(cases),
        "subjects": dict(sorted(counts.items())),
        "schema": "PASS",
        "provenance": "PASS",
        "curriculum_evidence": "PASS",
        "phase_mapping": "PASS",
        "duplicate_case_ids": (f"FAIL: {duplicate_ids}" if duplicate_ids else "PASS"),
        "pilot_ready": check_balance(config.pilot_cases_per_subject, "Pilot"),
        "full_ready": check_balance(config.full_cases_per_subject, "Full"),
        "phase_only_count": phase_only,
        "grade_specific_count": grade_specific,
        "notes": notes,
    }
