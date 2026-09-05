from typing import Literal

from pydantic import BaseModel, Field, model_validator

GradeTarget = Literal["SD6", "SMP7", "SMP9", "SMA10"]
GenerationTask = Literal["materi"]
CurriculumPhase = Literal["C", "D", "E"]
EvidenceScope = Literal["phase_only", "grade_specific"]
SourceType = Literal[
    "official_curriculum",
    "official_textbook",
    "official_atp",
    "other_authoritative",
    "project_capsule",
]

EXPECTED_PHASE_GRADE: dict[GradeTarget, tuple[CurriculumPhase, int]] = {
    "SD6": ("C", 6),
    "SMP7": ("D", 7),
    "SMP9": ("D", 9),
    "SMA10": ("E", 10),
}

EXPECTED_TARGETS = frozenset(EXPECTED_PHASE_GRADE)


class SourceReference(BaseModel):
    source_id: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    source_url: str | None = Field(default=None)
    source_type: SourceType
    source_version: str | None = Field(default=None)
    locator: str | None = Field(default=None)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_provenance_honesty(self) -> "SourceReference":
        if self.source_type == "project_capsule":
            if self.source_url is not None:
                raise ValueError(
                    "project_capsule references must not carry a source_url."
                )
        elif self.source_url is None or self.locator is None:
            raise ValueError(
                f"source_type={self.source_type!r} claims an external source "
                "and must provide both source_url and locator."
            )

        return self


class GradeEvidence(BaseModel):
    curriculum_phase: CurriculumPhase
    grade: int = Field(ge=1, le=13)
    evidence_source: str = Field(min_length=1)
    evidence_locator: str | None = Field(default=None)
    evidence_text: str = Field(min_length=1)
    evidence_scope: EvidenceScope = "phase_only"

    @model_validator(mode="after")
    def validate_grade_specific_claim(self) -> "GradeEvidence":
        if self.evidence_scope == "grade_specific" and self.evidence_locator is None:
            raise ValueError(
                "evidence_scope='grade_specific' requires evidence_locator."
            )

        return self


class ControlledGenerationCase(BaseModel):
    case_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    task_type: GenerationTask = "materi"
    reference: SourceReference
    target_evidence: dict[GradeTarget, GradeEvidence]

    @model_validator(mode="after")
    def validate_controlled_condition(self) -> "ControlledGenerationCase":
        if set(self.target_evidence) != EXPECTED_TARGETS:
            raise ValueError(
                "Each controlled case must provide target evidence "
                "for SD6, SMP7, SMP9, and SMA10."
            )

        for target, evidence in self.target_evidence.items():
            expected_phase, expected_grade = EXPECTED_PHASE_GRADE[target]

            if (
                evidence.curriculum_phase != expected_phase
                or evidence.grade != expected_grade
            ):
                raise ValueError(
                    f"Invalid phase mapping for {target}: "
                    f"expected phase {expected_phase} grade {expected_grade}, "
                    f"got phase {evidence.curriculum_phase} grade {evidence.grade}."
                )

        return self
