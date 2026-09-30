import random
from typing import Literal

from pydantic import BaseModel, Field

from edu_eval.generation.schema import GradeTarget

RUBRIC_DIMENSIONS = (
    "factual_correctness",
    "lexical_appropriateness",
    "syntactic_appropriateness",
    "conceptual_appropriateness",
    "pedagogical_appropriateness",
    "overall_grade_fit",
)

GRADE_FIT_ANCHORS = {
    1: "Clearly inappropriate; multiple levels too high/low.",
    2: "Many parts require substantial simplification/reworking.",
    3: "Borderline; understandable but with a clear mismatch.",
    4: "Generally appropriate; only minor mismatch.",
    5: "Highly appropriate; vocabulary, syntax, conceptual depth, and "
    "presentation are consistently appropriate.",
}


class RatingRecord(BaseModel):
    rating_id: str = Field(min_length=1)
    pack_id: str | None = None
    case_id: str = Field(min_length=1)
    target_grade: GradeTarget
    rater_id: str = Field(min_length=1)
    factual_correctness: int = Field(ge=1, le=5)
    lexical_appropriateness: int = Field(ge=1, le=5)
    syntactic_appropriateness: int = Field(ge=1, le=5)
    conceptual_appropriateness: int = Field(ge=1, le=5)
    pedagogical_appropriateness: int = Field(ge=1, le=5)
    overall_grade_fit: int = Field(ge=1, le=5)


class PairwiseRecord(BaseModel):
    pair_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    target_grade: GradeTarget
    rater_id: str = Field(min_length=1)
    left_ref: str = Field(min_length=1)
    right_ref: str = Field(min_length=1)
    choice: Literal["A", "B", "tie"]


def assign_blind_labels(model_ids: list[str], seed: int) -> dict[str, str]:
    if not model_ids:
        raise ValueError("model_ids must not be empty.")

    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids must be unique.")

    ordered = sorted(model_ids)
    rng = random.Random(seed)
    labels = [chr(ord("A") + position) for position in range(len(ordered))]
    rng.shuffle(labels)

    return dict(zip(ordered, labels, strict=True))


def shuffled(items: list, seed: int) -> list:
    ordered = list(items)
    random.Random(seed).shuffle(ordered)

    return ordered
