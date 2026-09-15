import pytest
from pydantic import ValidationError

from edu_eval.human_eval.protocol import (
    GRADE_FIT_ANCHORS,
    RUBRIC_DIMENSIONS,
    PairwiseRecord,
    RatingRecord,
    assign_blind_labels,
    shuffled,
)


def test_rubric_has_six_dimensions_and_five_anchors():
    assert len(RUBRIC_DIMENSIONS) == 6
    assert set(GRADE_FIT_ANCHORS) == {1, 2, 3, 4, 5}


def test_rating_record_accepts_valid_scores():
    record = RatingRecord(
        rating_id="R-1",
        case_id="CG-001",
        target_grade="SD6",
        rater_id="guru-01",
        factual_correctness=5,
        lexical_appropriateness=4,
        syntactic_appropriateness=4,
        conceptual_appropriateness=4,
        pedagogical_appropriateness=4,
        overall_grade_fit=4,
    )

    assert record.overall_grade_fit == 4


def test_rating_record_rejects_out_of_range():
    with pytest.raises(ValidationError):
        RatingRecord(
            rating_id="R-1",
            case_id="CG-001",
            target_grade="SD6",
            rater_id="guru-01",
            factual_correctness=6,
            lexical_appropriateness=4,
            syntactic_appropriateness=4,
            conceptual_appropriateness=4,
            pedagogical_appropriateness=4,
            overall_grade_fit=4,
        )


def test_pairwise_record_rejects_bad_choice():
    with pytest.raises(ValidationError):
        PairwiseRecord(
            pair_id="P-1",
            case_id="CG-001",
            target_grade="SD6",
            rater_id="guru-01",
            left_ref="A",
            right_ref="B",
            choice="C",
        )


def test_blind_labels_deterministic_and_unique():
    first = assign_blind_labels(["m1", "m0", "m3"], seed=42)
    second = assign_blind_labels(["m0", "m1", "m3"], seed=42)

    assert first == second
    assert sorted(first.values()) == ["A", "B", "C"]


def test_blind_labels_reject_duplicates():
    with pytest.raises(ValueError):
        assign_blind_labels(["m1", "m1"], seed=1)


def test_shuffled_is_deterministic():
    assert shuffled([1, 2, 3, 4], seed=7) == shuffled([1, 2, 3, 4], seed=7)
