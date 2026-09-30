from edu_eval.human_eval.workflow import (
    coverage_report,
    sample_pairwise,
    split_calibration,
    validate_ratings,
)


def _pack():
    return [
        {
            "pack_id": f"PILOT-{i:03d}",
            "blind_model_label": "A",
            "case_id": f"CG-{i:03d}",
            "target_grade": grade,
            "output_text": "materi",
        }
        for i, grade in enumerate(["SD6", "SMP7", "SMP9", "SMA10"] * 2)
    ]


def _rating(pack_id="PILOT-000", case="CG-000", grade="SD6", rater="guru-01"):
    return {
        "rating_id": f"{rater}:{pack_id}",
        "case_id": case,
        "target_grade": grade,
        "rater_id": rater,
        "factual_correctness": 4,
        "lexical_appropriateness": 4,
        "syntactic_appropriateness": 4,
        "conceptual_appropriateness": 4,
        "pedagogical_appropriateness": 4,
        "overall_grade_fit": 4,
    }


def test_split_calibration_deterministic():
    first, rest_first = split_calibration(_pack(), 2, seed=7)
    second, rest_second = split_calibration(_pack(), 2, seed=7)

    assert [item["pack_id"] for item in first] == [item["pack_id"] for item in second]
    assert len(first) == 2
    assert len(first) + len(rest_first) == 8
    assert len(rest_second) == 6


def test_sample_pairwise_fraction():
    subset = sample_pairwise(_pack(), 0.25, seed=7)

    assert len(subset) == 2


def test_validate_ratings_clean():
    assert (
        validate_ratings([_rating(), _rating(pack_id="PILOT-001", case="CG-001")]) == []
    )


def test_validate_ratings_catches_problems():
    bad = {"rating_id": "x", "case_id": "CG-000", "target_grade": "SD6"}
    duplicate = _rating()

    errors = validate_ratings([_rating(), duplicate, bad])

    assert len(errors) == 2


def test_coverage_report():
    pack = _pack()
    ratings = [
        _rating(pack_id="PILOT-000", case="CG-000", grade="SD6"),
        _rating(pack_id="PILOT-001", case="CG-001", grade="SMP7"),
    ]

    report = coverage_report(ratings, pack)

    assert report["n_total"] == 8
    assert report["n_rated"] == 2
    assert report["raters"] == ["guru-01"]
