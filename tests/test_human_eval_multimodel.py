from edu_eval.human_eval.pack import build_pack
from edu_eval.human_eval.workflow import coverage_report, validate_ratings

DIMS = (
    "factual_correctness",
    "lexical_appropriateness",
    "syntactic_appropriateness",
    "conceptual_appropriateness",
    "pedagogical_appropriateness",
    "overall_grade_fit",
)


def _pack():
    records = [
        {"model_id": m, "case_id": "CG-001", "target_grade": "SD6", "output_text": "x"}
        for m in ("m0", "m1", "m4")
    ]
    refs = {"CG-001": {"subject": "IPA", "concept": "c", "reference_text": "r"}}
    pack, _ = build_pack(records, refs, 42)

    return pack


def _ratings(pack, rater="r1"):
    return [
        {
            "rating_id": f"{rater}:{item['pack_id']}",
            "pack_id": item["pack_id"],
            "case_id": item["case_id"],
            "target_grade": item["target_grade"],
            "rater_id": rater,
            **{d: 3 for d in DIMS},
        }
        for item in pack
    ]


def test_same_case_and_grade_from_three_models_is_valid():
    pack = _pack()

    assert validate_ratings(_ratings(pack)) == []


def test_true_duplicate_is_still_caught():
    pack = _pack()
    ratings = _ratings(pack)

    assert len(validate_ratings(ratings + [ratings[0]])) == 1


def test_coverage_counts_outputs_not_case_grade_pairs():
    pack = _pack()
    report = coverage_report(_ratings(pack), pack)

    assert report["n_total"] == 3
    assert report["n_rated"] == 3
    assert report["per_rater"] == {"r1": 3}
    assert report["unknown_pack_ids"] == []


def test_two_raters_on_same_item_are_not_duplicates():
    pack = _pack()

    assert validate_ratings(_ratings(pack, "r1") + _ratings(pack, "r2")) == []


def test_legacy_export_without_pack_id_still_works():
    pack = _pack()
    ratings = [{k: v for k, v in r.items() if k != "pack_id"} for r in _ratings(pack)]

    assert validate_ratings(ratings) == []
    assert coverage_report(ratings, pack)["n_rated"] == 3
