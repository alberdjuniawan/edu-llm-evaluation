from edu_eval.knowledge.diagnostics import correctness, position_bias_report


def _record(predicted, gold, key="mean_log_likelihood"):
    return {
        "question_id": "q",
        "gold_index": gold,
        "predictions": {
            key: {"predicted_index": predicted, "correct": predicted == gold}
        },
    }


def test_correctness_prefers_current_key():
    record = {
        "question_id": "q",
        "gold_index": 0,
        "predictions": {
            "mean_log_likelihood": {"predicted_index": 1, "correct": False},
            "official": {"predicted_index": 0, "correct": True},
        },
    }

    assert correctness(record) is False


def test_correctness_legacy_key():
    assert correctness(_record(2, 2, key="official")) is True
    assert correctness(_record(0, 2, key="official")) is False


def test_position_bias_report():
    records = [
        _record(0, 0),
        _record(0, 1),
        _record(1, 1),
        _record(0, 2, key="official"),
    ]

    report = position_bias_report(records)

    assert report["n"] == 4
    assert report["predicted_counts"] == {0: 3, 1: 1}
    assert report["gold_counts"] == {0: 1, 1: 2, 2: 1}
    assert report["accuracy_by_gold_position"] == {0: 1.0, 1: 0.5, 2: 0.0}
