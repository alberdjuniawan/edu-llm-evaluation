from edu_eval.human_eval.protocol import RatingRecord, shuffled


def split_calibration(
    pack: list[dict], batch_size: int, seed: int
) -> tuple[list[dict], list[dict]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    ordered = shuffled(pack, seed)

    return ordered[:batch_size], ordered[batch_size:]


def sample_pairwise(pack: list[dict], fraction: float, seed: int) -> list[dict]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")

    count = max(1, int(len(pack) * fraction))

    return shuffled(pack, seed)[:count]


def validate_ratings(ratings: list[dict]) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for index, raw in enumerate(ratings):
        try:
            record = RatingRecord.model_validate(raw)
        except ValueError as exc:
            errors.append(f"row {index}: invalid record: {exc}")

            continue

        key = (record.rater_id, record.case_id, record.target_grade)

        if key in seen:
            errors.append(f"row {index}: duplicate rating for {key}.")

        seen.add(key)

    return errors


def coverage_report(ratings: list[dict], pack: list[dict]) -> dict:
    pack_keys = {(item["case_id"], item["target_grade"]) for item in pack}
    rated_keys = {
        (record["case_id"], record["target_grade"])
        for record in ratings
        if isinstance(record, dict) and "case_id" in record
    }
    by_grade: dict[str, int] = {}

    for case_id, grade in rated_keys:
        by_grade[grade] = by_grade.get(grade, 0) + 1

    return {
        "n_total": len(pack_keys),
        "n_rated": len(rated_keys & pack_keys),
        "by_grade": by_grade,
        "raters": sorted(
            {
                record["rater_id"]
                for record in ratings
                if isinstance(record, dict) and "rater_id" in record
            }
        ),
    }
