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


def pack_id_of(record: dict) -> str:
    """Pack item a rating belongs to (explicit field, else the tail of rating_id)."""
    explicit = record.get("pack_id")

    if explicit:
        return str(explicit)

    return str(record.get("rating_id", "")).rsplit(":", 1)[-1]


def validate_ratings(ratings: list[dict]) -> list[str]:
    """One rating per (rater, pack item). A pack item is one blinded output, so the
    same case+grade from different models is NOT a duplicate."""
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()

    for index, raw in enumerate(ratings):
        try:
            record = RatingRecord.model_validate(raw)
        except ValueError as exc:
            errors.append(f"row {index}: invalid record: {exc}")

            continue

        key = (record.rater_id, pack_id_of(raw))

        if key in seen:
            errors.append(f"row {index}: duplicate rating for {key}.")

        seen.add(key)

    return errors


def coverage_report(ratings: list[dict], pack: list[dict]) -> dict:
    by_pack = {item["pack_id"]: item for item in pack}
    valid = [r for r in ratings if isinstance(r, dict) and "rating_id" in r]
    rated_ids = {pack_id_of(r) for r in valid}
    known = rated_ids & set(by_pack)
    by_grade: dict[str, int] = {}

    for pack_id in known:
        grade = by_pack[pack_id]["target_grade"]
        by_grade[grade] = by_grade.get(grade, 0) + 1

    per_rater: dict[str, set[str]] = {}

    for record in valid:
        pack_id = pack_id_of(record)

        if pack_id in by_pack and "rater_id" in record:
            per_rater.setdefault(record["rater_id"], set()).add(pack_id)

    return {
        "n_total": len(by_pack),
        "n_rated": len(known),
        "by_grade": by_grade,
        "per_rater": {rater: len(ids) for rater, ids in sorted(per_rater.items())},
        "unknown_pack_ids": sorted(rated_ids - set(by_pack)),
        "raters": sorted(per_rater),
    }
