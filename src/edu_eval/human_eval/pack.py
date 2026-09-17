from edu_eval.human_eval.protocol import assign_blind_labels, shuffled


def build_pack(
    records: list[dict], references: dict[str, dict], seed: int
) -> tuple[list[dict], dict[str, str]]:
    models = sorted({record["model_id"] for record in records})

    if not models:
        raise ValueError("records must not be empty.")

    mapping = assign_blind_labels(models, seed)
    pack = []

    for position, record in enumerate(shuffled(records, seed + 1)):
        reference = references.get(record["case_id"], {})
        pack.append(
            {
                "pack_id": f"PILOT-{position:03d}",
                "blind_model_label": mapping[record["model_id"]],
                "case_id": record["case_id"],
                "target_grade": record["target_grade"],
                "subject": reference.get("subject", ""),
                "concept": reference.get("concept", ""),
                "reference_text": reference.get("reference_text", ""),
                "output_text": record["output_text"],
            }
        )

    return pack, mapping
