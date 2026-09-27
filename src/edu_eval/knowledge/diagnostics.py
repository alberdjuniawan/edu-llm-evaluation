def correctness(record: dict) -> bool:
    predictions = record["predictions"]

    if "mean_log_likelihood" in predictions:
        return bool(predictions["mean_log_likelihood"]["correct"])

    return bool(predictions["official"]["correct"])


def position_bias_report(records: list[dict]) -> dict:
    predicted_counts: dict[int, int] = {}
    gold_counts: dict[int, int] = {}
    gold_correct: dict[int, int] = {}
    gold_total: dict[int, int] = {}

    for record in records:
        predictions = record["predictions"]

        if "mean_log_likelihood" in predictions:
            predicted = int(predictions["mean_log_likelihood"]["predicted_index"])
        else:
            predicted = int(predictions["official"]["predicted_index"])

        gold = int(record["gold_index"])
        predicted_counts[predicted] = predicted_counts.get(predicted, 0) + 1
        gold_counts[gold] = gold_counts.get(gold, 0) + 1
        gold_total[gold] = gold_total.get(gold, 0) + 1

        if correctness(record):
            gold_correct[gold] = gold_correct.get(gold, 0) + 1

    return {
        "n": len(records),
        "predicted_counts": predicted_counts,
        "gold_counts": gold_counts,
        "accuracy_by_gold_position": {
            position: gold_correct.get(position, 0) / total
            for position, total in gold_total.items()
        },
    }
