import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

from edu_eval.knowledge.indommlu import load_indommlu_csv
from edu_eval.knowledge.schema import KnowledgeQuestion
from edu_eval.knowledge.scorer import KnowledgePrediction, KnowledgeScorer
from edu_eval.models.loader import ModelLoader
from edu_eval.models.registry import ModelRegistry
from edu_eval.runtime.env import (
    collect_runtime,
    get_git_commit,
    seed_everything,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "IndoMMLU.csv"
MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"
KNOWLEDGE_CONFIG = PROJECT_ROOT / "configs" / "knowledge.yaml"
RUNTIME_CONFIG = PROJECT_ROOT / "configs" / "runtime.yaml"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "knowledge"


def load_runtime_seed(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("Runtime configuration must be a YAML mapping.")

    runtime = config.get("runtime")

    if not isinstance(runtime, dict):
        raise TypeError("'runtime' must be a YAML mapping.")

    seed = runtime.get("seed")

    if not isinstance(seed, int):
        raise TypeError("'runtime.seed' must be an integer.")

    return seed


def load_knowledge_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("Knowledge configuration must be a YAML mapping.")

    knowledge = config.get("knowledge")

    if not isinstance(knowledge, dict):
        raise TypeError("'knowledge' must be a YAML mapping.")

    return knowledge


def prediction_correct(result: KnowledgePrediction) -> bool:
    predictions = result["predictions"]

    if "mean_log_likelihood" in predictions:
        return bool(predictions["mean_log_likelihood"]["correct"])

    return bool(predictions["official"]["correct"])


def read_existing_results(
    path: Path,
    questions: list[KnowledgeQuestion],
) -> tuple[int, int]:
    if not path.exists():
        return 0, 0

    processed = 0
    correct = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}.") from exc

            if processed >= len(questions):
                raise RuntimeError(
                    f"Existing results contain more records than "
                    f"expected questions in {path}."
                )

            expected_id = questions[processed].question_id
            actual_id = result.get("question_id")

            if actual_id != expected_id:
                raise RuntimeError(
                    f"Existing result mismatch at position {processed}: "
                    f"expected question_id={expected_id}, "
                    f"got {actual_id}."
                )

            correct += int(prediction_correct(result))

            processed += 1

    return processed, correct


def write_results(
    path: Path,
    results: Sequence[KnowledgePrediction],
) -> None:
    with path.open("a", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_level(
    scorer: KnowledgeScorer,
    questions: list[KnowledgeQuestion],
    level: str,
    output_path: Path,
    batch_size: int,
) -> dict:
    total_questions = len(questions)

    processed_questions, correct_questions = read_existing_results(
        output_path,
        questions,
    )

    if processed_questions == total_questions:
        print(f"{level}: done {total_questions}/{total_questions}")

        return {
            "level": level,
            "questions": total_questions,
            "correct": correct_questions,
            "accuracy": correct_questions / total_questions,
            "resumed_from": processed_questions,
            "newly_processed": 0,
            "elapsed_seconds": 0.0,
            "throughput_questions_per_second": None,
        }

    remaining_questions = questions[processed_questions:]

    print(f"{level}: resume {processed_questions}/{total_questions}")

    start_time = time.perf_counter()

    chunk_size = 100

    for start in range(
        0,
        len(remaining_questions),
        chunk_size,
    ):
        question_batch = remaining_questions[start : start + chunk_size]

        results = scorer.predict_many(
            question_batch,
            batch_size=batch_size,
        )

        write_results(
            output_path,
            results,
        )

        processed_questions += len(results)

        correct_questions += sum(prediction_correct(result) for result in results)

        elapsed = time.perf_counter() - start_time
        newly_processed = processed_questions - (
            total_questions - len(remaining_questions)
        )

        throughput = newly_processed / elapsed

        print(
            f"{level}: {processed_questions}/{total_questions} "
            f"acc={correct_questions / processed_questions:.4f} "
            f"{throughput:.1f}/s"
        )

    elapsed = time.perf_counter() - start_time

    newly_processed = processed_questions - (total_questions - len(remaining_questions))

    return {
        "level": level,
        "questions": total_questions,
        "correct": correct_questions,
        "accuracy": correct_questions / total_questions,
        "resumed_from": total_questions - len(remaining_questions),
        "newly_processed": newly_processed,
        "elapsed_seconds": elapsed,
        "throughput_questions_per_second": (newly_processed / elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full IndoMMLU knowledge benchmark.",
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="Model ID from configs/models.yaml.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for benchmark results.",
    )

    args = parser.parse_args()

    knowledge_config = load_knowledge_config(KNOWLEDGE_CONFIG)

    seed = load_runtime_seed(RUNTIME_CONFIG)
    seed_everything(seed)

    registry = ModelRegistry.from_yaml(MODELS_CONFIG)

    spec = registry.get(args.model_id)

    results_root = args.results_dir.resolve()

    model_dir = results_root / spec.model_id

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"knowledge {spec.model_id} batch={knowledge_config['batch_size']}")

    questions = load_indommlu_csv(DATA_PATH)

    zero_shot_questions = [
        question for question in questions if not question.is_for_fewshot
    ]

    primary_questions = [
        question
        for question in zero_shot_questions
        if question.level in knowledge_config["primary_levels"]
    ]

    retention_questions = [
        question
        for question in zero_shot_questions
        if question.level in knowledge_config["retention_levels"]
    ]

    print(
        f"n_zero={len(zero_shot_questions)} "
        f"n_primary={len(primary_questions)} "
        f"n_retention={len(retention_questions)}"
    )

    tokenizer, model, device = ModelLoader.load(spec)

    dtype = next(model.parameters()).dtype

    print(f"device={device} dtype={dtype}")

    scorer = KnowledgeScorer(model=model, tokenizer=tokenizer, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    benchmark_start = time.perf_counter()

    primary_results_path = model_dir / "primary_sd_smp.jsonl"
    retention_results_path = model_dir / "retention_sma.jsonl"

    primary_summary = run_level(
        scorer=scorer,
        questions=primary_questions,
        level="SD+SMP",
        output_path=primary_results_path,
        batch_size=knowledge_config["batch_size"],
    )

    retention_summary = run_level(
        scorer=scorer,
        questions=retention_questions,
        level="SMA",
        output_path=retention_results_path,
        batch_size=knowledge_config["batch_size"],
    )

    benchmark_elapsed = time.perf_counter() - benchmark_start

    peak_memory_gb = None

    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    metadata = {
        "git_commit": get_git_commit(PROJECT_ROOT),
        "seed": seed,
        "model_id": spec.model_id,
        "model_source": spec.source,
        "model_revision": spec.revision,
        "device": str(device),
        "dtype": str(dtype),
        "scoring": {
            "score_mode": knowledge_config["primary_score_mode"],
            "batch_size": knowledge_config["batch_size"],
            "exclude_fewshot": knowledge_config["exclude_fewshot"],
        },
        "dataset": {
            "zero_shot_questions": len(zero_shot_questions),
            "primary_questions": len(primary_questions),
            "retention_questions": len(retention_questions),
        },
        "primary": primary_summary,
        "retention": retention_summary,
        "benchmark_elapsed_seconds": benchmark_elapsed,
        "benchmark_elapsed_minutes": benchmark_elapsed / 60,
        "peak_gpu_memory_gb": peak_memory_gb,
        "runtime": collect_runtime(),
    }

    metadata_path = model_dir / "run_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(
        f"primary={primary_summary['accuracy']:.4f} sma={retention_summary['accuracy']:.4f}"
    )
    print(f"{benchmark_elapsed / 60:.1f} min -> {model_dir}")

    del scorer
    del model
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
