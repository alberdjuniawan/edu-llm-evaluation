import argparse
import gc
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

from edu_eval.knowledge.diagnostics import correctness
from edu_eval.knowledge.indommlu import load_indommlu_csv
from edu_eval.knowledge.letter_scorer import LetterScorer
from edu_eval.knowledge.preflight import (
    check_batch_equivalence,
    evaluate_knowledge_preflight,
)
from edu_eval.knowledge.schema import KnowledgeQuestion
from edu_eval.knowledge.scorer import KnowledgeScorer
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


def result_signature(result: dict) -> tuple:
    """What produced this record; used so resume never mixes scoring methods."""
    mode = result.get("score_mode", "mean_log_likelihood")

    if mode == "letter":
        return ("letter", result.get("letter_style"), result.get("prompt_format"))

    return ("mean_log_likelihood", result.get("score_span"))


def result_tag(signature: tuple, limit: int | None) -> str:
    tag = "-".join(str(part) for part in signature)

    return f"{tag}-n{limit}" if limit else tag


def read_existing_results(
    path: Path,
    questions: list[KnowledgeQuestion],
    expected_signature: tuple,
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

            if result_signature(result) != expected_signature:
                raise RuntimeError(
                    f"{path}:{line_number} was scored with "
                    f"{result_signature(result)} but this run uses "
                    f"{expected_signature}. Use a different results dir/tag."
                )

            expected_id = questions[processed].question_id
            actual_id = result.get("question_id")

            if actual_id != expected_id:
                raise RuntimeError(
                    f"Existing result mismatch at position {processed}: "
                    f"expected question_id={expected_id}, "
                    f"got {actual_id}."
                )

            correct += int(correctness(result))
            processed += 1

    return processed, correct


def write_results(path: Path, results: Sequence[dict]) -> None:
    with path.open("a", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_level(
    scorer,
    questions: list[KnowledgeQuestion],
    level: str,
    output_path: Path,
    batch_size: int,
) -> dict:
    total_questions = len(questions)

    processed_questions, correct_questions = read_existing_results(
        output_path,
        questions,
        scorer.signature,
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
    already_done = processed_questions

    print(f"{level}: resume {processed_questions}/{total_questions}")

    start_time = time.perf_counter()
    chunk_size = 100

    for start in range(0, len(remaining_questions), chunk_size):
        question_batch = remaining_questions[start : start + chunk_size]
        results = scorer.predict_many(question_batch, batch_size=batch_size)

        write_results(output_path, results)

        processed_questions += len(results)
        correct_questions += sum(correctness(result) for result in results)

        elapsed = time.perf_counter() - start_time
        newly_processed = processed_questions - already_done

        print(
            f"{level}: {processed_questions}/{total_questions} "
            f"acc={correct_questions / processed_questions:.4f} "
            f"{newly_processed / elapsed:.1f}/s"
        )

    elapsed = time.perf_counter() - start_time
    newly_processed = processed_questions - already_done

    return {
        "level": level,
        "questions": total_questions,
        "correct": correct_questions,
        "accuracy": correct_questions / total_questions,
        "resumed_from": already_done,
        "newly_processed": newly_processed,
        "elapsed_seconds": elapsed,
        "throughput_questions_per_second": newly_processed / elapsed,
    }


def run_preflight(
    scorer,
    pool: list[KnowledgeQuestion],
    count: int,
    seed: int,
    batch_size: int,
) -> dict:
    """Score a small random sample; abort the run if the setup is clearly broken."""
    sample = random.Random(seed).sample(pool, min(count, len(pool)))
    results = scorer.predict_many(sample, batch_size=batch_size)
    ok, stats, failures, warnings = evaluate_knowledge_preflight(results)
    report = {"stats": stats, "failures": failures, "warnings": warnings}

    if isinstance(scorer, LetterScorer):
        equiv_ok, worst = check_batch_equivalence(scorer, sample[:8])
        report["batch_equivalence"] = {"ok": equiv_ok, "max_abs_logprob_diff": worst}

        if not equiv_ok:
            failures.append(
                f"hasil batch != hasil satu-satu (selisih maks {worst:.3f}): "
                "padding/position_ids bermasalah"
            )
            ok = False

    print(f"preflight: {json.dumps(report, ensure_ascii=False)}")

    for warning in warnings:
        print(f"PREFLIGHT WARNING: {warning}")

    if not ok:
        for failure in failures:
            print(f"PREFLIGHT WARNING: {failure}")

    return report


def build_scorer(args, knowledge_config, model, tokenizer, device):
    if args.score_mode == "letter":
        return LetterScorer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            style=args.letter_style,
            prompt_format=args.prompt_format,
            enable_thinking=knowledge_config.get("chat_enable_thinking", False),
        )

    return KnowledgeScorer(
        model=model, tokenizer=tokenizer, device=device, span=args.span
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the IndoMMLU knowledge benchmark.",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--score-mode",
        choices=["letter", "mean_log_likelihood"],
        default=None,
        help="letter = official protocol (default from config).",
    )
    parser.add_argument("--letter-style", choices=["natural", "official"], default=None)
    parser.add_argument("--prompt-format", choices=["plain", "chat"], default=None)
    parser.add_argument(
        "--span",
        choices=["full", "answer"],
        default=None,
        help="Only for --score-mode mean_log_likelihood (sensitivity).",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Deterministic random subset (own results tag); for sensitivity checks.",
    )
    parser.add_argument("--preflight-questions", type=int, default=None)
    parser.add_argument(
        "--ignore-preflight",
        action="store_true",
        help="Continue even if preflight fails. Deliberate use only.",
    )
    args = parser.parse_args()

    knowledge_config = load_knowledge_config(KNOWLEDGE_CONFIG)
    args.score_mode = args.score_mode or knowledge_config["primary_score_mode"]
    args.letter_style = args.letter_style or knowledge_config.get(
        "letter_style", "natural"
    )
    args.prompt_format = args.prompt_format or knowledge_config.get(
        "prompt_format", "plain"
    )
    args.span = args.span or knowledge_config.get("score_span", "answer")
    batch_size = args.batch_size or knowledge_config["batch_size"]
    preflight_count = (
        args.preflight_questions
        if args.preflight_questions is not None
        else knowledge_config.get("preflight_questions", 64)
    )

    seed = load_runtime_seed(RUNTIME_CONFIG)
    seed_everything(seed)

    registry = ModelRegistry.from_yaml(MODELS_CONFIG)
    spec = registry.get(args.model_id)
    spec.require_pinned_identity()

    if args.score_mode == "letter":
        signature = ("letter", args.letter_style, args.prompt_format)
    else:
        signature = ("mean_log_likelihood", args.span)

    tag = result_tag(signature, args.limit)
    model_dir = args.results_dir.resolve() / spec.model_id / tag
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"knowledge {spec.model_id} {tag} batch={batch_size}")

    questions = load_indommlu_csv(DATA_PATH)
    zero_shot_questions = [q for q in questions if not q.is_for_fewshot]

    if args.limit:
        keep = {
            q.question_id
            for q in random.Random(seed).sample(zero_shot_questions, args.limit)
        }
        zero_shot_questions = [q for q in zero_shot_questions if q.question_id in keep]

    primary_questions = [
        q for q in zero_shot_questions if q.level in knowledge_config["primary_levels"]
    ]
    retention_questions = [
        q
        for q in zero_shot_questions
        if q.level in knowledge_config["retention_levels"]
    ]

    print(
        f"n_zero={len(zero_shot_questions)} "
        f"n_primary={len(primary_questions)} "
        f"n_retention={len(retention_questions)}"
    )

    tokenizer, model, device = ModelLoader.load(spec)
    dtype = next(model.parameters()).dtype

    print(f"device={device} dtype={dtype}")

    scorer = build_scorer(args, knowledge_config, model, tokenizer, device)
    preflight_report = None

    if preflight_count > 0 and not args.ignore_preflight:
        preflight_report = run_preflight(
            scorer,
            primary_questions + retention_questions,
            preflight_count,
            seed,
            batch_size,
        )
    elif preflight_count > 0:
        print("PREFLIGHT dilewati (--ignore-preflight).")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    benchmark_start = time.perf_counter()

    primary_summary = run_level(
        scorer,
        primary_questions,
        "SD+SMP",
        model_dir / "primary_sd_smp.jsonl",
        batch_size,
    )
    retention_summary = run_level(
        scorer,
        retention_questions,
        "SMA",
        model_dir / "retention_sma.jsonl",
        batch_size,
    )

    benchmark_elapsed = time.perf_counter() - benchmark_start
    peak_memory_gb = None

    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    scoring = {
        "score_mode": args.score_mode,
        "score_span": args.span if args.score_mode == "mean_log_likelihood" else None,
        "letter_style": args.letter_style if args.score_mode == "letter" else None,
        "prompt_format": args.prompt_format if args.score_mode == "letter" else None,
        "batch_size": batch_size,
        "exclude_fewshot": knowledge_config["exclude_fewshot"],
        "limit": args.limit,
    }

    if isinstance(scorer, LetterScorer):
        scoring["letter_scorer"] = scorer.describe()

    metadata = {
        "git_commit": get_git_commit(PROJECT_ROOT),
        "seed": seed,
        "model_id": spec.model_id,
        "model_source": spec.source,
        "model_revision": spec.revision,
        "model_weights_sha256": spec.weights_sha256,
        "device": str(device),
        "dtype": str(dtype),
        "scoring": scoring,
        "preflight": preflight_report,
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

    with (model_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(
        f"primary={primary_summary['accuracy']:.4f} sma={retention_summary['accuracy']:.4f}"
    )
    print(f"{benchmark_elapsed / 60:.1f} min -> {model_dir}")

    del scorer
    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
