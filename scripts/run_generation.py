import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

from edu_eval.generation.cases import (
    load_cases,
    select_full,
    select_pilot,
    validation_report,
)
from edu_eval.generation.config import GenerationConfig
from edu_eval.generation.controlled import (
    ControlledGenerationResult,
    ControlledGenerationRunner,
)
from edu_eval.generation.runner import GenerationRunner
from edu_eval.generation.schema import ControlledGenerationCase
from edu_eval.models.loader import ModelLoader
from edu_eval.models.registry import ModelRegistry
from edu_eval.runtime.env import (
    collect_runtime,
    get_git_commit,
    seed_everything,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"
GENERATION_CONFIG = PROJECT_ROOT / "configs" / "generation.yaml"
RUNTIME_CONFIG = PROJECT_ROOT / "configs" / "runtime.yaml"
DEFAULT_CASES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "controlled_generation_cases.jsonl"
)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "generation"


def expected_prompt_id(
    prompt_version: str,
    case_id: str,
    target_grade: str,
) -> str:
    return hashlib.sha256(
        f"{prompt_version}:{case_id}:{target_grade}".encode()
    ).hexdigest()[:16]


def read_existing_keys(path: Path, prompt_version: str) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()

    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}.") from exc

            key = (record["case_id"], record["target_grade"])

            if record.get("prompt_id") != expected_prompt_id(
                prompt_version,
                record["case_id"],
                record["target_grade"],
            ):
                raise RuntimeError(
                    f"prompt_id mismatch at {path}:{line_number} for {key}: "
                    "existing output was generated under a different "
                    "prompt_version. Move it aside before re-running."
                )

            done.add(key)

    return done


def write_results(
    path: Path,
    results: Sequence[ControlledGenerationResult],
) -> None:
    with path.open("a", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled grade-level generation.",
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="Model ID from configs/models.yaml.",
    )

    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="pilot",
        help="pilot = N cases/subject (default), full = all cases.",
    )

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Canonical cases JSONL.",
    )

    parser.add_argument(
        "--generation-config",
        type=Path,
        default=GENERATION_CONFIG,
        help="Generation YAML config.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for generation results.",
    )

    args = parser.parse_args()

    generation_config = GenerationConfig.from_yaml(args.generation_config)

    with RUNTIME_CONFIG.open("r", encoding="utf-8") as file:
        seed = yaml.safe_load(file)["runtime"]["seed"]

    seed_everything(seed)

    registry = ModelRegistry.from_yaml(MODELS_CONFIG)
    spec = registry.get(args.model_id)

    cases = load_cases(args.cases)
    report = validation_report(cases, generation_config)

    print(f"generation {spec.model_id} {args.mode} cases={report['total_cases']}")

    selected: list[ControlledGenerationCase]
    if args.mode == "pilot":
        selected = select_pilot(cases, generation_config)
    else:
        if report["full_ready"] != "PASS":
            raise RuntimeError(
                "Full dataset not ready: "
                f"{report['subjects']} vs "
                f"{generation_config.full_cases_per_subject}/subject."
            )
        selected = select_full(cases, generation_config)

    print(f"selected={len(selected)}")

    results_root = args.results_dir.resolve()
    model_dir = results_root / args.mode / spec.model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    output_path = model_dir / "controlled_outputs.jsonl"

    done = read_existing_keys(output_path, generation_config.prompt_version)
    print(f"resume={len(done)}")

    tokenizer, model, device = ModelLoader.load(spec)
    dtype = next(model.parameters()).dtype
    print(f"device={device} dtype={dtype}")

    generation_runner = GenerationRunner(
        model=model,
        tokenizer=tokenizer,
        device=device,
        config=generation_config,
    )

    controlled_runner = ControlledGenerationRunner(
        generation_runner=generation_runner,
        model_id=spec.model_id,
        target_grades=list(generation_config.target_grades),
        prompt_version=generation_config.prompt_version,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_expected = len(selected) * len(generation_config.target_grades)
    newly_done = 0
    truncated = 0
    benchmark_start = time.perf_counter()

    for case in selected:
        for grade in generation_config.target_grades:
            if (case.case_id, grade) in done:
                continue

            result = controlled_runner.generate(case, grade)
            key = (result["case_id"], result["target_grade"])
            write_results(output_path, [result])
            done.add(key)
            newly_done += 1
            truncated += int(result["hit_max_new_tokens"])
            print(f"{len(done)}/{total_expected} {key[0]} {key[1]}")

    benchmark_elapsed = time.perf_counter() - benchmark_start
    peak_memory_gb = None
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    device_name = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    metadata = {
        "git_commit": get_git_commit(PROJECT_ROOT),
        "seed": seed,
        "model_id": spec.model_id,
        "model_source": spec.source,
        "model_revision": spec.revision,
        "mode": args.mode,
        "device": str(device),
        "device_name": device_name,
        "dtype": str(dtype),
        "generation": {
            "max_new_tokens": generation_config.max_new_tokens,
            "do_sample": generation_config.do_sample,
            "use_cache": generation_config.use_cache,
            "prompt_version": generation_config.prompt_version,
            "target_grades": list(generation_config.target_grades),
        },
        "dataset": {
            "cases_path": str(args.cases.resolve()),
            "selected_cases": len(selected),
            "expected_outputs": total_expected,
        },
        "run": {
            "resumed_outputs": len(done) - newly_done,
            "newly_processed": newly_done,
            "total_outputs": len(done),
            "truncated_outputs": truncated,
            "elapsed_seconds": benchmark_elapsed,
            "elapsed_minutes": benchmark_elapsed / 60,
        },
        "peak_gpu_memory_gb": peak_memory_gb,
        "runtime": collect_runtime(),
        "validation": report,
    }

    with (model_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(
        f"{len(done)}/{total_expected} trunc={truncated} {benchmark_elapsed / 60:.1f} min"
    )

    del controlled_runner
    del generation_runner
    del model
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
