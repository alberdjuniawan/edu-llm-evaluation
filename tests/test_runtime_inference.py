import os
import random
from pathlib import Path

import pytest

from edu_eval.inference.benchmark import FirstTokenStreamer
from edu_eval.knowledge.scorer import KnowledgeScorer
from edu_eval.runtime.env import (
    collect_runtime,
    get_git_commit,
    seed_everything,
)


def test_seed_everything_is_deterministic():
    seed_everything(42)
    first = [random.random() for _ in range(3)]

    seed_everything(42)
    second = [random.random() for _ in range(3)]

    assert first == second


def test_get_git_commit_returns_none_or_hex():
    commit = get_git_commit(Path("."))

    assert commit is None or (
        len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    )


def test_collect_runtime_has_expected_keys():
    runtime = collect_runtime()

    assert "torch" in runtime
    assert "cuda_available" in runtime
    assert "transformers" in runtime


def test_first_token_streamer_ttft():
    streamer = FirstTokenStreamer()

    assert streamer.ttft_seconds() is None

    streamer.put([1, 2, 3])

    assert streamer.ttft_seconds() is not None
    assert streamer.ttft_seconds() >= 0.0


def test_predict_from_scores_picks_best_mean():
    scores = [
        {
            "choice_index": 0,
            "log_likelihood": -10.0,
            "mean_log_likelihood": -2.0,
            "token_count": 5,
        },
        {
            "choice_index": 1,
            "log_likelihood": -4.0,
            "mean_log_likelihood": -1.0,
            "token_count": 4,
        },
    ]

    prediction = KnowledgeScorer._predict_from_scores(scores, gold_index=1)

    assert prediction["predicted_index"] == 1
    assert prediction["correct"] is True
    assert prediction["margin"] == pytest.approx(1.0)


def _smoke_model_cached() -> bool:
    cache_roots = [
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")),
        Path.home() / ".cache" / "huggingface",
    ]

    return any(
        (root / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct").exists()
        for root in cache_roots
    )


@pytest.mark.skipif(
    not _smoke_model_cached(),
    reason="smoke model not in local HF cache",
)
def test_inference_benchmark_smoke():
    import torch

    from edu_eval.inference.benchmark import benchmark_generation
    from edu_eval.models.loader import ModelLoader
    from edu_eval.models.registry import ModelRegistry

    registry = ModelRegistry.from_yaml(Path("configs/models.yaml"))
    spec = registry.get("smoke_qwen")
    tokenizer, model, device = ModelLoader.load(spec)

    try:
        result = benchmark_generation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=["Halo, apa kabar?"],
            max_new_tokens=4,
        )
    finally:
        del model
        del tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert result["n_prompts"] == 1
    assert result["total_output_tokens"] > 0
    assert result["tokens_per_second"] > 0
