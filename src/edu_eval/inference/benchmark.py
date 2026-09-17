import time
from typing import Any, TypedDict

import torch
from transformers.generation.streamers import BaseStreamer


class FirstTokenStreamer(BaseStreamer):
    def __init__(self) -> None:
        self.start: float | None = None
        self.first_token_time: float | None = None

    def begin(self) -> None:
        self.start = time.perf_counter()
        self.first_token_time = None

    def put(self, value: Any) -> None:
        if self.start is None:
            self.start = time.perf_counter()

        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def end(self) -> None:
        return None

    def ttft_seconds(self) -> float | None:
        if self.start is None or self.first_token_time is None:
            return None

        return self.first_token_time - self.start


class BenchmarkResult(TypedDict):
    n_prompts: int
    max_new_tokens: int
    total_output_tokens: int
    total_seconds: float
    tokens_per_second: float
    mean_latency_seconds: float
    mean_ttft_seconds: float | None
    peak_vram_gb: float | None


def benchmark_generation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    max_new_tokens: int = 32,
) -> BenchmarkResult:
    if not prompts:
        raise ValueError("prompts must not be empty.")

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    latencies: list[float] = []
    ttfts: list[float] = []
    total_tokens = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_start = time.perf_counter()

    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(device)

            streamer = FirstTokenStreamer()
            streamer.begin()
            start = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                streamer=streamer,
            )
            latencies.append(time.perf_counter() - start)

            sequences = getattr(outputs, "sequences", outputs)
            total_tokens += int(sequences.shape[1] - inputs["input_ids"].shape[1])

            ttft = streamer.ttft_seconds()

            if ttft is not None:
                ttfts.append(ttft)

    total_seconds = time.perf_counter() - total_start
    peak_vram_gb = None

    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "n_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "total_output_tokens": total_tokens,
        "total_seconds": total_seconds,
        "tokens_per_second": total_tokens / max(total_seconds, 1e-9),
        "mean_latency_seconds": sum(latencies) / len(latencies),
        "mean_ttft_seconds": (sum(ttfts) / len(ttfts) if ttfts else None),
        "peak_vram_gb": peak_vram_gb,
    }
