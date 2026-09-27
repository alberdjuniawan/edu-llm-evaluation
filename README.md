# edu-llm-evaluation

Baseline evaluation for Indonesian educational LLMs: does an existing
checkpoint have the knowledge, grade-level alignment, and pedagogical
quality to generate SD/SMP learning content? New training is a diagnostic
response, never the default next step.

## Layout

```text
configs/   experiment settings (models, knowledge, generation, …)
src/edu_eval/   reusable packages (models, knowledge, generation,
                linguistic, human_eval, statistics, inference, runtime)
scripts/   reproducible full runs (resumable, incremental JSONL)
notebooks/ 00–07  pilot runs and analysis
tests/     technical validation
data/      raw inputs + canonical 60-case dataset
results/   experiment outputs (gitignored)
human_eval/   offline rater app (index.html)
```

Rule: notebook = pilot, script = full run, tests = validation.
Swap model = add entry to `configs/models.yaml`, pass `--model-id`.

## Quickstart

```bash
uv sync
uv run pytest -q
uv run python scripts/build_controlled_cases.py
uv run python scripts/run_knowledge.py --model-id smoke_qwen
uv run python scripts/run_generation.py --model-id smoke_qwen --mode pilot
uv run python scripts/analyze_results.py --model-id smoke_qwen
```

## Status

Done: pipeline engineering, smoke runs (knowledge 12,860 Q, generation
40/40), 60-case dataset, rater tooling, 68 tests green.
Pending: 5 research checkpoints (need ≥24GB GPU), SME case review,
protocol freeze, human eval, gates + Pareto, inference optimization.
