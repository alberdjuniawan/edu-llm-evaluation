# edu-llm-evaluation

Baseline evaluation for Indonesian educational LLMs: do existing
checkpoints have the knowledge, grade-level alignment, and pedagogical
quality to generate SD/SMP learning content? New training is a diagnostic
response, never the default next step.

This repo proves pipelines, not conclusions. Numbers from smoke/staging
models validate that the machinery works; research claims about 9B
checkpoints require the full protocol (Section 11).

## 1. Prerequisites

- Python 3.12, `uv` (local) or plain `pip` (Colab — there is no `uv`
  there, replace `uv run python` with `!python`)
- GPU sized to the model: 0.5B runs on 4GB; 3B staging needs ~8GB
  (T4 15GB is comfortable); 9B BF16 targets ≥24GB VRAM
- Hugging Face access for gated checkpoints (`HF_TOKEN`)
- `data/raw/IndoMMLU.csv` (14,981 rows; gitignored, never edited)

## 2. Install

```bash
git clone <repo-url> && cd edu-llm-evaluation
uv sync
uv run pytest -q -m "not integration"   # fast unit suite, no data needed
```

Colab notes: after `pip install -e .`, **restart the runtime once**
(editable path hooks load at startup); keep Colab's preinstalled torch
instead of forcing the pinned version; mount Drive and point every
`--results-dir` at it because sessions die.

## 3. Register any model

Add one entry to `configs/models.yaml` — no code changes, ever:

```yaml
  - model_id: my_model
    model_role: research        # smoke_test | research
    stage: cpt                  # smoke | pretrained | cpt | sft_v1 | sft_v2 | merged_sft
    source: org/model-repo      # HF id or local path
    revision: <commit-hash>     # required for research runs
    lineage_status: verified    # verified | unresolved (default)
    precision: bf16             # bf16 | fp16 | fp32 (default: bf16 on CUDA)
```

Rules: never invent IDs or revisions; `lineage_status` stays
`unresolved` until source + revision are confirmed; every run takes
`--model-id`.

## 4. Data prep

```bash
uv run python scripts/build_controlled_cases.py   # 60 cases, validated before write
```

`data/raw/` is frozen input. The canonical dataset is exactly one file
(`data/processed/controlled_generation_cases.jsonl`); pilot/full are
config subsets, never separate files.

## 5. Run order

```bash
# 0. audit (downloads weights on first run)
# 1. knowledge, full (resumable; rerun same command after interruption)
uv run python scripts/run_knowledge.py --model-id <id> --span answer
# 2. span sensitivity on a sample (is answer-span the right primary?)
uv run python scripts/compare_spans.py --model-id <id> --count 120
# 3. generation pilot (40 outputs) — read truncation/mention rates first
uv run python scripts/run_generation.py --model-id <id> --mode pilot
# 4. generation full (240 outputs) — refuses to run unless research_ready == PASS
uv run python scripts/run_generation.py --model-id <id> --mode full
# 5. diagnostics + analysis (no GPU needed)
uv run python scripts/run_linguistic.py --inputs <outputs.jsonl>
uv run python scripts/analyze_results.py --model-id <id> [--model-id <id2>]
uv run python scripts/check_protocol.py <run_metadata_1.json> [<run_metadata_2.json> ...]
```

Go/no-go: pilot truncation mass-hitting the 256 ceiling → resolve ceiling
(`--max-new-tokens` override + mini-pilot) before full; `check_protocol`
ERROR → runs are not comparable, do not compare them.

## 6. Human track (two modes)

- **Solo mode (author calibration):** rate 20–40 items yourself to debug
  the rubric, validate the app/export/schema chain, and reject bad
  cases. Label it as author calibration. It unblocks protocol freeze —
  it is NOT grade-fit evidence.
- **Panel mode (research evidence):** notebook 05 builds the blinded
  pack (`pilot_pack.jsonl`, no model identity; mapping stays with
  analysts) → raters use `human_eval/index.html` offline → exports
  validate against `RatingRecord` → reliability (Krippendorff),
  ordinal analysis, Kendall tau. Optional weak substitute when a panel
  is impossible: intra-rater test-retest, reported as a lower bound.

## 7. Config reference

| File | Key fields | Frozen? |
|---|---|---|
| `models.yaml` | source, revision, lineage, precision | revision required for research |
| `runtime.yaml` | seed | yes (42) |
| `knowledge.yaml` | score_mode, score_span (`answer`), levels, batch_size | span + levels yes |
| `generation.yaml` | max_new_tokens 256, greedy, `prompt_version: cg_v1`, grades, subjects, pilot/full counts | prompt + decoding yes |
| `linguistic.yaml` | min chars, english-formula flag | formulas are diagnostic-only |
| `human_eval.yaml` | 1–5 scale, dimensions, blinding seed, pairwise fraction, calibration size | yes |
| `inference.yaml` | precisions, batch sizes, TTFT/TPOT flags | reference BF16 first |

Change frozen values deliberately (protocol decision + log), never
mid-benchmark.

## 8. Results layout

```text
results/knowledge/<model>/{primary_sd_smp,retention_sma}.jsonl + run_metadata.json
results/generation/<pilot|full>/<model>/controlled_outputs.jsonl + run_metadata.json
results/linguistic/<name>.jsonl  results/human_eval/pilot_pack.jsonl + ratings-*.jsonl
results/inference/<model>_<precision>.json
```

Every `run_metadata.json` carries model/revision, dataset + config
hashes, prompt version, seed, commit, runtime, timings, VRAM. Resume
reuses stale-safe identity: changed revision/dataset/config aborts
instead of mixing outputs.

## 9. Troubleshooting

- OOM → lower `batch_size` (math is batch-parity tested, results identical)
- Session died → rerun the same command (incremental JSONL resumes)
- `ModuleNotFoundError` after install → restart runtime once
- `nano` missing / double-click downloads → edit via Python cell or re-upload
- Token 403 on clone → token needs repo scope (classic) or repo selected + Contents read (fine-grained)
- `ModuleNotFoundError: edu_eval` on Colab → `sys.path.insert(0, '<repo>/src')` or restart runtime
- Legacy `official` score key → still read; old metadata without new keys → warning, not error

## 10. Gates and status

Done: pipeline engineering, smoke runs (knowledge 12,860 Q, generation
40/40), 60-case dataset, rater tooling, 123 tests green.
Pending: 3 research checkpoints M0 base / M1 CPT / M2 SR-All
(need ≥24GB GPU; revisions frozen at audit, comparisons stage-wise
only, never causal), SME case review,
protocol freeze, panel human eval, gates G1–G4 + Pareto, inference
optimization + quality regression. No weighted single score, ever.
