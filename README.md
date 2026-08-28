# edu-llm-evaluation

Baseline evaluation for Indonesian educational LLMs (Tim 2): does an
existing checkpoint have the knowledge, grade-level alignment, and
pedagogical quality to generate SD/SMP learning content — before any
inference optimization? New training is a diagnostic response, never the
default next step.

Design ground truth: `edu_llm_evaluation_full_pipeline.md` (+
`controlled_generation_research_spec.md` for the controlled-generation
component). Frozen protocol: `docs/evaluation_protocol.md`. Data honesty
rules: `docs/data_provenance.md`. Run history: `docs/experiment_log.md`.

## Research map

| RQ | Question | Evidence | Code |
|---|---|---|---|
| RQ1 | Enough SD/SMP knowledge? | IndoMMLU SD+SMP (+SMA control) | `knowledge/`, `run_knowledge.py` |
| RQ2 | Same concept at the right level? | controlled generation SD6/SMP7/SMP9/SMA10 | `generation/`, `run_generation.py` |
| RQ3 | Can we measure it reliably? | diagnostics + human + stats | `linguistic/`, `human_eval/`, `statistics/` |
| RQ4 | Which checkpoint becomes baseline? | gates G1–G4 + Pareto | `statistics/`, nb `06` |
| RQ5 | Quality–efficiency trade-off? | precision/batch/cache/serving | `inference/`, nb `07` |

Guardrails: readability ≠ grade fit · knowledge ≠ pedagogy · stage-wise
comparison, never causal training claims · blinded human eval ·
LLM-judge only secondary · no latency win without quality regression.

## Layout

```text
configs/   models.yaml  runtime.yaml  knowledge.yaml  generation.yaml
           linguistic.yaml  human_eval.yaml  inference.yaml
docs/      evaluation_protocol.md  experiment_log.md  data_provenance.md
data/      raw/  interim/  processed/controlled_generation_cases.jsonl  schemas/
src/edu_eval/  runtime/ models/ knowledge/ generation/ linguistic/
               human_eval/ statistics/ inference/
notebooks/ 00_environment_audit  01_model_audit  02_knowledge_pilot
           03_controlled_generation_pilot  04_linguistic_analysis
           05_human_eval_analysis  06_baseline_selection  07_inference_benchmark
scripts/   run_knowledge.py  run_generation.py  run_linguistic.py
           analyze_results.py  run_experiment.py  build_controlled_cases.py
tests/     results/
```

Rule: **notebook** = pilot/narrative · **script** = full reproducible run ·
**tests** = technical validation. Code files are intentionally comment-free;
all rationale lives here and in `docs/`.

## Quickstart

```bash
uv sync
uv run pytest -q                                   # technical validation
uv run python scripts/build_controlled_cases.py    # build + validate 60 cases

# Full runs (one model at a time, resumable, incremental JSONL)
uv run python scripts/run_knowledge.py --model-id smoke_qwen
uv run python scripts/run_generation.py --model-id smoke_qwen --mode pilot   # 40 outputs
uv run python scripts/run_generation.py --model-id smoke_qwen --mode full    # 240 outputs
uv run python scripts/run_linguistic.py --inputs results/generation/pilot/smoke_qwen/controlled_outputs.jsonl
uv run python scripts/analyze_results.py --model-id smoke_qwen
uv run python scripts/run_experiment.py --step knowledge -- --model-id smoke_qwen
```

Swap model = add entry to `configs/models.yaml`, pass `--model-id`. No code
changes needed.

## Frozen protocol (details)

Decoding (primary): `max_new_tokens: 256` (ceiling, not target),
`do_sample: false`, `use_cache: true`, `prompt_version: cg_v1`. No
`min_new_tokens`, no forced vocabulary, no sentence-count constraints.

Knowledge: likelihood-based MCQ, mean log-likelihood per candidate
(`mean_log_likelihood`). Primary SD+SMP, retention SMA, few-shot excluded.
Pre-rename artifacts using the `official` key still resume.

Prompt `cg_v1`:

```text
Buat materi pembelajaran berbahasa Indonesia berdasarkan materi sumber berikut.

Mata pelajaran: {subject}
Konsep: {concept}
Target peserta didik: {level} kelas {grade}

Materi sumber:
{reference_text}

Tugas:
Jelaskan konsep tersebut sebagai materi pembelajaran yang sesuai untuk target
peserta didik. Gunakan materi sumber sebagai landasan faktual dan jangan
bertentangan dengan materi tersebut. Sesuaikan tingkat bahasa, kedalaman
konsep, dan cara penjelasan dengan target peserta didik. Jangan menyebutkan
instruksi ini atau target grade secara eksplisit dalam jawaban.
```

`prompt_id = sha256("{prompt_version}:{case_id}:{target_grade}")[:16]`.

Generation output record: `model_id, case_id, source_id, subject, concept,
task_type, target_grade, target_level, prompt_id, prompt_version,
generation_config{max_new_tokens, do_sample, use_cache}, output_text,
input_tokens, output_tokens, latency_seconds, hit_max_new_tokens`.

Knowledge prediction record: `question_id, level, grade, subject,
gold_index, choice_scores[], predictions.mean_log_likelihood{
predicted_index, correct, gold_score, margin }`.

Every `run_metadata.json` records `git_commit, seed, model_id,
model_source, model_revision, dataset counts, generation config, runtime
versions, hardware, elapsed, peak VRAM`.

## Dataset

One canonical file, 60 cases (12 × IPA, Bahasa Indonesia, Pendidikan
Pancasila, Informatika, IPS), 4 target grades each. Phase mapping enforced
per case: SD6→C/6, SMP7→D/7, SMP9→D/9, SMA10→E/10 (SMP7/SMP9 share Phase D
by design). References are honestly labeled project-authored capsules
(`source_type: project_capsule`, no URL); grade evidence is phase-level
(`evidence_scope: phase_only`) until reviewed. Validators structurally
reject fabricated official provenance (official sources require URL +
locator; `grade_specific` claims require a locator). Subject-matter review
is required before human evaluation.

## Notebooks

| Nb | Purpose | Output |
|---|---|---|
| 00 | environment audit | console |
| 01 | model audit (loader, dtype, chat template) | console |
| 02 | knowledge pilot (93 stratified questions) | `results/knowledge_pilot/` |
| 03 | controlled generation pilot (10 cases × 4) | `results/generation/pilot/<model>/` |
| 04 | linguistic diagnostics on pilot outputs | `results/linguistic/` |
| 05 | blinded human-eval pack (labels + shuffled order) | `results/human_eval/` |
| 06 | gates G1–G4 status + bootstrap CI | console |
| 07 | inference micro-benchmark (BF16 reference) | `results/inference/` |

`_blind_mapping.json` is analysis-only and must never reach raters.

## Penilaian manusia (rater)

1. `notebooks/05` membangun `results/human_eval/pilot_pack.jsonl` (label
   buta + urutan acak + teks rujukan) — tanpa identitas model.
2. Rater membuka `human_eval/index.html` di browser (offline, tanpa
   install): isi ID, muat file pack, nilai 6 dimensi skala 1–5, unduh
   `ratings-<id>.jsonl`. Jawaban tersimpan otomatis di browser.
3. File ratings tervalidasi oleh skema `RatingRecord`
   (`src/edu_eval/human_eval/protocol.py`) lalu dianalisis (alpha,
   ordinal, Kendall tau).

## Pilot acceptance checklist (nb 03)

1. Valid outputs for all four grades
2. Factual reference preserved across grades
3. Not verbatim copies of the reference
4. Linguistic/conceptual variation across grades
5. No prompt/grade instruction leak
6. Truncation rare or understood
7. No inherently invalid case for a target grade
8. Factual errors judgeable against the reference
9. Scorable without hidden project knowledge
10. Suitable for automatic linguistic analysis

Reject cases by logging, never by silent replacement.

## Pilot findings (smoke model, 40/40)

- 38/40 output mencapai ceiling 256 token. Penyebab belum dibuktikan;
  `max_new_tokens=256` berpotensi terlalu ketat untuk task materi dan
  wajib diverifikasi sebelum full research run.
- Potential mention 27/40 (grade_mention 27, instruction_disclosure 6,
  meta 0; contoh "siswa SMA kelas 7" untuk target SMP7). Klasifikasi
  heuristik, wajib inspeksi manual; grade mention natural belum tentu
  instruction leakage.

## Research checkpoints (Colab)

Panduan lengkap: `docs/colab_research_plan.md`. Ringkasnya: butuh GPU
≥24GB (A100 40GB aman), daftarkan checkpoint di `configs/models.yaml`
dengan revision hash asli, audit 1 checkpoint dulu (cek
`trust_remote_code`), run satu model per sesi dengan `--results-dir` ke
Google Drive (resume otomatis), lalu analisis + human eval di laptop.

## Status (definition of done)

Done: env/model audit · knowledge scorer + smoke dry run + resume runner ·
60-case dataset + validation · generation runner + pilot (40/40 smoke) ·
linguistic/human-eval/statistics/inference packages + configs + tests.
Pending: 5 research checkpoints audit · full 9B benchmarks (target
environment ≥24GB VRAM untuk artifact BF16 ~18–19GB + overhead) · SME
review 60 kasus (technical-ready → research-ready) · protocol freeze ·
human eval + power analysis · gates/Pareto · inference optimization
+ quality regression.

Tests/mypy/ruff = software correctness, bukan validitas ilmiah.
