import importlib.util
import json
from pathlib import Path


def _load():
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "migrate_knowledge_results.py"
    )
    spec = importlib.util.spec_from_file_location("migrate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_migrates_legacy_layout_and_is_idempotent(tmp_path):
    mod = _load()
    model = tmp_path / "m0"
    model.mkdir()
    (model / "primary_sd_smp.jsonl").write_text("{}\n")
    (model / "retention_sma.jsonl").write_text("{}\n")
    (model / "run_metadata.json").write_text(
        json.dumps(
            {"scoring": {"score_mode": "mean_log_likelihood", "score_span": "answer"}}
        )
    )

    first = mod.migrate(tmp_path)

    assert len(first) == 3
    assert (model / "mean_log_likelihood-answer" / "primary_sd_smp.jsonl").exists()
    assert not (model / "primary_sd_smp.jsonl").exists()
    assert mod.migrate(tmp_path) == []
