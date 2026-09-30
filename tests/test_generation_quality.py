import pytest

from edu_eval.generation.config import GenerationConfig
from edu_eval.generation.quality import (
    evaluate_preflight,
    output_flags,
    repetition_ratio,
    run_preflight,
)
from edu_eval.generation.template_check import require_thinking_off, template_report
from edu_eval.models.schema import ModelSpec


def test_flags_catch_the_real_failure_modes():
    assert output_flags("Hmm, pengguna meminta materi tentang fotosintesis", True)[
        "thinking_leak"
    ]
    assert output_flags("-\n-\n-\n-\n-\n-\nx", False)["dash_spam"]
    assert output_flags(
        "Materi ini dapat digunakan untuk berbagai jenjang. " * 12, False
    )["repetitive"]
    assert output_flags("", False)["empty"]


def test_clean_material_passes_all_flags():
    flags = output_flags(
        "Fotosintesis adalah proses tumbuhan membuat makanan dengan bantuan cahaya matahari.",
        False,
    )

    assert not any(
        flags[k]
        for k in ("empty", "truncated", "thinking_leak", "dash_spam", "repetitive")
    )


def test_repetition_ratio_bounds():
    assert repetition_ratio("a b c d e f g") == 0.0
    assert repetition_ratio("a b c d " * 10) > 0.8


def _result(text, hit=False):
    return {
        "case_id": "c",
        "target_grade": "SD6",
        "output_text": text,
        "hit_max_new_tokens": hit,
    }


def test_preflight_rejects_pilot_like_output():
    ok, _, reasons = evaluate_preflight([_result("Hmm, pengguna meminta x", True)] * 4)

    assert not ok and len(reasons) >= 2


def test_preflight_accepts_good_output():
    ok, _, reasons = evaluate_preflight(
        [_result("Tumbuhan membuat makanan sendiri.")] * 4
    )

    assert ok and reasons == []


class _Case:
    def __init__(self, case_id):
        self.case_id = case_id


def test_run_preflight_skips_done_and_returns_unwritten_results():
    calls = []

    def generate(case, grade):
        calls.append((case.case_id, grade))

        return {
            "case_id": case.case_id,
            "target_grade": grade,
            "output_text": "Isi materi yang wajar.",
            "hit_max_new_tokens": False,
        }

    results, ok, _, _ = run_preflight(
        generate, [_Case("c1")], ["SD6", "SMP7"], done={("c1", "SD6")}
    )

    assert ok and calls == [("c1", "SMP7")] and len(results) == 1


class _Tok:
    def __init__(self, ends_open):
        self.chat_template = "enable_thinking"
        self.ends_open = ends_open

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        if kw.get("enable_thinking") is False and not self.ends_open:
            return "<|im_start|>assistant\n<think>\n\n</think>\n\n"

        return "<|im_start|>assistant\n<think>\n"


def test_template_check_passes_when_flag_is_honoured():
    report = require_thinking_off(_Tok(ends_open=False), False)

    assert report["open_think_block"] is False


def test_template_check_fails_when_flag_is_ignored():
    with pytest.raises(RuntimeError):
        require_thinking_off(_Tok(ends_open=True), False)

    assert template_report(_Tok(True), None)["open_think_block"] is True


def test_generation_config_has_thinking_off_and_v2_prompt():
    config = GenerationConfig.from_yaml("configs/generation.yaml")

    assert config.enable_thinking is False
    assert config.prompt_version == "cg_v2"


def _spec(**kw):
    base = {
        "model_id": "m",
        "model_role": "research",
        "source": "org/name",
        "stage": "base",
    }
    base.update(kw)

    return ModelSpec(**base)


def test_research_model_requires_pinned_revision():
    with pytest.raises(ValueError):
        _spec().require_pinned_identity()

    _spec(revision="abc123").require_pinned_identity()


def test_local_source_requires_weights_hash():
    with pytest.raises(ValueError):
        _spec(source="/content/m4_fixed", revision="abc").require_pinned_identity()

    _spec(
        source="/content/m4_fixed", weights_sha256="deadbeef"
    ).require_pinned_identity()


def test_non_research_role_is_not_forced():
    _spec(model_role="smoke_test", stage="smoke").require_pinned_identity()
