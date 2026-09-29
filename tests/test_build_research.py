import importlib.util
from pathlib import Path

import pytest


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _payload(case_id="CG-001", scope="grade_specific", source="official_textbook"):
    locator = "loc" if scope == "grade_specific" else None
    url = None if source == "project_capsule" else "https://example.com/x"

    return {
        "case_id": case_id,
        "subject": "IPA",
        "concept": "c",
        "reference": {
            "source_id": "r",
            "source_title": "t",
            "source_url": url,
            "source_type": source,
            "source_version": None,
            "locator": "p.1" if source != "project_capsule" else None,
            "text": "x",
        },
        "target_evidence": {
            target: {
                "curriculum_phase": phase,
                "grade": grade,
                "evidence_source": "s",
                "evidence_locator": locator,
                "evidence_text": "t",
                "evidence_scope": scope,
            }
            for target, phase, grade in (
                ("SD6", "C", 6),
                ("SMP7", "D", 7),
                ("SMP9", "D", 9),
                ("SMA10", "E", 10),
            )
        },
    }


def test_template_validates_as_research():
    module = _load("build_research_cases.py")
    template = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "schemas"
        / "curation_template.jsonl"
    )
    cases = module.load_cases(template)

    assert len(cases) == 1
    assert "CONTOH" in cases[0].case_id


def test_capsule_rejected():
    module = _load("build_research_cases.py")

    with pytest.raises(ValueError, match="pilot-only"):
        module.check_research_record(
            _payload(case_id="CG-001", source="project_capsule"), 1
        )


def test_phase_only_evidence_rejected():
    module = _load("build_research_cases.py")
    payload = _payload()
    payload["target_evidence"]["SMP7"]["evidence_scope"] = "phase_only"

    with pytest.raises(ValueError, match="grade_specific"):
        module.check_research_record(payload, 3)


def test_missing_input_rejected(tmp_path, monkeypatch):
    module = _load("build_research_cases.py")
    monkeypatch.setattr(
        "sys.argv",
        ["build_research_cases.py", "--input", str(tmp_path / "absent.jsonl")],
    )

    with pytest.raises(FileNotFoundError):
        module.main()


def test_successful_build_marks_research_ready(tmp_path, monkeypatch):
    import json

    module = _load("build_research_cases.py")
    subjects = [
        "IPA",
        "Bahasa Indonesia",
        "Pendidikan Pancasila",
        "Informatika",
        "IPS",
    ]
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    lines = []

    for index, subject in enumerate(subjects):
        payload = _payload(case_id=f"CG-{index:03d}")
        payload["subject"] = subject
        lines.append(json.dumps(payload, ensure_ascii=False))

    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_research_cases.py",
            "--input",
            str(src),
            "--output",
            str(dst),
            "--min-per-subject",
            "1",
        ],
    )
    module.main()

    cases = module.load_cases(dst)

    assert len(cases) == 5
