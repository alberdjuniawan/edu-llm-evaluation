import importlib.util
import json
from pathlib import Path

import pytest


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _question(question_id):
    from edu_eval.knowledge.schema import KnowledgeQuestion

    return KnowledgeQuestion(
        question_id=question_id,
        source="test",
        level="SD",
        grade="6",
        subject="IPA",
        subject_group="STEM",
        question="q?",
        choices=["A. x", "B. y", "C. z"],
        answer_index=0,
    )


def _knowledge_record(question_id, correct=True):
    return {
        "question_id": question_id,
        "predictions": {"official": {"correct": correct}},
    }


def test_knowledge_torn_tail_truncated(tmp_path):
    module = _load("run_knowledge.py")
    path = tmp_path / "out.jsonl"
    path.write_text(
        json.dumps(_knowledge_record("0")) + "\n",
        encoding="utf-8",
    )

    with path.open("a", encoding="utf-8") as file:
        file.write('{"question_id": "1", "broken": ')

    processed, correct = module.read_existing_results(
        path, [_question("0"), _question("1")]
    )

    assert (processed, correct) == (1, 1)
    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_knowledge_middle_corruption_still_fatal(tmp_path):
    module = _load("run_knowledge.py")
    path = tmp_path / "out.jsonl"
    path.write_text(
        json.dumps(_knowledge_record("0"))
        + "\n{broken}\n"
        + json.dumps(_knowledge_record("1"))
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        module.read_existing_results(path, [_question("0"), _question("1")])


def _generation_record(case_id, grade, version="cg_v1"):
    module = _load("run_generation.py")

    return {
        "case_id": case_id,
        "target_grade": grade,
        "prompt_id": module.expected_prompt_id(version, case_id, grade),
    }


def test_generation_torn_tail_truncated(tmp_path):
    module = _load("run_generation.py")
    path = tmp_path / "out.jsonl"
    path.write_text(
        json.dumps(_generation_record("CG-001", "SD6")) + "\n",
        encoding="utf-8",
    )

    with path.open("a", encoding="utf-8") as file:
        file.write('{"case_id": "CG-002", ')

    done = module.read_existing_keys(path, "cg_v1")

    assert done == {("CG-001", "SD6")}
    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_generation_middle_corruption_still_fatal(tmp_path):
    module = _load("run_generation.py")
    path = tmp_path / "out.jsonl"
    path.write_text(
        json.dumps(_generation_record("CG-001", "SD6"))
        + "\n{oops}\n"
        + json.dumps(_generation_record("CG-002", "SD6"))
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        module.read_existing_keys(path, "cg_v1")
