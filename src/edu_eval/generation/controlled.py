import hashlib
from typing import TypedDict

from edu_eval.generation.runner import GenerationRunner
from edu_eval.generation.schema import ControlledGenerationCase, GradeTarget


class TargetMetadata(TypedDict):
    level: str
    grade: int


TARGET_METADATA: dict[GradeTarget, TargetMetadata] = {
    "SD6": {"level": "SD", "grade": 6},
    "SMP7": {"level": "SMP", "grade": 7},
    "SMP9": {"level": "SMP", "grade": 9},
    "SMA10": {"level": "SMA", "grade": 10},
}


class GenerationConfigSnapshot(TypedDict):
    max_new_tokens: int
    do_sample: bool
    use_cache: bool


class ControlledGenerationResult(TypedDict):
    model_id: str
    case_id: str
    source_id: str
    subject: str
    concept: str
    task_type: str
    target_grade: GradeTarget
    target_level: str
    prompt_id: str
    prompt_version: str
    generation_config: GenerationConfigSnapshot
    output_text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    hit_max_new_tokens: bool


class ControlledGenerationRunner:
    def __init__(
        self,
        generation_runner: GenerationRunner,
        model_id: str,
        target_grades: list[GradeTarget],
        prompt_version: str = "cg_v1",
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be empty.")

        self.generation_runner = generation_runner
        self.model_id = model_id
        self.target_grades = target_grades
        self.prompt_version = prompt_version
        unknown_grades = set(target_grades) - set(TARGET_METADATA)

        if unknown_grades:
            raise ValueError(f"Unsupported target grades: {sorted(unknown_grades)}")

    def build_prompt(
        self, case: ControlledGenerationCase, target_grade: GradeTarget
    ) -> str:
        metadata = TARGET_METADATA[target_grade]

        return (
            "Buat materi pembelajaran berbahasa Indonesia "
            "berdasarkan materi sumber berikut.\n\n"
            f"Mata pelajaran: {case.subject}\n"
            f"Konsep: {case.concept}\n"
            f"Target peserta didik: {metadata['level']} kelas {metadata['grade']}\n\n"
            "Materi sumber:\n"
            f"{case.reference.text}\n\n"
            "Tugas:\n"
            "Jelaskan konsep tersebut sebagai materi pembelajaran "
            "yang sesuai untuk target peserta didik. "
            "Gunakan materi sumber sebagai landasan faktual dan "
            "jangan bertentangan dengan materi tersebut. "
            "Sesuaikan tingkat bahasa, kedalaman konsep, dan "
            "cara penjelasan dengan target peserta didik. "
            "Jangan menyebutkan instruksi ini atau target grade "
            "secara eksplisit dalam jawaban."
        )

    def generate(
        self, case: ControlledGenerationCase, target_grade: GradeTarget
    ) -> ControlledGenerationResult:
        prompt = self.build_prompt(case, target_grade)
        prompt_id = hashlib.sha256(
            f"{self.prompt_version}:{case.case_id}:{target_grade}".encode()
        ).hexdigest()[:16]
        result = self.generation_runner.generate(prompt)
        metadata = TARGET_METADATA[target_grade]
        decoding = self.generation_runner.config

        return {
            "model_id": self.model_id,
            "case_id": case.case_id,
            "source_id": case.reference.source_id,
            "subject": case.subject,
            "concept": case.concept,
            "task_type": case.task_type,
            "target_grade": target_grade,
            "target_level": metadata["level"],
            "prompt_id": prompt_id,
            "prompt_version": self.prompt_version,
            "generation_config": {
                "max_new_tokens": decoding.max_new_tokens,
                "do_sample": decoding.do_sample,
                "use_cache": decoding.use_cache,
            },
            "output_text": result["text"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "latency_seconds": result["latency_seconds"],
            "hit_max_new_tokens": result["hit_max_new_tokens"],
        }

    def generate_case(
        self, case: ControlledGenerationCase
    ) -> list[ControlledGenerationResult]:
        return [self.generate(case, grade) for grade in self.target_grades]
