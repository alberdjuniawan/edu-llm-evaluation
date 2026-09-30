from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from edu_eval.generation.schema import GradeTarget


class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(gt=0)
    do_sample: bool = False
    use_cache: bool = True
    enable_thinking: bool | None = None

    prompt_version: str = Field(min_length=1)

    target_grades: list[GradeTarget] = Field(
        min_length=1,
    )

    subjects: list[str] = Field(
        min_length=1,
    )

    pilot_cases_per_subject: int = Field(
        gt=0,
    )

    full_cases_per_subject: int = Field(
        gt=0,
    )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "GenerationConfig":
        config_path = Path(path)

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise TypeError("Generation configuration must be a YAML mapping.")

        raw_generation = config.get("generation")

        if not isinstance(raw_generation, dict):
            raise TypeError("'generation' must be a YAML mapping.")

        return cls.model_validate(raw_generation)


def require_deterministic(config: GenerationConfig, allow_sampling: bool) -> None:
    if config.do_sample and not allow_sampling:
        raise RuntimeError(
            "Primary protocol is deterministic (do_sample=false). "
            "Pass --allow-sampling for robustness runs."
        )
