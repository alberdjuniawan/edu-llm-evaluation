from pathlib import Path

import yaml

from edu_eval.models.schema import ModelSpec


class ModelRegistry:
    def __init__(self, models: list[ModelSpec]) -> None:
        self._models = {model.model_id: model for model in models}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelRegistry":
        config_path = Path(path)

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise TypeError("Model configuration must be a YAML mapping.")

        raw_models = config.get("models")

        if not isinstance(raw_models, list):
            raise TypeError("'models' must be a list.")

        models = [ModelSpec.model_validate(raw) for raw in raw_models]
        model_ids = [model.model_id for model in models]

        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Duplicate model_id found in model registry.")

        return cls(models)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"Model '{model_id}' is not registered.") from exc

    def all(self) -> list[ModelSpec]:
        return list(self._models.values())
