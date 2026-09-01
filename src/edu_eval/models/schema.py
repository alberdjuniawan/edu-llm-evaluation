from typing import Literal

from pydantic import BaseModel, Field

ModelRole = Literal[
    "smoke_test",
    "research",
]

ModelStage = Literal[
    "smoke",
    "pretrained",
    "cpt",
    "sft_v1",
    "sft_v2",
    "merged_sft",
]

DocumentationStatus = Literal[
    "complete",
    "partial",
    "unknown",
]


class ModelSpec(BaseModel):
    model_id: str = Field(
        min_length=1,
        description="Identified for evaluation pipeline.",
    )

    model_role: ModelRole

    stage: ModelStage

    source: str = Field(
        min_length=1,
        description="HF repo id or local checkpoint path.",
    )

    revision: str | None = Field(
        default=None,
        description="Exact model revision, tag, or commit hash.",
    )

    architecture: str | None = None

    parameter_count: int | None = Field(
        default=None,
        ge=0,
    )

    dtype: str | None = None

    context_length: int | None = Field(
        default=None,
        gt=0,
    )

    chat_template_available: bool | None = None

    documentation_status: DocumentationStatus = "unknown"
