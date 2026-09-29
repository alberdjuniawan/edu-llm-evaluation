from typing import Literal

from pydantic import BaseModel, Field

ModelRole = Literal[
    "smoke_test",
    "research",
]

ModelStage = Literal[
    "smoke",
    "base",
    "pretrained",
    "cpt",
    "sft_v1",
    "sft_v2",
    "merged_sft",
    "final",
]

DocumentationStatus = Literal[
    "complete",
    "partial",
    "unknown",
]

LineageStatus = Literal[
    "verified",
    "unresolved",
]

Precision = Literal[
    "bf16",
    "fp16",
    "fp32",
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

    lineage_group: str | None = Field(
        default=None,
        description="Research lineage this checkpoint belongs to.",
    )

    parent_model_id: str | None = Field(
        default=None,
        description="Registered parent checkpoint, if any.",
    )

    provenance_url: str | None = Field(
        default=None,
        description="Artifact or documentation URL.",
    )

    lineage_status: LineageStatus = Field(
        default="unresolved",
        description="Verified means source and revision confirmed.",
    )

    precision: Precision | None = Field(
        default=None,
        description="Forced compute precision; defaults apply when unset.",
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
