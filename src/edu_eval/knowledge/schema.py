from pydantic import BaseModel, Field


class KnowledgeQuestion(BaseModel):
    question_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    level: str = Field(min_length=1)
    grade: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    subject_group: str = Field(min_length=1)
    question: str = Field(min_length=1)
    choices: list[str] = Field(min_length=3, max_length=5)
    answer_index: int = Field(ge=0, le=4)
    is_for_fewshot: bool = False
