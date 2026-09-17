from typing import TypedDict

GRADE_MENTION_PATTERNS = (
    "kelas 6",
    "kelas 7",
    "kelas 9",
    "kelas 10",
)

INSTRUCTION_PATTERNS = (
    "target grade",
    "instruksi ini",
    "prompt",
)

META_PATTERNS = (
    "sebagai model bahasa",
    "sebagai ai",
    "model ai",
)


class MentionFlags(TypedDict):
    grade_mention: bool
    instruction_disclosure: bool
    meta_prompt_disclosure: bool


def _hits(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()

    return any(pattern in lowered for pattern in patterns)


def classify_mentions(text: str) -> MentionFlags:
    return {
        "grade_mention": _hits(text, GRADE_MENTION_PATTERNS),
        "instruction_disclosure": _hits(text, INSTRUCTION_PATTERNS),
        "meta_prompt_disclosure": _hits(text, META_PATTERNS),
    }
