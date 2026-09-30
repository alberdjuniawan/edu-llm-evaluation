import re
from typing import TypedDict

GRADE_MENTION_RE = re.compile(
    r"\bkelas\s*(6|7|9|10|vi|vii|ix|x|enam|tujuh|sembilan|sepuluh)\b",
    re.IGNORECASE,
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
        "grade_mention": bool(GRADE_MENTION_RE.search(text)),
        "instruction_disclosure": _hits(text, INSTRUCTION_PATTERNS),
        "meta_prompt_disclosure": _hits(text, META_PATTERNS),
    }
