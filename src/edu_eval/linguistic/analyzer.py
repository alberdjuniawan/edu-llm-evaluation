import re
from typing import TypedDict

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_WORD_RE = re.compile(r"[A-Za-z']+")
_VOWEL_GROUP_RE = re.compile(r"[aiueoAIUEO]+")

ID_FUNCTION_WORDS = frozenset(
    {
        "yang",
        "dan",
        "di",
        "ke",
        "dari",
        "untuk",
        "dengan",
        "pada",
        "adalah",
        "ialah",
        "ini",
        "itu",
        "tersebut",
        "sebagai",
        "dalam",
        "secara",
        "telah",
        "sudah",
        "akan",
        "bisa",
        "dapat",
        "tidak",
        "tak",
        "bukan",
        "jangan",
        "sangat",
        "lebih",
        "kurang",
        "paling",
        "juga",
        "serta",
        "atau",
        "tetapi",
        "tapi",
        "namun",
        "karena",
        "sebab",
        "jika",
        "kalau",
        "apabila",
        "agar",
        "supaya",
        "bahwa",
        "yaitu",
        "yakni",
        "antara",
        "oleh",
        "terhadap",
        "mengenai",
        "tentang",
        "setiap",
        "semua",
        "seluruh",
        "para",
        "sang",
        "si",
        "per",
        "antar",
        "tanpa",
        "hingga",
        "sampai",
        "sejak",
        "demi",
        "kepada",
        "cukup",
        "hanya",
        "saja",
        "lagi",
        "masih",
        "saling",
        "masing",
    }
)


class LinguisticFeatures(TypedDict):
    char_count: int
    word_count: int
    sentence_count: int
    words_per_sentence: float
    avg_word_length_chars: float
    lexical_diversity_ttr: float
    lexical_density: float
    fkgl: float
    fog: float
    ari: float
    english_formula: bool


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text.strip())

    return [part.strip() for part in parts if part.strip()]


def tokenize_words(text: str) -> list[str]:
    return [match.lower() for match in _WORD_RE.findall(text)]


def count_syllables_id(word: str) -> int:
    groups = _VOWEL_GROUP_RE.findall(word)

    return max(1, len(groups))


def analyze_text(text: str) -> LinguisticFeatures:
    sentences = split_sentences(text)
    words = tokenize_words(text)

    word_count = len(words)
    sentence_count = len(sentences)

    if word_count == 0:
        return {
            "char_count": len(text),
            "word_count": 0,
            "sentence_count": sentence_count,
            "words_per_sentence": 0.0,
            "avg_word_length_chars": 0.0,
            "lexical_diversity_ttr": 0.0,
            "lexical_density": 0.0,
            "fkgl": 0.0,
            "fog": 0.0,
            "ari": 0.0,
            "english_formula": True,
        }

    words_per_sentence = word_count / max(1, sentence_count)
    avg_word_length = sum(len(word) for word in words) / word_count
    syllables = [count_syllables_id(word) for word in words]
    syllables_per_word = sum(syllables) / word_count
    complex_words = sum(1 for count in syllables if count >= 3)

    char_count = len(text)
    chars_per_word = sum(len(word) for word in words) / word_count

    fkgl = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    fog = 0.4 * (words_per_sentence + 100 * complex_words / word_count)
    ari = 4.71 * chars_per_word + 0.5 * words_per_sentence - 21.43

    content_words = sum(1 for word in words if word not in ID_FUNCTION_WORDS)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "words_per_sentence": words_per_sentence,
        "avg_word_length_chars": avg_word_length,
        "lexical_diversity_ttr": len(set(words)) / word_count,
        "lexical_density": content_words / word_count,
        "fkgl": fkgl,
        "fog": fog,
        "ari": ari,
        "english_formula": True,
    }
