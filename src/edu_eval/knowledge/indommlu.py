import csv
from pathlib import Path

from edu_eval.knowledge.schema import KnowledgeQuestion

ANSWER_INDEX = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
}

KNOWN_INVALID_ROW_INDICES = frozenset({4223, 14150})

SPECIAL_CASE_LEVELS = frozenset({"SD-SMP-SMA", "SD-SMP"})

LEVEL_MAP = {
    "SMA": "SMA",
    "Seleksi PTN": "University entrance test",
    "SD": "SD",
    "SMP": "SMP",
    "Kelas I SD": "SD",
    "Kelas II SD": "SD",
    "Kelas III SD": "SD",
    "Kelas IV SD": "SD",
    "V SD": "SD",
    "VI SD": "SD",
    "Kelas X SMA": "SMA",
    "Kelas XI SMA": "SMA",
    "Kelas XII SMA": "SMA",
    "VII SMP": "SMP",
    "VIII SMP": "SMP",
    "IX SMP": "SMP",
}

SUBJECT_GROUP = {
    "Sejarah": "Humanities",
    "Geografi": "Social science",
    "Bahasa Lampung": "Local Languages & Cultures",
    "IPS": "Social science",
    "Bahasa Bali": "Local Languages & Cultures",
    "Bahasa Makassar": "Local Languages & Cultures",
    "Bahasa Banjar": "Local Languages & Cultures",
    "Kimia": "STEM",
    "Biologi": "STEM",
    "IPA": "STEM",
    "Agama Kristen": "Humanities",
    "Kesenian": "Humanities",
    "Agama Islam": "Humanities",
    "Agama Hindu": "Humanities",
    "Bahasa Madura": "Local Languages & Cultures",
    "Penjaskes": "Humanities",
    "Bahasa Indonesia": "Indonesian Language",
    "Fisika": "STEM",
    "Budaya Alam Minangkabau": "Local Languages & Cultures",
    "Bahasa Dayak Ngaju": "Local Languages & Cultures",
    "Sosiologi": "Social science",
    "Ekonomi": "Social science",
    "Bahasa Sunda": "Local Languages & Cultures",
    "Bahasa Jawa": "Local Languages & Cultures",
    "PPKN": "Social science",
}


def normalize_level_and_grade(
    raw_level: str,
    raw_grade: str,
) -> tuple[str, int]:
    level = raw_level.strip()
    grade = raw_grade.strip()

    if level in SPECIAL_CASE_LEVELS:
        numeric_grade = float(grade)

        if 1 <= numeric_grade <= 6:
            level = "SD"
        elif 7 <= numeric_grade <= 9:
            level = "SMP"
        elif numeric_grade >= 10:
            level = "SMA"
        else:
            raise ValueError(f"Invalid grade {raw_grade!r} for level {raw_level!r}.")

    try:
        fixed_level = LEVEL_MAP[level]
    except KeyError as exc:
        raise ValueError(f"Unknown IndoMMLU level: {raw_level!r}") from exc

    if grade in {"PTN", "2023-10-12 00:00:00"}:
        fixed_grade = 13
    elif grade == "4,5,6":
        fixed_grade = 6
    else:
        try:
            fixed_grade = int(float(grade))
        except ValueError as exc:
            raise ValueError(f"Invalid IndoMMLU grade: {raw_grade!r}") from exc

    return fixed_level, fixed_grade


def parse_fewshot_flag(raw_value: str) -> bool:
    value = str(raw_value).strip().lower()

    if value in {"1", "true"}:
        return True

    if value in {"0", "false"}:
        return False

    raise ValueError(f"Invalid is_for_fewshot value: {raw_value!r}")


def parse_choices(raw_choices: str) -> list[str]:
    choices = raw_choices.split("\n")

    if not 3 <= len(choices) <= 5:
        raise ValueError(f"Expected 3-5 choices, got {len(choices)}.")

    return choices


def find_answer_index(
    choices: list[str],
    answer_key: str,
) -> int:
    answer_key = answer_key.strip().upper()

    if answer_key not in ANSWER_INDEX:
        raise ValueError(f"Invalid answer key: {answer_key!r}")

    expected_prefix = f"{answer_key}."

    for index, choice in enumerate(choices):
        if choice.startswith(expected_prefix):
            return index

    raise ValueError(f"Answer key {answer_key!r} has no matching choice.")


def get_subject_group(subject: str) -> str:
    try:
        return SUBJECT_GROUP[subject]
    except KeyError as exc:
        raise ValueError(f"Unknown IndoMMLU subject: {subject!r}") from exc


def format_indommlu_prompt(item: KnowledgeQuestion) -> str:
    if item.level == "University entrance test":
        prompt_level = "seleksi masuk universitas"
    else:
        prompt_level = f"{item.grade} {item.level}"

    option_text = "\n".join(item.choices)

    return (
        f"Ini adalah soal {item.subject} untuk {prompt_level}. "
        "Pilihlah salah satu jawaban yang dianggap benar!\n\n"
        f"{item.question}\n"
        f"{option_text}\n\n"
        "Jawaban: "
    )


def load_indommlu_csv(
    path: str | Path,
) -> list[KnowledgeQuestion]:
    csv_path = Path(path)

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("IndoMMLU CSV has no header.")

        required_columns = {
            "level",
            "kelas",
            "subject",
            "soal",
            "jawaban",
            "kunci",
            "is_for_fewshot",
        }

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        questions: list[KnowledgeQuestion] = []

        for row_index, row in enumerate(reader):
            level, grade = normalize_level_and_grade(
                row["level"],
                row["kelas"],
            )

            choices = parse_choices(row["jawaban"])

            try:
                answer_index = find_answer_index(
                    choices,
                    row["kunci"],
                )
            except ValueError:
                if row_index in KNOWN_INVALID_ROW_INDICES:
                    continue

                raise

            subject = row["subject"].strip()

            questions.append(
                KnowledgeQuestion(
                    question_id=str(row_index),
                    source=(row["sumber"].strip() if row.get("sumber") else "IndoMMLU"),
                    level=level,
                    grade=str(grade),
                    subject=subject,
                    question=row["soal"].strip(),
                    choices=choices,
                    answer_index=answer_index,
                    is_for_fewshot=parse_fewshot_flag(row["is_for_fewshot"]),
                    subject_group=get_subject_group(subject),
                )
            )

    return questions
