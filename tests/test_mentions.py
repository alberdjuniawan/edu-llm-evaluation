from edu_eval.linguistic.mentions import classify_mentions


def test_grade_mention_detected():
    flags = classify_mentions("Materi ini untuk siswa kelas 7 SMP.")

    assert flags["grade_mention"] is True
    assert flags["instruction_disclosure"] is False
    assert flags["meta_prompt_disclosure"] is False


def test_instruction_disclosure_detected():
    flags = classify_mentions("Target grade yang diberikan pada prompt adalah SMP7.")

    assert flags["instruction_disclosure"] is True


def test_meta_disclosure_detected():
    flags = classify_mentions("Sebagai model bahasa, saya akan menjelaskan.")

    assert flags["meta_prompt_disclosure"] is True


def test_clean_text_has_no_flags():
    flags = classify_mentions("Fotosintesis terjadi di daun dengan bantuan cahaya.")

    assert flags == {
        "grade_mention": False,
        "instruction_disclosure": False,
        "meta_prompt_disclosure": False,
    }
