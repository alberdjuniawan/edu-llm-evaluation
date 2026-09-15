from edu_eval.linguistic.analyzer import (
    analyze_text,
    count_syllables_id,
    split_sentences,
    tokenize_words,
)


def test_empty_text_returns_zeros():
    features = analyze_text("")

    assert features["word_count"] == 0
    assert features["words_per_sentence"] == 0.0
    assert features["english_formula"] is True


def test_sentence_splitting():
    assert split_sentences("Satu. Dua! Tiga?") == ["Satu.", "Dua!", "Tiga?"]


def test_word_tokenization_lowercases():
    assert tokenize_words("Halo Dunia") == ["halo", "dunia"]


def test_syllable_heuristic_minimum_one():
    assert count_syllables_id("dan") == 1
    assert count_syllables_id("makanan") == 3


def test_repetition_lowers_ttr():
    varied = analyze_text("Kucing makan ikan. Anjing minum susu.")
    repeated = analyze_text("Kucing kucing kucing. Kucing kucing kucing.")

    assert varied["lexical_diversity_ttr"] > repeated["lexical_diversity_ttr"]


def test_function_words_lower_density():
    dense = analyze_text("Fotosintesis menghasilkan glukosa oksigen.")
    filler = analyze_text("Ini adalah dan yang untuk dengan sangat juga.")

    assert dense["lexical_density"] > filler["lexical_density"]


def test_english_formulas_flagged():
    features = analyze_text("Air menguap karena panas matahari.")

    assert features["english_formula"] is True
    assert isinstance(features["fkgl"], float)
