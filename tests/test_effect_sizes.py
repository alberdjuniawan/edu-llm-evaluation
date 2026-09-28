import pytest

from edu_eval.statistics.metrics import cliffs_delta, cohens_h


def test_cohens_h_zero_for_equal_rates():
    assert cohens_h(0.5, 0.5) == pytest.approx(0.0)


def test_cohens_h_sign_and_symmetry():
    assert cohens_h(0.7, 0.6) == pytest.approx(-cohens_h(0.6, 0.7))
    assert cohens_h(0.7, 0.6) > 0


def test_cohens_h_clamps_bounds():
    assert cohens_h(1.0, 0.0) == pytest.approx(3.14159265, rel=1e-6)


def test_cliffs_delta_identical_is_zero():
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_cliffs_delta_dominance():
    assert cliffs_delta([3, 4, 5], [1, 1, 1]) == pytest.approx(1.0)
    assert cliffs_delta([1, 1, 1], [3, 4, 5]) == pytest.approx(-1.0)


def test_cliffs_delta_rejects_empty():
    with pytest.raises(ValueError):
        cliffs_delta([], [1.0])
