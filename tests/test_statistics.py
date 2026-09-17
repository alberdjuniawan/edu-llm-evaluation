import pytest

from edu_eval.statistics.metrics import (
    bootstrap_ci,
    kendall_tau_b,
    krippendorff_alpha,
    paired_bootstrap_ci,
)


def test_kendall_identical_is_one():
    assert kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_kendall_reversed_is_minus_one():
    assert kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_kendall_rejects_unequal_length():
    with pytest.raises(ValueError):
        kendall_tau_b([1, 2], [1])


def test_alpha_perfect_agreement_is_one():
    ratings = [[5, 5, 5], [3, 3, 3], [1, 1, 1], [4, 4, 4]]

    assert krippendorff_alpha(ratings, level="nominal") == pytest.approx(1.0)
    assert krippendorff_alpha(ratings, level="ordinal") == pytest.approx(1.0)


def test_alpha_total_disagreement_nominal():
    ratings = [[1, 2], [2, 1]]

    assert krippendorff_alpha(ratings, level="nominal") == pytest.approx(-0.5)


def test_alpha_handles_missing_data():
    ratings = [[5, 5, None], [3, None, 3], [1, 1, 1]]

    alpha = krippendorff_alpha(ratings, level="ordinal")

    assert -1.0 <= alpha <= 1.0


def test_alpha_rejects_unknown_level():
    with pytest.raises(ValueError):
        krippendorff_alpha([[1, 1]], level="interval")


def test_bootstrap_ci_constant_values():
    low, high = bootstrap_ci([1.0, 1.0, 1.0], n_boot=100, seed=1)

    assert (low, high) == (1.0, 1.0)


def test_bootstrap_ci_contains_sample_mean():
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    low, high = bootstrap_ci(values, n_boot=500, seed=42)

    assert low <= 0.6 <= high


def test_paired_bootstrap_ci_equal_samples_covers_zero():
    sample = [0.0, 1.0, 1.0, 0.0]
    low, high = paired_bootstrap_ci(sample, sample, n_boot=500, seed=42)

    assert low <= 0.0 <= high


def test_paired_bootstrap_rejects_unequal_length():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 0.0])
