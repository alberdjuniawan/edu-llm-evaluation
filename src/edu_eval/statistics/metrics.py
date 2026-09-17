import math
import random
from collections.abc import Callable, Sequence


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def kendall_tau_b(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    if len(a) != len(b):
        raise ValueError("Orderings must have equal length.")

    if len(a) < 2:
        raise ValueError("Orderings need at least two items.")

    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0

    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da = (a[i] > a[j]) - (a[i] < a[j])
            db = (b[i] > b[j]) - (b[i] < b[j])

            if da == 0 and db == 0:
                continue

            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da == db:
                concordant += 1
            else:
                discordant += 1

    denominator = math.sqrt(
        (concordant + discordant + ties_a) * (concordant + discordant + ties_b)
    )

    if denominator == 0:
        return 0.0

    return (concordant - discordant) / denominator


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
    stat: Callable[[Sequence[float]], float] = _mean,
) -> tuple[float, float]:
    if not values:
        raise ValueError("values must not be empty.")

    if n_boot <= 0:
        raise ValueError("n_boot must be positive.")

    rng = random.Random(seed)
    estimates = sorted(
        stat([rng.choice(values) for _ in values]) for _ in range(n_boot)
    )

    lower_q = (1 - ci) / 2
    upper_q = 1 - lower_q

    lower = estimates[min(n_boot - 1, int(lower_q * n_boot))]
    upper = estimates[min(n_boot - 1, int(upper_q * n_boot))]

    return (lower, upper)


def paired_bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    if len(x) != len(y):
        raise ValueError("Paired samples must have equal length.")

    diffs = [float(a) - float(b) for a, b in zip(x, y, strict=True)]

    return bootstrap_ci(diffs, n_boot=n_boot, ci=ci, seed=seed, stat=_mean)


def krippendorff_alpha(
    ratings: Sequence[Sequence[float | None]],
    level: str = "ordinal",
) -> float:
    if level not in ("nominal", "ordinal"):
        raise ValueError("level must be 'nominal' or 'ordinal'.")

    values = sorted({value for item in ratings for value in item if value is not None})

    if len(values) < 2:
        raise ValueError("Need at least two distinct observed values.")

    index = {value: position for position, value in enumerate(values)}
    size = len(values)
    observed = [[0.0] * size for _ in range(size)]
    total_pairs = 0.0

    for item in ratings:
        present = [value for value in item if value is not None]

        if len(present) < 2:
            continue

        weight = 1.0 / (len(present) - 1)

        for i, left in enumerate(present):
            for j, right in enumerate(present):
                if i == j:
                    continue

                observed[index[left]][index[right]] += weight

        total_pairs += len(present)

    marginals = [sum(row) for row in observed]
    grand_total = sum(marginals)

    if grand_total == 0:
        raise ValueError("No pairable ratings found.")

    if level == "nominal":
        distance = [[0.0 if c == k else 1.0 for k in range(size)] for c in range(size)]
    else:
        positions = []
        cumulative = 0.0

        for count in marginals:
            positions.append((cumulative + count / 2) / grand_total)
            cumulative += count

        distance = [
            [(positions[c] - positions[k]) ** 2 for k in range(size)]
            for c in range(size)
        ]

    observed_disagreement = sum(
        observed[c][k] * distance[c][k] for c in range(size) for k in range(size)
    )

    expected_disagreement = sum(
        marginals[c] * marginals[k] / max(1.0, grand_total - 1) * distance[c][k]
        for c in range(size)
        for k in range(size)
    )

    if expected_disagreement == 0:
        return 1.0

    return 1 - observed_disagreement / expected_disagreement
