from __future__ import annotations

import numpy as np

from scripts.analyze_synthetic_benchmark import bootstrap_mean_interval, holm_adjust


def test_holm_adjust_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_bootstrap_interval_is_exact_for_constant_differences() -> None:
    interval = bootstrap_mean_interval(
        np.array([-0.2, -0.2, -0.2]), 100, np.random.default_rng(7)
    )
    assert np.allclose(interval, (-0.2, -0.2))
