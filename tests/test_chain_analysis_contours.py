import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from jaxptpolypol.chain_analysis import (
    credible_intervals,
    plot_credible_contours,
)


def _correlated_normal(n_chains=4, n_draws=4000, seed=0):
    """4 chains of a 2-D correlated Gaussian, shape (chain, draw)."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.6], [0.6, 2.0]])
    chol = np.linalg.cholesky(cov)
    z = rng.normal(size=(n_chains, n_draws, 2)) @ chol.T
    return z[..., 0], z[..., 1]


def test_plot_credible_contours_draws_one_collection_per_level():
    x, y = _correlated_normal()
    fig, ax = plt.subplots()
    try:
        out = plot_credible_contours(x, y, ax=ax, levels=(0.68, 0.95))
        assert out is ax
        assert len(ax.collections) >= 2
    finally:
        plt.close(fig)


def test_plot_credible_contours_accepts_flat_arrays():
    """Pooled (already-flattened) samples must work too."""
    x, y = _correlated_normal()
    fig, ax = plt.subplots()
    try:
        plot_credible_contours(x.ravel(), y.ravel(), ax=ax)
        assert len(ax.collections) >= 2
    finally:
        plt.close(fig)


def test_plot_credible_contours_sorts_levels_ascending():
    """ArviZ requires ascending hdi_probs; we must not pass (0.95, 0.68)."""
    x, y = _correlated_normal()
    fig, ax = plt.subplots()
    try:
        plot_credible_contours(x, y, ax=ax, levels=(0.95, 0.68))
        assert len(ax.collections) >= 2
    finally:
        plt.close(fig)


def test_plot_credible_contours_rejects_invalid_levels():
    x, y = _correlated_normal()
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            plot_credible_contours(x, y, ax=ax, levels=(0.68, 1.0))
        with pytest.raises(ValueError, match="levels must be non-empty"):
            plot_credible_contours(x, y, ax=ax, levels=())
    finally:
        plt.close(fig)


def test_plot_credible_contours_rejects_mismatched_shapes():
    x, y = _correlated_normal()
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="same shape"):
            plot_credible_contours(x, y[:, :10], ax=ax)
    finally:
        plt.close(fig)


def test_credible_intervals_recovers_known_gaussian():
    """68%/95% HDI of a standard normal are ~+/-1.0 and ~+/-1.96."""
    rng = np.random.default_rng(1)
    draws = rng.normal(size=(4, 20000))
    table = credible_intervals({"a": draws}, var_names=["a"],
                               levels=(0.68, 0.95))
    lo68, hi68 = table["a"][0.68]
    lo95, hi95 = table["a"][0.95]
    assert lo68 == pytest.approx(-1.0, abs=0.06)
    assert hi68 == pytest.approx(1.0, abs=0.06)
    assert lo95 == pytest.approx(-1.96, abs=0.08)
    assert hi95 == pytest.approx(1.96, abs=0.08)


def test_credible_intervals_95_contains_68():
    """Nesting is a mathematical property of HDIs -- cannot go stale."""
    rng = np.random.default_rng(2)
    draws = rng.normal(size=(4, 8000))
    table = credible_intervals({"a": draws}, var_names=["a"],
                               levels=(0.68, 0.95))
    lo68, hi68 = table["a"][0.68]
    lo95, hi95 = table["a"][0.95]
    assert lo95 <= lo68 and hi95 >= hi68


def test_credible_intervals_empirical_coverage():
    """The 68% interval must actually contain ~68% of the draws."""
    rng = np.random.default_rng(3)
    draws = rng.normal(size=(4, 20000))
    table = credible_intervals({"a": draws}, var_names=["a"], levels=(0.68,))
    lo, hi = table["a"][0.68]
    frac = float(np.mean((draws >= lo) & (draws <= hi)))
    assert frac == pytest.approx(0.68, abs=0.02)


def test_credible_intervals_rejects_missing_variable():
    rng = np.random.default_rng(4)
    with pytest.raises(KeyError, match="not found"):
        credible_intervals({"a": rng.normal(size=(2, 100))},
                           var_names=["b"], levels=(0.68,))
