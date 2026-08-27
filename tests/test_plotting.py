import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from jaxptpolypol.plotting import plot_Gaussian, plot_contours


def test_plot_gaussian_uses_pinv_for_singular_fisher():
    fisher = np.diag([4.0, 9.0, 0.0])
    pos = np.array([1.0, 2.0, 3.0])

    fig, ax = plt.subplots()
    try:
        with pytest.warns(RuntimeWarning, match="pseudo-inverse fallback"):
            plot_Gaussian(fisher, pos, 0, ax=ax, color="C0")
        left, right = ax.get_xlim()
        assert np.isfinite(left)
        assert np.isfinite(right)
        assert left < right
        assert len(ax.lines) == 1
    finally:
        plt.close(fig)


def test_plot_contours_uses_pinv_for_singular_fisher():
    fisher = np.diag([4.0, 9.0, 0.0])
    pos = np.array([1.0, 2.0, 3.0])

    fig, ax = plt.subplots()
    try:
        with pytest.warns(RuntimeWarning, match="pseudo-inverse fallback"):
            plot_contours(fisher, pos, [0, 1], ax=ax, fill=False, color="C1")
        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        assert np.isfinite(left)
        assert np.isfinite(right)
        assert np.isfinite(bottom)
        assert np.isfinite(top)
        assert left < right
        assert bottom < top
        assert len(ax.patches) == 2
    finally:
        plt.close(fig)


from jaxptpolypol.plotting import credible_level_nstd


def test_credible_level_nstd_1d_matches_sigma_multiples():
    """1-D interpretation: 0.6827 -> 1 sigma, 0.9545 -> 2 sigma."""
    assert credible_level_nstd(0.6827, ndim=1) == pytest.approx(1.0, abs=1e-4)
    assert credible_level_nstd(0.9545, ndim=1) == pytest.approx(2.0, abs=1e-4)


def test_credible_level_nstd_2d_is_larger_than_1d():
    """2-D credible regions need a LARGER multiplier for the same mass.

    This is the whole point of the switch: a 1-sigma ellipse encloses only
    39.35% of 2-D mass, so a true 68% 2-D contour sits at 1.5096 sigma.
    """
    assert credible_level_nstd(0.68, ndim=2) == pytest.approx(1.5096, abs=1e-4)
    assert credible_level_nstd(0.95, ndim=2) == pytest.approx(2.4477, abs=1e-4)
    assert credible_level_nstd(0.68, ndim=2) > credible_level_nstd(0.68, ndim=1)


def test_credible_level_nstd_rejects_invalid_mass():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            credible_level_nstd(bad, ndim=2)


def test_credible_level_nstd_rejects_bad_ndim():
    with pytest.raises(ValueError, match="ndim must be 1 or 2"):
        credible_level_nstd(0.68, ndim=3)


def test_plot_contours_default_is_unchanged_backward_compat():
    """PIN: the default must keep drawing 1-sigma/2-sigma ellipses.

    Every committed notebook figure was produced with this default, and the
    repo treats committed notebook outputs as fixtures. If this test fails,
    someone changed the default and silently altered every existing figure.
    """
    fisher = np.diag([4.0, 9.0])
    pos = np.array([0.0, 0.0])

    fig, ax = plt.subplots()
    try:
        plot_contours(fisher, pos, np.array([0, 1]), ax=ax, fill=False)
        widths = sorted(float(e.get_width()) for e in ax.patches)
        # sigma_x = 1/2; 1-sigma ellipse width = 2*1*0.5 = 1.0, 2-sigma = 2.0
        # (0.6827/0.9545 are 4-decimal roundings of the true 1/2-sigma mass,
        # so norm.ppf legitimately returns ~1.0000217/~2.0000024, matching
        # the pre-existing default behaviour bit-for-bit.)
        assert widths == pytest.approx([1.0, 2.0], abs=1e-4)
    finally:
        plt.close(fig)


def test_plot_contours_level_kind_mass2d_draws_larger_ellipses():
    """level_kind='mass2d' reinterprets cls as 2-D probability mass."""
    fisher = np.diag([4.0, 9.0])
    pos = np.array([0.0, 0.0])

    fig, ax = plt.subplots()
    try:
        plot_contours(fisher, pos, np.array([0, 1]), cls=[0.68, 0.95],
                      level_kind="mass2d", ax=ax, fill=False)
        widths = sorted(float(e.get_width()) for e in ax.patches)
        # width = 2 * nstd * sigma_x, sigma_x = 0.5 -> width == nstd
        assert widths == pytest.approx([1.5096, 2.4477], abs=1e-3)
    finally:
        plt.close(fig)


def test_plot_contours_rejects_unknown_level_kind():
    fisher = np.diag([4.0, 9.0])
    pos = np.array([0.0, 0.0])
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="level_kind must be"):
            plot_contours(fisher, pos, np.array([0, 1]),
                          level_kind="nonsense", ax=ax)
    finally:
        plt.close(fig)
