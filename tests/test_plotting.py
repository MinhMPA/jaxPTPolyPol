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
