"""
Plotting utilities for Fisher contours and 1-d marginals.
"""

from __future__ import annotations

import numpy as np
from jax.scipy.linalg import eigh, inv
from jax.scipy.stats import norm

__all__ = ["plot_contours", "plot_Gaussian", "triangle_plot"]


def plot_contours(
    fisher,
    pos,
    inds,
    cls=None,
    ax=None,
    **kwargs,
):
    """Plot 2-d Fisher ellipses for a pair of parameters.

    Parameters
    ----------
    fisher : array, shape (n, n)
        Fisher matrix (already restricted to varied parameters).
    pos : array, shape (n,)
        Fiducial parameter values (varied subset).
    inds : array-like, shape (2,)
        Indices of the two parameters to plot.
    cls : array-like, optional
        Confidence levels.  Default ``[0.6827, 0.9545]`` (1 and 2 sigma).
    ax : matplotlib Axes, optional
    **kwargs
        Passed to ``matplotlib.patches.Ellipse``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    if cls is None:
        cls = np.array([0.6827, 0.9545])

    inds = np.asarray(inds)
    cov = inv(fisher)[np.ix_(inds, inds)]
    vals, vecs = eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    nstds = norm.ppf(0.5 * (1 + np.asarray(cls)))

    if ax is None:
        ax = plt.gca()
    for nstd in nstds:
        ax.add_artist(
            Ellipse(
                xy=pos[inds],
                width=2.0 * nstd * np.sqrt(vals[0]),
                height=2.0 * nstd * np.sqrt(vals[1]),
                angle=theta,
                **kwargs,
            )
        )
    nstdmax = max(nstds)
    sx = nstdmax * np.sqrt(cov[0, 0])
    sy = nstdmax * np.sqrt(cov[1, 1])
    ax.set_xlim(pos[inds[0]] - 1.5 * sx, pos[inds[0]] + 1.5 * sx)
    ax.set_ylim(pos[inds[1]] - 1.5 * sy, pos[inds[1]] + 1.5 * sy)


def plot_Gaussian(
    fisher,
    pos,
    ind,
    cl=0.9545,
    ax=None,
    **kwargs,
):
    """Plot 1-d marginalized Gaussian for a single parameter.

    Parameters
    ----------
    fisher : array, shape (n, n)
        Fisher matrix (varied subset).
    pos : array, shape (n,)
        Fiducial parameter values (varied subset).
    ind : int
        Index of the parameter.
    cl : float
        Confidence level for the x-axis range.
    ax : matplotlib Axes, optional
    **kwargs
        Passed to ``ax.plot``.
    """
    import matplotlib.pyplot as plt

    mu = np.array(pos[ind])
    sigma = np.sqrt(inv(fisher)[ind, ind])
    nstd = float(norm.ppf(0.5 * (1 + cl)))
    if ax is None:
        ax = plt.gca()
    x = np.linspace(mu - 1.5 * nstd * sigma, mu + 1.5 * nstd * sigma, 100)
    ax.plot(x, np.exp(-0.5 * ((x - mu) / sigma) ** 2), **kwargs)
    ax.set_xlim(x[0], x[-1])


def triangle_plot(
    fisher,
    packed_varied,
    cosmo_marg_idx,
    cosmo_param_names,
    title=None,
    figsize_per_param=2.5,
    **contour_kwargs,
):
    """Produce a triangle plot for selected cosmological parameters.

    Parameters
    ----------
    fisher : array, shape (n_varied, n_varied)
        Marginalized Fisher matrix.
    packed_varied : array, shape (n_varied,)
        Fiducial values for all varied parameters.
    cosmo_marg_idx : array of int
        Indices of the cosmological parameters within the varied subset.
    cosmo_param_names : tuple of str
        LaTeX labels for each cosmological parameter.
    title : str, optional
        Figure super-title.
    figsize_per_param : float
        Figure size per parameter panel.
    **contour_kwargs
        Passed to :func:`plot_contours` and :func:`plot_Gaussian`
        (e.g. ``color='royalblue'``).

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    n = len(cosmo_param_names)
    cosmo_marg_idx = np.asarray(cosmo_marg_idx)

    contour_kw = {"fill": False}
    contour_kw.update(contour_kwargs)
    gauss_kw = {k: v for k, v in contour_kwargs.items() if k in ("color", "ls", "lw")}

    fig = plt.figure(
        figsize=(n * figsize_per_param, n * figsize_per_param),
        constrained_layout=True,
    )
    for i in range(n):
        for j in range(n):
            ax = plt.subplot(n, n, i * n + j + 1)
            if j < i:
                plot_contours(
                    fisher,
                    packed_varied,
                    np.array([cosmo_marg_idx[j], cosmo_marg_idx[i]]),
                    ax=ax,
                    **contour_kw,
                )
            elif j == i:
                plot_Gaussian(
                    fisher,
                    packed_varied,
                    cosmo_marg_idx[i],
                    ax=ax,
                    **gauss_kw,
                )
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(cosmo_param_names[j])
            if j == 0:
                ax.set_ylabel(cosmo_param_names[i])

    if title:
        fig.suptitle(title, fontsize=16)
    return fig
