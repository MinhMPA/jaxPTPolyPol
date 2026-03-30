"""
Plotting utilities for Fisher contours and 1-d marginals.
"""

from __future__ import annotations

import numpy as np
import warnings

from jax.scipy.stats import norm

__all__ = ["plot_contours", "plot_Gaussian", "triangle_plot"]


def _covariance_from_fisher(fisher, *, rcond: float = 1e-12) -> np.ndarray:
    """Return a finite covariance matrix, falling back to a pseudo-inverse."""
    fisher_np = np.asarray(fisher, dtype=float)
    if fisher_np.ndim != 2 or fisher_np.shape[0] != fisher_np.shape[1]:
        raise ValueError(
            f"fisher must be a square matrix, got shape {fisher_np.shape}"
        )
    if not np.all(np.isfinite(fisher_np)):
        raise ValueError("fisher contains NaN or Inf entries")

    fisher_np = 0.5 * (fisher_np + fisher_np.T)

    used_pinv = False
    try:
        cov = np.linalg.inv(fisher_np)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher_np, rcond=rcond, hermitian=True)
        used_pinv = True

    cov = np.asarray(0.5 * (cov + cov.T), dtype=float)
    if not np.all(np.isfinite(cov)):
        cov = np.linalg.pinv(fisher_np, rcond=rcond, hermitian=True)
        cov = np.asarray(0.5 * (cov + cov.T), dtype=float)
        used_pinv = True

    if not np.all(np.isfinite(cov)):
        raise ValueError("could not compute a finite covariance from fisher")

    if used_pinv:
        warnings.warn(
            "Fisher matrix is singular or ill-conditioned; plotting uses a "
            "pseudo-inverse fallback.",
            RuntimeWarning,
            stacklevel=3,
        )

    return cov


def _nonnegative_or_none(value: float, *, atol: float = 1e-15) -> float | None:
    """Return a clipped non-negative scalar or ``None`` for invalid values."""
    value = float(value)
    if not np.isfinite(value):
        return None
    if value < 0.0:
        if abs(value) <= atol:
            return 0.0
        return None
    return value


def _fallback_halfspan(center: float) -> float:
    """Small finite axis span used when a marginal width collapses to zero."""
    return 1e-6 * max(abs(center), 1.0)


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
    cov = _covariance_from_fisher(fisher)[np.ix_(inds, inds)]
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.array(
        [
            _nonnegative_or_none(vals[0]),
            _nonnegative_or_none(vals[1]),
        ],
        dtype=object,
    )
    if vals[0] is None or vals[1] is None:
        warnings.warn(
            f"Skipping contour for indices {inds.tolist()} because the "
            "marginal covariance is not positive semidefinite.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    vals = np.asarray(vals, dtype=float)
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
    sx2 = _nonnegative_or_none(cov[0, 0])
    sy2 = _nonnegative_or_none(cov[1, 1])
    if sx2 is None or sy2 is None:
        warnings.warn(
            f"Skipping contour axis limits for indices {inds.tolist()} because "
            "the marginal variances are not finite and non-negative.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    sx = nstdmax * np.sqrt(sx2)
    sy = nstdmax * np.sqrt(sy2)
    sx = max(float(sx), _fallback_halfspan(float(pos[inds[0]])))
    sy = max(float(sy), _fallback_halfspan(float(pos[inds[1]])))
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

    mu = float(np.asarray(pos[ind], dtype=float))
    cov = _covariance_from_fisher(fisher)
    sigma2 = _nonnegative_or_none(cov[ind, ind])
    if sigma2 is None:
        warnings.warn(
            f"Skipping Gaussian marginal for index {ind} because the variance "
            "is not finite and non-negative.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    sigma = float(np.sqrt(sigma2))
    nstd = float(norm.ppf(0.5 * (1 + cl)))
    if ax is None:
        ax = plt.gca()
    halfspan = 1.5 * nstd * sigma
    halfspan = max(halfspan, _fallback_halfspan(mu))
    x = np.linspace(mu - halfspan, mu + halfspan, 100)
    if sigma > 0.0:
        y = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    else:
        y = np.zeros_like(x)
        y[len(y) // 2] = 1.0
    ax.plot(x, y, **kwargs)
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
