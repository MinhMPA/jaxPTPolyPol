"""
Derived cosmological parameter helpers.

These utilities are intended for Fisher/covariance projections from the
native sampled basis into a derived basis such as ``(Omega_m, H0, sigma8)``.
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from .params import CosmoParams

__all__ = [
    "format_derived_comparison_rows",
    "make_derived_projection_fn",
    "make_lcdm_derived_params_fn",
    "sigma8_from_linear_pk",
    "sigmaR_from_linear_pk",
]


def _spherical_tophat_window(x):
    """Fourier-space spherical top-hat window ``W(x)``."""
    x = jnp.asarray(x, dtype=jnp.float64)
    x2 = x * x
    small = jnp.abs(x) < 1e-3
    series = 1.0 - x2 / 10.0 + x2 * x2 / 280.0
    x_safe = jnp.where(small, 1.0, x)
    exact = 3.0 * (jnp.sin(x_safe) - x_safe * jnp.cos(x_safe)) / (x_safe**3)
    return jnp.where(small, series, exact)


def sigmaR_from_linear_pk(
    k,
    pk,
    *,
    radius: float = 8.0,
):
    r"""Return the linear-theory top-hat variance ``sigma_R``.

    Parameters
    ----------
    k : array_like
        Wavenumber grid in ``h/Mpc``.
    pk : array_like
        Linear matter power spectrum on the same grid in ``(Mpc/h)^3``.
    radius : float, optional
        Top-hat radius in ``Mpc/h``.
    """
    k = jnp.ravel(jnp.asarray(k, dtype=jnp.float64))
    pk = jnp.ravel(jnp.asarray(pk, dtype=jnp.float64))
    window = _spherical_tophat_window(k * float(radius))
    integrand = (k**3) * pk * (window**2) / (2.0 * jnp.pi**2)
    sigma2 = jnp.trapezoid(integrand, x=jnp.log(k))
    sigma2 = jnp.maximum(sigma2, 0.0)
    return jnp.sqrt(sigma2)


def sigma8_from_linear_pk(k, pk):
    """Return ``sigma8`` from a sampled linear power spectrum."""
    return sigmaR_from_linear_pk(k, pk, radius=8.0)


def _emulator_input_dict(
    cosmo_obj: CosmoParams,
    *,
    emulator_parameters,
    sigma8_redshift: float,
    extra_cosmo=None,
):
    """Build an emulator input dict, injecting ``z=0`` when required.

    ``extra_cosmo`` optionally supplies fixed cosmology inputs the emulator
    expects but that are absent from the sampled native basis (e.g. the
    baryon-feedback nuisances ``A_b``/``eta_b``/``logT_AGN`` when only the LCDM
    core is varied). Values are injected as constants, so they carry no
    derivative w.r.t. the sampled cosmology.
    """
    base = cosmo_obj.to_dict()
    if extra_cosmo:
        base = {
            **base,
            **{k: jnp.atleast_1d(jnp.asarray(v, dtype=jnp.float64))
               for k, v in extra_cosmo.items()},
        }
    z_value = jnp.atleast_1d(jnp.asarray(sigma8_redshift, dtype=jnp.float64))

    if emulator_parameters is None:
        return {**base, "z": z_value}

    result = {}
    for key in emulator_parameters:
        if key == "z":
            result[key] = z_value
        else:
            if key not in base:
                raise KeyError(
                    f"emulator expects cosmology key {key!r}, which is missing "
                    "from the supplied native cosmology basis"
                )
            result[key] = base[key]
    return result


def make_lcdm_derived_params_fn(
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    *,
    pklin_emulator,
    mnu_fixed: float = 0.06,
    sigma8_redshift: float = 0.0,
):
    r"""Return a differentiable map to ``(Omega_m, H0, sigma8)``.

    The returned function expects a flat cosmology vector in the native basis
    described by ``cosmo_keys``/``cosmo_sizes`` and evaluates the derived
    parameters at the fiducial point for Fisher projection.
    """
    has_mnu = "mnu" in cosmo_keys
    emulator_modes = jnp.asarray(pklin_emulator.modes, dtype=jnp.float64)
    emulator_parameters = getattr(pklin_emulator, "parameters", None)
    if emulator_parameters is not None:
        emulator_parameters = tuple(emulator_parameters)

    def derived_fn(native_params):
        cosmo_obj = CosmoParams.from_array(
            jnp.asarray(native_params, dtype=jnp.float64),
            cosmo_keys,
            cosmo_sizes,
        )
        h = cosmo_obj.h[0]
        omega_b = cosmo_obj.omega_b[0]
        omega_cdm = cosmo_obj.omega_cdm[0]
        mnu = cosmo_obj.mnu[0] if has_mnu else jnp.asarray(mnu_fixed, dtype=jnp.float64)

        omega_nu = mnu / 93.14
        omega_m = omega_b + omega_cdm + omega_nu
        Omega_m = omega_m / h**2
        H0 = 100.0 * h

        emulator_input = _emulator_input_dict(
            cosmo_obj,
            emulator_parameters=emulator_parameters,
            sigma8_redshift=sigma8_redshift,
        )
        pklin = jnp.ravel(jnp.asarray(pklin_emulator.predict(emulator_input), dtype=jnp.float64))
        sigma8 = sigma8_from_linear_pk(emulator_modes / h, pklin)

        return jnp.array([Omega_m, H0, sigma8], dtype=jnp.float64)

    return derived_fn


def make_derived_projection_fn(
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    *,
    pklin_emulator,
    fiducial_native,
    source_indices,
    native_indices,
    mnu_fixed: float = 0.06,
    sigma8_redshift: float = 0.0,
):
    r"""Return ``(derived_fn, derived_names)`` mapping a COMPARISON basis vector
    to the reported derived basis.

    This is the wrapper the joint MCMC notebooks need around
    :func:`make_lcdm_derived_params_fn`, which alone is not enough because the
    sampled/comparison basis is neither the native cosmology basis nor in the
    reported order. Four things happen here:

    1. **Native-slot scatter.** ``theta`` lives in the comparison basis, whose
       ``source_indices`` map onto ``native_indices`` of the native cosmology
       vector. Everything else keeps its ``fiducial_native`` value, so native
       parameters that are not varied enter as constants and carry no
       derivative.
    2. **The library call**, which returns ``(Omega_m, H0, sigma8)``.
    3. **The reorder** to the reported axis order ``(Omega_m, sigma8, H0)``.
    4. **The mnu prepend** when ``'mnu'`` is in ``cosmo_keys``: ``Sigma m_nu`` is
       an IDENTITY coordinate of the map, taken straight off the scattered
       native vector, so the projected chain column is bit-identical to the
       sampled one.

    Parameters absent from the map (tau, the bias block, theta_lin, ...) are
    marginalized by simply not entering it, while their correlations still widen
    the projected covariance through ``C = F^-1``.

    Parameters
    ----------
    cosmo_keys, cosmo_sizes : tuple
        The NATIVE cosmology basis, as on :class:`~.params.CosmoParams`.
    pklin_emulator : object
        Linear-``P(k)`` emulator, forwarded to
        :func:`make_lcdm_derived_params_fn`.
    fiducial_native : array_like, shape (n_native,)
        Fiducial native cosmology vector; supplies every slot not written by
        ``native_indices``.
    source_indices : sequence of int
        Positions in the comparison basis to read.
    native_indices : sequence of int
        Native-vector slots they are written to; same length and order as
        ``source_indices``.
    mnu_fixed : float, optional
        Neutrino mass used when ``'mnu'`` is not part of ``cosmo_keys``.
    sigma8_redshift : float, optional
        Redshift at which ``sigma8`` is evaluated.

    Returns
    -------
    derived_fn : callable
        ``theta_comparison -> derived vector``; jit- and vmap-able.
    derived_names : tuple of str
        ``('Omega_m', 'sigma8', 'H0')``, or ``('mnu', 'Omega_m', 'sigma8',
        'H0')`` when ``'mnu'`` is in ``cosmo_keys``.
    """
    source_indices = [int(i) for i in source_indices]
    native_indices = [int(i) for i in native_indices]
    if len(source_indices) != len(native_indices):
        raise ValueError(
            f"source_indices has {len(source_indices)} entries but "
            f"native_indices has {len(native_indices)}; they index the same map"
        )

    fid_native = jnp.asarray(fiducial_native, dtype=jnp.float64)
    if fid_native.ndim != 1:
        raise ValueError(
            f"fiducial_native must be a flat native cosmology vector, got shape "
            f"{tuple(fid_native.shape)}"
        )
    n_native = int(fid_native.shape[0])
    if native_indices and not (0 <= min(native_indices)
                               and max(native_indices) < n_native):
        raise ValueError(
            f"native_indices {native_indices} fall outside the {n_native}-slot "
            "native cosmology vector"
        )

    core = make_lcdm_derived_params_fn(
        cosmo_keys, cosmo_sizes, pklin_emulator=pklin_emulator,
        mnu_fixed=mnu_fixed, sigma8_redshift=sigma8_redshift,
    )

    src_idx = jnp.array(source_indices)
    nat_idx = jnp.array(native_indices)

    has_mnu = "mnu" in cosmo_keys
    if has_mnu:
        # Flat offset, not ``cosmo_keys.index``: identical whenever every
        # cosmology entry is a scalar, correct also when one is not.
        mnu_nat = int(np.sum(np.asarray(cosmo_sizes[:cosmo_keys.index("mnu")])))
        derived_names = ("mnu", "Omega_m", "sigma8", "H0")

        def derived_fn(theta):
            native = fid_native.at[nat_idx].set(
                jnp.asarray(theta, dtype=jnp.float64)[src_idx])
            Omega_m, H0, sigma8 = core(native)
            return jnp.array([native[mnu_nat], Omega_m, sigma8, H0],
                             dtype=jnp.float64)
    else:
        derived_names = ("Omega_m", "sigma8", "H0")

        def derived_fn(theta):
            native = fid_native.at[nat_idx].set(
                jnp.asarray(theta, dtype=jnp.float64)[src_idx])
            Omega_m, H0, sigma8 = core(native)
            return jnp.array([Omega_m, sigma8, H0], dtype=jnp.float64)

    return derived_fn, derived_names


def format_derived_comparison_rows(names, fiducial, samples, fisher_sigma):
    """Format the derived Fisher-vs-MCMC comparison table.

    Pure and plotting-free: given the projected chain and the projected Fisher
    widths it returns the printable lines plus the summary statistics behind
    them, so the caller can assert on the numbers it just printed.

    ``fisher_sigma`` is passed in rather than re-derived from the projected
    Fisher on purpose: the notebooks take it as ``sqrt(diag(cov_derived))`` from
    :func:`~.inference.project_fisher_to_derived`, and ``sqrt(diag(inv(
    F_derived)))`` differs from that in the last bits.

    Returns
    -------
    lines : list of str
        Header row, one row per derived parameter, then the residual-pull line.
    mean, sigma, pulls : ndarray
        Chain mean, chain standard deviation, and
        ``(mean - fiducial) / fisher_sigma``.
    """
    names = tuple(names)
    fid = np.asarray(fiducial, dtype=float)
    samples = np.asarray(samples, dtype=float)
    fisher_sigma = np.asarray(fisher_sigma, dtype=float)
    if samples.ndim != 2:
        raise ValueError(
            f"samples must be (n_draws, n_derived), got shape {samples.shape}")
    for label, arr in (("fiducial", fid), ("fisher_sigma", fisher_sigma)):
        if arr.shape != (len(names),):
            raise ValueError(
                f"{label} has shape {arr.shape}, expected ({len(names)},) to "
                f"match names {names}")
    if samples.shape[1] != len(names):
        raise ValueError(
            f"samples has {samples.shape[1]} columns but there are "
            f"{len(names)} derived names {names}")

    mean = samples.mean(axis=0)
    sigma = samples.std(axis=0)
    pulls = (mean - fid) / fisher_sigma

    lines = [f"{'param':>9s} {'fid':>10s} {'MCMC mean':>12s} "
             f"{'Fisher sig':>12s} {'MCMC sig':>12s} {'ratio':>7s}"]
    for i, name in enumerate(names):
        lines.append(
            f"{name:>9s} {fid[i]:10.5g} {mean[i]:12.5g} "
            f"{fisher_sigma[i]:12.5g} {sigma[i]:12.5g} "
            f"{sigma[i] / fisher_sigma[i]:7.2f}")
    lines.append("residual pulls (sigma_F units): "
                 + "  ".join(f"{n}={pulls[i]:+.2f}"
                             for i, n in enumerate(names)))
    return lines, mean, sigma, pulls
