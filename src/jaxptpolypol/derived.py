"""
Derived cosmological parameter helpers.

These utilities are intended for Fisher/covariance projections from the
native sampled basis into a derived basis such as ``(Omega_m, H0, sigma8)``.
"""

from __future__ import annotations

import jax.numpy as jnp

from .params import CosmoParams

__all__ = [
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
