"""
Theory prediction factory: power spectrum multipoles with optional AP effect.

The central function :func:`make_pk_ell_fn` returns a closure

    ``pk_fn(params, *, k) -> jnp.ndarray``

that is ready to be JIT-compiled and differentiated::

    pk_fn = make_pk_ell_fn(...)
    jac   = jax.jacfwd(jax.jit(pk_fn))(packed_params, k=k)

All configuration (emulators, fiducial cosmology, AP on/off, single- vs
multi-bin) is captured inside the closure so that the caller never needs
to pass ``static_argnames``.
"""

from __future__ import annotations

import jax
import numpy as np
import jax.numpy as jnp

from .params import CosmoParams, SurveyParams, unpack_params, unpack_multibin_params
from .model import CosmoEmulator, PS1LoopModel

__all__ = [
    "compute_fiducial_distances",
    "make_multipole_projector",
    "make_pk_ell_fn",
]


def _prepare_linear_interp(x_grid, x_targets):
    """Precompute linear-interpolation indices and weights for static targets."""
    idx_lo = np.searchsorted(x_grid, x_targets, side="right") - 1
    idx_lo = np.clip(idx_lo, 0, len(x_grid) - 2)
    idx_hi = idx_lo + 1

    x_lo = x_grid[idx_lo]
    x_hi = x_grid[idx_hi]
    weights = (x_targets - x_lo) / (x_hi - x_lo)

    return (
        jnp.asarray(idx_lo, dtype=jnp.int32),
        jnp.asarray(idx_hi, dtype=jnp.int32),
        jnp.asarray(weights),
    )


# ---------------------------------------------------------------------------
# Fiducial distances helper
# ---------------------------------------------------------------------------
def compute_fiducial_distances(
    cosmo: CosmoParams,
    z_bins: tuple[float, ...],
    mnu: float | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Pre-compute fiducial H(z) and D_A(z) for each redshift bin.

    These are evaluated **once** (outside JIT) and become static constants
    inside the closure returned by :func:`make_pk_ell_fn`.

    Parameters
    ----------
    cosmo : CosmoParams
        Fiducial cosmology.
    z_bins : tuple of float
        Central redshifts of the bins to evaluate.
    mnu : float, optional
        Neutrino mass sum [eV].  If *None*, looked up from ``cosmo``
        (key ``'mnu'``); if absent there, defaults to 0.06 eV.

    Returns
    -------
    Hz_fid : tuple of float
        Fiducial H(z) per bin [km/s/Mpc].
    DAz_fid : tuple of float
        Fiducial angular diameter distance per bin [Mpc].
    """
    from ps_1loop_jax import background as bg

    omb = float(cosmo.omega_b[0])
    omc = float(cosmo.omega_cdm[0])
    h = float(cosmo.h[0])
    if mnu is None:
        try:
            mnu = float(cosmo.mnu[0])
        except (KeyError, IndexError, ValueError):
            mnu = 0.06

    Hz_fid: list[float] = []
    DAz_fid: list[float] = []
    for z in z_bins:
        Hz_fid.append(float(bg.Hz(omb, omc, h, z, mnu)))
        DAz_fid.append(float(bg.angular_diameter_distance(omb, omc, h, z, mnu)))
    return tuple(Hz_fid), tuple(DAz_fid)


# ---------------------------------------------------------------------------
# Gauss-Legendre multipole projector
# ---------------------------------------------------------------------------
def make_multipole_projector(ells, n_gl=16):
    r"""Pre-compute Gauss-Legendre nodes and multipole projection weights.

    The multipole integral

    .. math::
        P_\ell(k) = (2\ell+1) \int_0^1 P(k,\mu)\,\mathcal{L}_\ell(\mu)\,d\mu

    is approximated by Gauss-Legendre quadrature:

    .. math::
        P_\ell(k) \approx \sum_i W_{\ell i}\,P(k,\mu_i)

    where :math:`W_{\ell i} = (2\ell+1)\,w_i\,\mathcal{L}_\ell(\mu_i)` are
    pretabulated constants.

    Parameters
    ----------
    ells : tuple of int
        Multipole orders (e.g. ``(0, 2, 4)``).
    n_gl : int
        Number of Gauss-Legendre nodes.  Default 16, which is more than
        sufficient for smooth integrands.  For reference, GL with 16 nodes
        integrates polynomials of degree ≤ 31 *exactly*.

        The 1-loop RSD model produces integrands with polynomial degree
        up to ~12 in μ (from Z-kernel terms like f³μ⁴ × L₄(μ)).
        ``n_gl >= 8`` is required for exact integration of these terms;
        ``n_gl = 16`` provides a safe margin.

        **Note on accuracy with IR resummation**: when ``do_irres=True``,
        the integrand also contains the BAO damping factor
        :math:`e^{-k^2 \Sigma^2(\mu)}`, which is non-polynomial in μ.
        As a result, GL(16) is not exact for this integrand and agrees
        with Simpson(256) to ~10⁻⁵ (hexadecapole) through ~10⁻⁷
        (monopole).  The dominant error is Simpson's O(h⁴) rule; GL is
        the more accurate of the two.  Both are far below statistical
        uncertainties (~10⁻²).

    Returns
    -------
    mu_gl : jnp.ndarray, shape (n_gl,)
        Gauss-Legendre nodes on [0, 1], sorted ascending.
    W : jnp.ndarray, shape (n_ell, n_gl)
        Projection weight matrix.  ``P_ells = P(k, mu_gl) @ W.T``
        gives shape ``(n_k, n_ell)``.
    """
    max_ell = max(ells)
    # 1-loop RSD integrands have polynomial degree up to ~(max_ell + 8)
    # GL(N) is exact for degree <= 2N-1
    min_n_gl = (max_ell + 8 + 2) // 2  # ceil((max_ell + 8 + 1) / 2)
    if n_gl < min_n_gl:
        raise ValueError(
            f"n_gl={n_gl} is too small for ells={ells}. "
            f"Need at least {min_n_gl} GL nodes to exactly integrate "
            f"1-loop RSD integrands (polynomial degree ~{max_ell + 8})."
        )

    # Standard GL on [-1, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_gl)

    # Transform to [0, 1]: mu = (x + 1) / 2,  w' = w / 2
    mu_np = (nodes + 1.0) / 2.0
    w_np = weights / 2.0

    # Legendre polynomials at GL nodes (numpy, computed once)
    from numpy.polynomial.legendre import legval
    W_np = np.zeros((len(ells), n_gl), dtype=np.float64)
    for i, ell in enumerate(ells):
        coeffs = np.zeros(ell + 1)
        coeffs[ell] = 1.0
        W_np[i] = (2 * ell + 1) * w_np * legval(mu_np, coeffs)

    # Use default JAX dtype (respects jax_enable_x64 setting)
    mu_gl = jnp.asarray(mu_np)
    W = jnp.asarray(W_np)

    return mu_gl, W


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_pk_ell_fn(
    *,
    ells: tuple[int, ...],
    pklin_emulator: CosmoEmulator,
    ps1loop_model: PS1LoopModel,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    survey_keys: tuple,
    # --- AP configuration ---
    ap: bool = True,
    Hz_fid: tuple[float, ...] | None = None,
    DAz_fid: tuple[float, ...] | None = None,
    # --- Multi-bin configuration ---
    z_bins: tuple[float, ...] | None = None,
    # --- Neutrino mass ---
    mnu_fixed: float = 0.06,
    # --- Quadrature ---
    num: int = 256,
    n_gl: int | None = None,
    # --- AP background acceleration ---
    background_mode: str = "direct",
    background_nz: int = 256,
):
    r"""Build a differentiable theory-prediction function.

    Returns a closure ``pk_fn(params, \*, k)`` whose only traced arguments
    are the flat parameter vector *params* and the wavenumber grid *k*.

    Parameters
    ----------
    ells : tuple of int
        Multipole orders, e.g. ``(0, 2, 4)``.
    pklin_emulator : CosmoEmulator
        Linear power spectrum emulator.
    ps1loop_model : PS1LoopModel
        1-loop power spectrum model.
    cosmo_keys, cosmo_sizes : tuple
        Static metadata of :class:`CosmoParams` (keys and per-key sizes).
    survey_keys : tuple
        Static flat key list of :class:`SurveyParams`.
    ap : bool
        If *True* (default), include the Alcock-Paczyński distortion.
        Requires *Hz_fid* and *DAz_fid* (or compute them with
        :func:`compute_fiducial_distances`).
    Hz_fid : tuple of float, optional
        Fiducial H(z) per bin [km/s/Mpc].
    DAz_fid : tuple of float, optional
        Fiducial D_A(z) per bin [Mpc].
    z_bins : tuple of float, optional
        Redshift per bin.  If *None*, single-bin mode: z is read from
        the parameter vector.  If given, multi-bin mode: z per bin is
        static (not differentiated) and survey parameters are per-bin.
    mnu_fixed : float
        Neutrino mass [eV] used when ``'mnu'`` is **not** among the
        varied cosmological parameters.  Default 0.06.
    num : int
        Number of uniform mu-quadrature points for Simpson's rule
        integration.  Only used when ``n_gl`` is *None* (legacy mode).
        Default 256.
    n_gl : int or None
        If given, use Gauss-Legendre quadrature with this many nodes
        for the multipole projection instead of Simpson's rule.
        Recommended: 16 (sufficient for smooth integrands, ~16× fewer
        mu-points than the default ``num=256``).  Also computes P(k,μ)
        only **once** per bin instead of once per (bin, ℓ).
    background_mode : {"direct", "tabulated"}
        Strategy for AP background distances in multi-bin mode.

        - ``"direct"`` (default): call the background routines directly
          for each bin.
        - ``"tabulated"``: compute :math:`\chi(z)` once on a fixed
          redshift grid spanning the survey bins, then linearly
          interpolate :math:`D_A(z)=\chi(z)/(1+z)` at the bin centers.

        ``"tabulated"`` removes repeated per-bin Simpson integrations in
        the AP path and is typically much faster for multi-bin MCMC.
    background_nz : int
        Number of redshift grid points used when
        ``background_mode='tabulated'``. Must be at least 2. Ignored in
        single-bin mode and when ``background_mode='direct'``.

    Returns
    -------
    pk_fn : callable
        ``pk_fn(params, *, k) -> jnp.ndarray`` of shape
        ``(n_bins * n_ell * n_k,)``.
    """
    from ps_1loop_jax import background as bg

    # ---- Derived constants ------------------------------------------------
    has_mnu = "mnu" in cosmo_keys
    multi_bin = z_bins is not None
    n_bins = len(z_bins) if multi_bin else 1
    n_cosmo = sum(cosmo_sizes)
    n_survey = len(survey_keys)

    # ---- Validate ---------------------------------------------------------
    if ap:
        if Hz_fid is None or DAz_fid is None:
            raise ValueError(
                "ap=True requires Hz_fid and DAz_fid.  Use "
                "compute_fiducial_distances() to obtain them."
            )
        if len(Hz_fid) != n_bins or len(DAz_fid) != n_bins:
            raise ValueError(
                f"Hz_fid/DAz_fid length ({len(Hz_fid)}) != n_bins ({n_bins})"
            )
    if background_mode not in ("direct", "tabulated"):
        raise ValueError(
            "background_mode must be 'direct' or 'tabulated', "
            f"got {background_mode!r}"
        )
    if background_mode == "tabulated" and background_nz < 2:
        raise ValueError("background_nz must be at least 2.")

    # Cache the emulator modes array (static)
    _emulator_modes = jnp.array(pklin_emulator.modes)
    _z_bins_arr = None if z_bins is None else jnp.asarray(z_bins)
    _one_plus_z = None if z_bins is None else (1.0 + _z_bins_arr)
    use_tabulated_background = ap and multi_bin and background_mode == "tabulated"

    if use_tabulated_background:
        z_max = max(z_bins)
        z_bg_grid = np.linspace(0.0, z_max, background_nz, dtype=np.float64)
        idx_lo, idx_hi, interp_w = _prepare_linear_interp(
            z_bg_grid, np.asarray(z_bins, dtype=np.float64)
        )
        _z_bg_grid = jnp.asarray(z_bg_grid)
        _bg_idx_lo = idx_lo
        _bg_idx_hi = idx_hi
        _bg_interp_w = interp_w

    # ---- Gauss-Legendre projector (if requested) --------------------------
    use_gl = n_gl is not None
    if use_gl:
        _mu_gl, _W_proj = make_multipole_projector(ells, n_gl)

    # ---- Build the closure ------------------------------------------------
    def pk_ell_fn(params, *, k):
        # 1. Unpack cosmological parameters (shared across bins)
        cosmo_obj = CosmoParams.from_array(
            params[:n_cosmo], cosmo_keys, cosmo_sizes
        )

        h = cosmo_obj.h[0]
        omb = cosmo_obj.omega_b[0]
        omc = cosmo_obj.omega_cdm[0]
        mnu = cosmo_obj.mnu[0] if has_mnu else mnu_fixed

        # Emulator k grid (shared, only depends on h)
        klin = _emulator_modes / h  # [Mpc^{-1}] -> [h/Mpc]
        cosmo_dict_base = cosmo_obj.to_dict()

        if ap and multi_bin:
            Hz_true_all = bg.Hz(omb, omc, h, _z_bins_arr, mnu)
            if use_tabulated_background:
                chi_grid = bg.chi(omb, omc, h, _z_bg_grid, mnu)
                chi_lo = chi_grid[_bg_idx_lo]
                chi_hi = chi_grid[_bg_idx_hi]
                chi_interp = (
                    (1.0 - _bg_interp_w) * chi_lo + _bg_interp_w * chi_hi
                )
                DAz_true_all = chi_interp / _one_plus_z
            else:
                DAz_true_all = jax.vmap(
                    lambda z_i: bg.angular_diameter_distance(
                        omb, omc, h, z_i, mnu
                    )
                )(_z_bins_arr)

        all_pk = []
        for i in range(n_bins):
            # 2a. Redshift for this bin
            z_i = z_bins[i] if multi_bin else cosmo_obj.z[0]

            # 2b. Unpack this bin's survey params
            if multi_bin:
                offset = n_cosmo + i * n_survey
                survey_obj = SurveyParams.from_array(
                    params[offset : offset + n_survey], survey_keys
                )
            else:
                survey_obj = SurveyParams.from_array(
                    params[n_cosmo:], survey_keys
                )

            # 3. Emulator prediction (override z for multi-bin)
            cosmo_dict = cosmo_dict_base
            if multi_bin:
                cosmo_dict = {**cosmo_dict_base, "z": jnp.atleast_1d(z_i)}
            pklin = pklin_emulator.predict(cosmo_dict)
            pklin_data = {"k": klin, "pk": pklin}

            # 4. Growth rate
            f_growth = bg.growth_rate_approx(omb, omc, h, z_i, mnu)

            # 5. ps1loop params dict
            ps1loop_params = {"h": h, "f": f_growth, **survey_obj.to_dict()}

            # 6. Compute multipoles
            if ap:
                if multi_bin:
                    Hz_true = Hz_true_all[i]
                    DAz_true = DAz_true_all[i]
                else:
                    Hz_true = bg.Hz(omb, omc, h, z_i, mnu)
                    DAz_true = bg.angular_diameter_distance(
                        omb, omc, h, z_i, mnu
                    )
                alpha_perp = DAz_true / DAz_fid[i]
                alpha_para = Hz_fid[i] / Hz_true

                if use_gl:
                    # Gauss-Legendre: compute P(k, mu) once, project all ells
                    pkmu = ps1loop_model.get_pkmu_ref(
                        k, _mu_gl, alpha_perp, alpha_para,
                        pklin_data, ps1loop_params,
                    )
                    # pkmu: (n_k, n_gl), _W_proj: (n_ell, n_gl)
                    # result: (n_k, n_ell) -> flatten to (n_ell * n_k,)
                    pk_ells_2d = pkmu @ _W_proj.T
                    pk_ells_i = pk_ells_2d.T.reshape(-1)
                else:
                    pk_ells_i = jnp.concatenate([
                        ps1loop_model.get_pk_ell_ref(
                            k, ell, alpha_perp, alpha_para,
                            pklin_data, ps1loop_params, num=num,
                        )
                        for ell in ells
                    ])
            else:
                if use_gl:
                    pkmu = ps1loop_model.get_pkmu(
                        k, _mu_gl, pklin_data, ps1loop_params,
                    )
                    pk_ells_2d = pkmu @ _W_proj.T
                    pk_ells_i = pk_ells_2d.T.reshape(-1)
                else:
                    pk_ells_i = jnp.concatenate([
                        ps1loop_model.get_pk_ell(
                            k, ell, pklin_data, ps1loop_params, num=num,
                        )
                        for ell in ells
                    ])

            all_pk.append(pk_ells_i)

        return jnp.concatenate(all_pk)

    # Attach metadata for introspection
    pk_ell_fn.ells = ells
    pk_ell_fn.ap = ap
    pk_ell_fn.n_bins = n_bins
    pk_ell_fn.z_bins = z_bins
    pk_ell_fn.has_mnu = has_mnu
    pk_ell_fn.num = num
    pk_ell_fn.n_gl = n_gl
    pk_ell_fn.background_mode = background_mode
    pk_ell_fn.background_nz = background_nz

    return pk_ell_fn
