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

from collections.abc import Sequence

import jax
import numpy as np
import jax.numpy as jnp

from .params import CosmoParams, FullShapeSurveyParams
from .model import BispectrumTreeModel, CosmoEmulator, PS1LoopModel
from .covariance import (
    gaussian_bispectrum_covariance,
    gaussian_bispectrum_covariance_multibin,
    gaussian_joint_covariance,
    gaussian_joint_covariance_multibin,
)

__all__ = [
    "build_bispectrum_triangles_from_k_grid",
    "compute_fiducial_distances",
    "kaiser_power_multipoles",
    "make_multipole_projector",
    "make_pk_ell_fn",
    "make_bk0_fn",
    "make_joint_pk_bk_fn",
    "make_gaussian_bk0_covariance_fn",
    "make_gaussian_joint_covariance_fn",
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


def _select_bin_input(arg, b: int, n_bins: int, name: str):
    """Return a shared input or the ``b``-th element of a per-bin sequence."""
    if isinstance(arg, (list, tuple)):
        if len(arg) != n_bins:
            raise ValueError(
                f"{name} sequence length ({len(arg)}) != n_bins ({n_bins})"
            )
        return arg[b]
    return arg


def _as_triangle_array(triangles, name: str = "triangles") -> jnp.ndarray:
    """Validate a triangle array of shape ``(n_tri, 3)``."""
    triangles = jnp.asarray(triangles)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape (n_tri, 3), got {triangles.shape}"
        )
    return triangles


def build_bispectrum_triangles_from_k_grid(
    k: jnp.ndarray,
    *,
    k_min: float = 0.02,
    k_max: float = 0.08,
    dk: float | jnp.ndarray | None = None,
    closure_tol: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Build canonical bispectrum triangles from the 1-d power-spectrum grid.

    The returned triangle list uses the same grid points as the input power
    spectrum, keeps only sides strictly inside ``(k_min, k_max)``, imposes the
    canonical ordering ``k1 >= k2 >= k3``, and applies the triangle-closure
    condition ``k1 <= k2 + k3``.

    Parameters
    ----------
    k : array, shape ``(n_k,)``
        Monotonic 1-d power-spectrum wavenumber grid.
    k_min, k_max : float
        Strict lower and upper cuts applied to each triangle side.
    dk : float or array, optional
        Power-spectrum bin width(s). If omitted, a scalar width is inferred
        from a uniformly spaced ``k`` grid. If an array is supplied it must
        have shape ``(n_k,)`` and the returned triangle widths are gathered
        edge-by-edge in canonical triangle order.
    closure_tol : float, optional
        Absolute tolerance used in the closure test. By default a small
        machine-precision-scaled tolerance is used.

    Returns
    -------
    triangles : array, shape ``(n_tri, 3)``
        Canonically ordered triangle centers.
    triangle_dk : scalar or array
        Matching triangle bin widths. A scalar is returned when ``dk`` is
        scalar or inferred from a uniform grid; otherwise the shape is
        ``(n_tri, 3)``.
    """
    k_np = np.asarray(k, dtype=np.float64)
    if k_np.ndim != 1:
        raise ValueError(f"k must be 1-d, got shape {k_np.shape}")
    if k_np.size == 0:
        raise ValueError("k must contain at least one grid point")
    if np.any(np.diff(k_np) <= 0.0):
        raise ValueError("k must be strictly increasing")
    if not k_min < k_max:
        raise ValueError(
            f"k_min must be smaller than k_max, got {k_min} >= {k_max}"
        )

    if dk is None:
        if k_np.size < 2:
            raise ValueError("dk=None requires at least two k-grid points")
        k_step = np.diff(k_np)
        if not np.allclose(k_step, k_step[0]):
            raise ValueError(
                "dk=None requires a uniformly spaced k grid; pass dk "
                "explicitly for non-uniform grids"
            )
        dk_spec: float | np.ndarray = float(k_step[0])
    else:
        dk_np = np.asarray(dk, dtype=np.float64)
        if dk_np.ndim == 0:
            dk_spec = float(dk_np)
        elif dk_np.shape == k_np.shape:
            dk_spec = dk_np
        else:
            raise ValueError(
                "dk must be a scalar or an array with the same shape as k, "
                f"got {dk_np.shape}"
            )

    keep = (k_np > k_min) & (k_np < k_max)
    selected_idx = np.flatnonzero(keep)[::-1]
    selected_k = k_np[selected_idx]
    selected_dk = None if np.isscalar(dk_spec) else dk_spec[selected_idx]

    tol = closure_tol
    if tol is None:
        scale = 0.0 if selected_k.size == 0 else float(np.max(selected_k))
        tol = 10.0 * np.finfo(np.float64).eps * max(scale, 1.0)

    triangles: list[tuple[float, float, float]] = []
    triangle_dk_rows: list[tuple[float, float, float]] = []
    for i, k1 in enumerate(selected_k):
        for j in range(i, selected_k.size):
            k2 = selected_k[j]
            for l in range(j, selected_k.size):
                k3 = selected_k[l]
                if k1 <= k2 + k3 + tol:
                    triangles.append((k1, k2, k3))
                    if selected_dk is not None:
                        triangle_dk_rows.append(
                            (selected_dk[i], selected_dk[j], selected_dk[l])
                        )

    triangles_arr = jnp.asarray(triangles, dtype=jnp.asarray(k).dtype)
    if triangles_arr.size == 0:
        triangles_arr = jnp.empty((0, 3), dtype=jnp.asarray(k).dtype)

    if selected_dk is None:
        triangle_dk_out = jnp.asarray(dk_spec, dtype=jnp.asarray(k).dtype)
    else:
        triangle_dk_out = jnp.asarray(
            triangle_dk_rows,
            dtype=jnp.result_type(jnp.asarray(k), jnp.asarray(dk)),
        )
        if triangle_dk_out.size == 0:
            triangle_dk_out = jnp.empty(
                (0, 3),
                dtype=jnp.result_type(jnp.asarray(k), jnp.asarray(dk)),
            )

    return triangles_arr, triangle_dk_out


def _legendre_eval(ell: int, mu: jnp.ndarray) -> jnp.ndarray:
    """Evaluate ``L_ell(mu)`` for the supported even multipoles."""
    if ell == 0:
        return jnp.ones_like(mu)
    if ell == 2:
        return 0.5 * (3.0 * mu**2 - 1.0)
    if ell == 4:
        return 0.125 * (35.0 * mu**4 - 30.0 * mu**2 + 3.0)
    raise ValueError(f"unsupported multipole ell={ell}; only 0, 2, 4 are supported")


def _power_shot_noise(params: dict) -> jnp.ndarray:
    """Return the constant shot-noise contribution added to ``Pt0``."""
    stoch = params.get("stoch", {})
    if isinstance(stoch, dict):
        p_shot = stoch.get("P_shot", stoch.get("Pshot", 0.0))
    else:
        p_shot = 0.0

    ndens = params.get("ndens", None)
    if ndens is None:
        return jnp.asarray(0.0)
    return jnp.asarray(p_shot, dtype=float) / jnp.asarray(ndens, dtype=float)


def kaiser_power_multipoles(
    k: jnp.ndarray,
    pk_data: dict[str, jnp.ndarray],
    *,
    b1,
    f,
    shot_noise=0.0,
    alpha_perp=None,
    alpha_para=None,
    ells: tuple[int, ...] = (0, 2, 4),
    n_gl: int = 16,
) -> jnp.ndarray:
    r"""Return Kaiser redshift-space power multipoles on a target ``k`` grid.

    Parameters
    ----------
    k : array, shape ``(n_k,)``
        Target wavenumber grid in ``[h/Mpc]``.
    pk_data : dict
        Linear power spectrum data with keys ``"k"`` and ``"pk"``.
    b1, f : float
        Linear bias and growth rate entering ``(b1 + f \mu^2)^2 P_lin``.
    shot_noise : float, optional
        Constant shot-noise term added to the total monopole.
    alpha_perp, alpha_para : float, optional
        If both are supplied, compute the reference-frame multipoles after AP
        remapping and Jacobian rescaling.
    ells : tuple of int
        Multipoles to return. Supported values are ``0``, ``2``, and ``4``.
    n_gl : int
        Number of Gauss-Legendre nodes used for the AP projection path.

    Returns
    -------
    pk_ell : array, shape ``(len(ells), n_k)``
        Multipoles ordered as requested in ``ells``.
    """
    k = jnp.atleast_1d(jnp.asarray(k, dtype=float))
    k_lin = jnp.asarray(pk_data["k"], dtype=float)
    p_lin = jnp.asarray(pk_data["pk"], dtype=float)
    b1 = jnp.asarray(b1, dtype=float)
    f = jnp.asarray(f, dtype=float)
    shot_noise = jnp.asarray(shot_noise, dtype=float)

    if alpha_perp is None and alpha_para is None:
        p_lin_k = jnp.interp(k, k_lin, p_lin)
        pref0 = b1**2 + (2.0 / 3.0) * b1 * f + (1.0 / 5.0) * f**2
        pref2 = (4.0 / 3.0) * b1 * f + (4.0 / 7.0) * f**2
        pref4 = (8.0 / 35.0) * f**2
        mapping = {
            0: pref0 * p_lin_k + shot_noise,
            2: pref2 * p_lin_k,
            4: pref4 * p_lin_k,
        }
        return jnp.stack([mapping[ell] for ell in ells], axis=0)

    if alpha_perp is None or alpha_para is None:
        raise ValueError(
            "alpha_perp and alpha_para must be both provided or both omitted"
        )

    mu_gl, w_proj = make_multipole_projector(ells, n_gl=n_gl)
    F = jnp.asarray(alpha_para, dtype=float) / jnp.asarray(alpha_perp, dtype=float)
    mu_ref = mu_gl[None, :]
    ap_fac = jnp.sqrt(1.0 + mu_ref**2 * (1.0 / F**2 - 1.0))
    k_true = k[:, None] / jnp.asarray(alpha_perp, dtype=float) * ap_fac
    mu_true = mu_ref / F / ap_fac
    p_lin_true = jnp.interp(k_true.reshape(-1), k_lin, p_lin).reshape(k_true.shape)
    p_true = (b1 + f * mu_true**2) ** 2 * p_lin_true + shot_noise
    p_ref = p_true / (
        jnp.asarray(alpha_perp, dtype=float) ** 2
        * jnp.asarray(alpha_para, dtype=float)
    )
    pk_ells_2d = p_ref @ w_proj.T
    return pk_ells_2d.T


def _make_theory_context_evaluator(
    *,
    pklin_emulator: CosmoEmulator,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    survey_keys: tuple,
    ap: bool,
    Hz_fid: tuple[float, ...] | None,
    DAz_fid: tuple[float, ...] | None,
    z_bins: tuple[float, ...] | None,
    mnu_fixed: float,
    background_mode: str,
    background_nz: int,
):
    """Build a closure that evaluates per-bin theory inputs from packed params."""
    from ps_1loop_jax import background as bg

    has_mnu = "mnu" in cosmo_keys
    multi_bin = z_bins is not None
    n_bins = len(z_bins) if multi_bin else 1
    n_cosmo = sum(cosmo_sizes)
    n_survey = len(survey_keys)

    if ap:
        if Hz_fid is None or DAz_fid is None:
            raise ValueError(
                "ap=True requires Hz_fid and DAz_fid. Use "
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

    emulator_modes = jnp.asarray(pklin_emulator.modes)
    z_bins_arr = None if z_bins is None else jnp.asarray(z_bins)
    one_plus_z = None if z_bins is None else (1.0 + z_bins_arr)
    use_tabulated_background = ap and multi_bin and background_mode == "tabulated"

    if use_tabulated_background:
        z_max = max(z_bins)
        z_bg_grid = np.linspace(0.0, z_max, background_nz, dtype=np.float64)
        idx_lo, idx_hi, interp_w = _prepare_linear_interp(
            z_bg_grid, np.asarray(z_bins, dtype=np.float64)
        )
        z_bg_grid = jnp.asarray(z_bg_grid)

    def evaluate_contexts(params):
        cosmo_obj = CosmoParams.from_array(params[:n_cosmo], cosmo_keys, cosmo_sizes)

        h = cosmo_obj.h[0]
        omb = cosmo_obj.omega_b[0]
        omc = cosmo_obj.omega_cdm[0]
        mnu = cosmo_obj.mnu[0] if has_mnu else mnu_fixed

        klin = emulator_modes / h
        cosmo_dict_base = cosmo_obj.to_dict()

        if ap and multi_bin:
            Hz_true_all = bg.Hz(omb, omc, h, z_bins_arr, mnu)
            if use_tabulated_background:
                chi_grid = bg.chi(omb, omc, h, z_bg_grid, mnu)
                chi_lo = chi_grid[idx_lo]
                chi_hi = chi_grid[idx_hi]
                chi_interp = (1.0 - interp_w) * chi_lo + interp_w * chi_hi
                DAz_true_all = chi_interp / one_plus_z
            else:
                DAz_true_all = jax.vmap(
                    lambda z_i: bg.angular_diameter_distance(omb, omc, h, z_i, mnu)
                )(z_bins_arr)

        contexts = []
        for i in range(n_bins):
            z_i = z_bins[i] if multi_bin else cosmo_obj.z[0]

            if multi_bin:
                offset = n_cosmo + i * n_survey
                survey_obj = FullShapeSurveyParams.from_array(
                    params[offset : offset + n_survey], survey_keys
                )
            else:
                survey_obj = FullShapeSurveyParams.from_array(
                    params[n_cosmo:], survey_keys
                )

            cosmo_dict = cosmo_dict_base
            if multi_bin:
                cosmo_dict = {**cosmo_dict_base, "z": jnp.atleast_1d(z_i)}

            pklin = pklin_emulator.predict(cosmo_dict)
            pklin_data = {"k": klin, "pk": pklin}
            f_growth = bg.growth_rate_approx(omb, omc, h, z_i, mnu)
            pk_params = {"h": h, "f": f_growth, **survey_obj.to_model_dict("pk")}
            bk_params = {"h": h, "f": f_growth, **survey_obj.to_model_dict("bk")}

            alpha_perp = None
            alpha_para = None
            if ap:
                if multi_bin:
                    Hz_true = Hz_true_all[i]
                    DAz_true = DAz_true_all[i]
                else:
                    Hz_true = bg.Hz(omb, omc, h, z_i, mnu)
                    DAz_true = bg.angular_diameter_distance(omb, omc, h, z_i, mnu)
                alpha_perp = DAz_true / DAz_fid[i]
                alpha_para = Hz_fid[i] / Hz_true

            contexts.append(
                {
                    "z": z_i,
                    "pklin_data": pklin_data,
                    "pk_params": pk_params,
                    "bk_params": bk_params,
                    "alpha_perp": alpha_perp,
                    "alpha_para": alpha_para,
                }
            )

        return tuple(contexts)

    metadata = {
        "has_mnu": has_mnu,
        "multi_bin": multi_bin,
        "n_bins": n_bins,
        "n_cosmo": n_cosmo,
        "n_survey": n_survey,
    }
    return evaluate_contexts, metadata


def _evaluate_pk_ell_from_context(
    k: jnp.ndarray,
    ells: tuple[int, ...],
    context: dict,
    *,
    ps1loop_model: PS1LoopModel,
    num: int,
    use_gl: bool,
    mu_gl: jnp.ndarray | None,
    w_proj: jnp.ndarray | None,
) -> jnp.ndarray:
    """Evaluate one bin of power-spectrum multipoles as ``(n_ell, n_k)``."""
    k = jnp.atleast_1d(jnp.asarray(k, dtype=float))
    alpha_perp = context["alpha_perp"]
    alpha_para = context["alpha_para"]
    pk_data = context["pklin_data"]
    params = context["pk_params"]

    if alpha_perp is not None and alpha_para is not None:
        if use_gl:
            pkmu = ps1loop_model.get_pkmu_ref(
                k,
                mu_gl,
                alpha_perp,
                alpha_para,
                pk_data,
                params,
            )
            pk_ells_2d = pkmu @ w_proj.T
            return pk_ells_2d.T
        return jnp.stack(
            [
                ps1loop_model.get_pk_ell_ref(
                    k,
                    ell,
                    alpha_perp,
                    alpha_para,
                    pk_data,
                    params,
                    num=num,
                )
                for ell in ells
            ],
            axis=0,
        )

    if use_gl:
        pkmu = ps1loop_model.get_pkmu(k, mu_gl, pk_data, params)
        pk_ells_2d = pkmu @ w_proj.T
        return pk_ells_2d.T
    return jnp.stack(
        [
            ps1loop_model.get_pk_ell(k, ell, pk_data, params, num=num)
            for ell in ells
        ],
        axis=0,
    )


def _evaluate_bk0_from_context(
    triangles: jnp.ndarray,
    context: dict,
    *,
    bispectrum_model: BispectrumTreeModel,
    num_mu: int,
    num_phi: int,
) -> jnp.ndarray:
    """Evaluate one bin of ``B0`` in the supplied triangle order."""
    triangles = _as_triangle_array(triangles)
    k1, k2, k3 = triangles.T
    return bispectrum_model.get_bk0(
        k1,
        k2,
        k3,
        context["pklin_data"],
        context["bk_params"],
        alpha_perp=context["alpha_perp"],
        alpha_para=context["alpha_para"],
        num_mu=num_mu,
        num_phi=num_phi,
    )


def _evaluate_bb_pk_ell_from_context(
    k: jnp.ndarray,
    context: dict,
    *,
    power_model: str,
    ps1loop_model: PS1LoopModel | None,
    num: int,
    use_gl: bool,
    mu_gl: jnp.ndarray | None,
    w_proj: jnp.ndarray | None,
    kaiser_n_gl: int,
) -> jnp.ndarray:
    """Return ``[Pt0, P2, P4]`` for the bispectrum Gaussian covariance."""
    if power_model == "kaiser":
        params = context["pk_params"]
        shot_noise = _power_shot_noise(params)
        return kaiser_power_multipoles(
            k,
            context["pklin_data"],
            b1=params["bias"]["b1"],
            f=params["f"],
            shot_noise=shot_noise,
            alpha_perp=context["alpha_perp"],
            alpha_para=context["alpha_para"],
            ells=(0, 2, 4),
            n_gl=kaiser_n_gl,
        )

    if power_model == "1loop":
        if ps1loop_model is None:
            raise ValueError(
                "power_model='1loop' requires ps1loop_model to be provided"
            )
        return _evaluate_pk_ell_from_context(
            k,
            (0, 2, 4),
            context,
            ps1loop_model=ps1loop_model,
            num=num,
            use_gl=use_gl,
            mu_gl=mu_gl,
            w_proj=w_proj,
        )

    raise ValueError(
        "power_model must be 'kaiser' or '1loop', "
        f"got {power_model!r}"
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
        Static flat key list for the packed survey layout. Legacy
        ``SurveyParams`` keys and role-aware ``FullShapeSurveyParams`` keys
        are both supported.
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
    evaluate_contexts, metadata = _make_theory_context_evaluator(
        pklin_emulator=pklin_emulator,
        cosmo_keys=cosmo_keys,
        cosmo_sizes=cosmo_sizes,
        survey_keys=survey_keys,
        ap=ap,
        Hz_fid=Hz_fid,
        DAz_fid=DAz_fid,
        z_bins=z_bins,
        mnu_fixed=mnu_fixed,
        background_mode=background_mode,
        background_nz=background_nz,
    )
    n_bins = metadata["n_bins"]
    has_mnu = metadata["has_mnu"]

    # ---- Gauss-Legendre projector (if requested) --------------------------
    use_gl = n_gl is not None
    mu_gl = None
    w_proj = None
    if use_gl:
        mu_gl, w_proj = make_multipole_projector(ells, n_gl)

    # ---- Build the closure ------------------------------------------------
    def pk_ell_fn(params, *, k):
        contexts = evaluate_contexts(params)
        all_pk = [
            _evaluate_pk_ell_from_context(
                k,
                ells,
                context,
                ps1loop_model=ps1loop_model,
                num=num,
                use_gl=use_gl,
                mu_gl=mu_gl,
                w_proj=w_proj,
            ).reshape(-1)
            for context in contexts
        ]
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


def make_bk0_fn(
    *,
    bispectrum_model: BispectrumTreeModel,
    pklin_emulator: CosmoEmulator,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    survey_keys: tuple,
    ap: bool = True,
    Hz_fid: tuple[float, ...] | None = None,
    DAz_fid: tuple[float, ...] | None = None,
    z_bins: tuple[float, ...] | None = None,
    mnu_fixed: float = 0.06,
    num_mu: int = 65,
    num_phi: int = 65,
    background_mode: str = "direct",
    background_nz: int = 256,
):
    r"""Build a differentiable bispectrum-monopole theory function.

    Returns a closure ``bk0_fn(params, *, triangles)``. In multi-bin mode the
    output is flattened in redshift-bin order, with each bin contributing the
    bispectrum monopole in the supplied triangle order.
    """
    evaluate_contexts, metadata = _make_theory_context_evaluator(
        pklin_emulator=pklin_emulator,
        cosmo_keys=cosmo_keys,
        cosmo_sizes=cosmo_sizes,
        survey_keys=survey_keys,
        ap=ap,
        Hz_fid=Hz_fid,
        DAz_fid=DAz_fid,
        z_bins=z_bins,
        mnu_fixed=mnu_fixed,
        background_mode=background_mode,
        background_nz=background_nz,
    )
    n_bins = metadata["n_bins"]
    has_mnu = metadata["has_mnu"]

    def bk0_fn(params, *, triangles):
        contexts = evaluate_contexts(params)
        blocks = []
        for b, context in enumerate(contexts):
            triangles_b = _as_triangle_array(
                _select_bin_input(triangles, b, n_bins, "triangles")
            )
            blocks.append(
                _evaluate_bk0_from_context(
                    triangles_b,
                    context,
                    bispectrum_model=bispectrum_model,
                    num_mu=num_mu,
                    num_phi=num_phi,
                ).reshape(-1)
            )
        return jnp.concatenate(blocks)

    bk0_fn.ap = ap
    bk0_fn.n_bins = n_bins
    bk0_fn.z_bins = z_bins
    bk0_fn.has_mnu = has_mnu
    bk0_fn.num_mu = num_mu
    bk0_fn.num_phi = num_phi
    bk0_fn.background_mode = background_mode
    bk0_fn.background_nz = background_nz

    return bk0_fn


def make_joint_pk_bk_fn(
    *,
    ells: tuple[int, ...] = (0, 2, 4),
    pklin_emulator: CosmoEmulator,
    ps1loop_model: PS1LoopModel,
    bispectrum_model: BispectrumTreeModel,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    survey_keys: tuple,
    ap: bool = True,
    Hz_fid: tuple[float, ...] | None = None,
    DAz_fid: tuple[float, ...] | None = None,
    z_bins: tuple[float, ...] | None = None,
    mnu_fixed: float = 0.06,
    num: int = 256,
    n_gl: int | None = None,
    num_mu: int = 65,
    num_phi: int = 65,
    background_mode: str = "direct",
    background_nz: int = 256,
):
    r"""Build a joint per-bin theory vector ``[P0, P2, P4, B0]``.

    The output is ordered by redshift bin. Each bin contributes one block

    ``[P0(all k), P2(all k), P4(all k), B0(all triangles)]``.
    """
    if tuple(ells) != (0, 2, 4):
        raise ValueError(
            "make_joint_pk_bk_fn requires ells=(0, 2, 4) so the output "
            "matches the joint covariance layout"
        )

    evaluate_contexts, metadata = _make_theory_context_evaluator(
        pklin_emulator=pklin_emulator,
        cosmo_keys=cosmo_keys,
        cosmo_sizes=cosmo_sizes,
        survey_keys=survey_keys,
        ap=ap,
        Hz_fid=Hz_fid,
        DAz_fid=DAz_fid,
        z_bins=z_bins,
        mnu_fixed=mnu_fixed,
        background_mode=background_mode,
        background_nz=background_nz,
    )
    n_bins = metadata["n_bins"]
    has_mnu = metadata["has_mnu"]
    use_gl = n_gl is not None
    mu_gl = None
    w_proj = None
    if use_gl:
        mu_gl, w_proj = make_multipole_projector(ells, n_gl)

    def joint_fn(params, *, k, triangles):
        contexts = evaluate_contexts(params)
        blocks = []
        for b, context in enumerate(contexts):
            triangles_b = _as_triangle_array(
                _select_bin_input(triangles, b, n_bins, "triangles")
            )
            pk_block = _evaluate_pk_ell_from_context(
                k,
                ells,
                context,
                ps1loop_model=ps1loop_model,
                num=num,
                use_gl=use_gl,
                mu_gl=mu_gl,
                w_proj=w_proj,
            ).reshape(-1)
            bk_block = _evaluate_bk0_from_context(
                triangles_b,
                context,
                bispectrum_model=bispectrum_model,
                num_mu=num_mu,
                num_phi=num_phi,
            ).reshape(-1)
            blocks.append(jnp.concatenate([pk_block, bk_block]))
        return jnp.concatenate(blocks)

    joint_fn.ells = ells
    joint_fn.ap = ap
    joint_fn.n_bins = n_bins
    joint_fn.z_bins = z_bins
    joint_fn.has_mnu = has_mnu
    joint_fn.num = num
    joint_fn.n_gl = n_gl
    joint_fn.num_mu = num_mu
    joint_fn.num_phi = num_phi
    joint_fn.background_mode = background_mode
    joint_fn.background_nz = background_nz
    joint_fn.layout = "[bin_0[P0,P2,P4,B0], ..., bin_N[P0,P2,P4,B0]]"

    return joint_fn


def make_gaussian_bk0_covariance_fn(
    *,
    pklin_emulator: CosmoEmulator,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    survey_keys: tuple,
    ps1loop_model: PS1LoopModel | None = None,
    ap: bool = True,
    Hz_fid: tuple[float, ...] | None = None,
    DAz_fid: tuple[float, ...] | None = None,
    z_bins: tuple[float, ...] | None = None,
    mnu_fixed: float = 0.06,
    bb_power_model: str = "kaiser",
    num: int = 256,
    n_gl: int | None = None,
    kaiser_n_gl: int = 16,
    background_mode: str = "direct",
    background_nz: int = 256,
):
    r"""Build a Gaussian ``C_BB`` closure for the bispectrum monopole.

    The bispectrum-covariance power inputs default to Kaiser multipoles
    ``[Pt0, P2, P4]`` built from the same packed cosmology and survey
    parameters. Set ``bb_power_model='1loop'`` to reuse the 1-loop
    power-spectrum multipoles instead.
    """
    evaluate_contexts, metadata = _make_theory_context_evaluator(
        pklin_emulator=pklin_emulator,
        cosmo_keys=cosmo_keys,
        cosmo_sizes=cosmo_sizes,
        survey_keys=survey_keys,
        ap=ap,
        Hz_fid=Hz_fid,
        DAz_fid=DAz_fid,
        z_bins=z_bins,
        mnu_fixed=mnu_fixed,
        background_mode=background_mode,
        background_nz=background_nz,
    )
    n_bins = metadata["n_bins"]
    has_mnu = metadata["has_mnu"]
    use_gl = n_gl is not None
    mu_gl = None
    w_proj = None
    if use_gl:
        mu_gl, w_proj = make_multipole_projector((0, 2, 4), n_gl)

    def bb_cov_fn(params, *, V_survey, k, triangles, triangle_dk):
        contexts = evaluate_contexts(params)
        pk_blocks = [
            _evaluate_bb_pk_ell_from_context(
                k,
                context,
                power_model=bb_power_model,
                ps1loop_model=ps1loop_model,
                num=num,
                use_gl=use_gl,
                mu_gl=mu_gl,
                w_proj=w_proj,
                kaiser_n_gl=kaiser_n_gl,
            )
            for context in contexts
        ]
        if n_bins == 1:
            return gaussian_bispectrum_covariance(
                V_survey,
                k,
                triangles,
                triangle_dk,
                pk_blocks[0],
            )

        return gaussian_bispectrum_covariance_multibin(
            V_survey,
            k,
            triangles,
            triangle_dk,
            jnp.stack(pk_blocks, axis=0),
        )

    bb_cov_fn.ap = ap
    bb_cov_fn.n_bins = n_bins
    bb_cov_fn.z_bins = z_bins
    bb_cov_fn.has_mnu = has_mnu
    bb_cov_fn.bb_power_model = bb_power_model
    bb_cov_fn.num = num
    bb_cov_fn.n_gl = n_gl
    bb_cov_fn.kaiser_n_gl = kaiser_n_gl
    bb_cov_fn.background_mode = background_mode
    bb_cov_fn.background_nz = background_nz

    return bb_cov_fn


def make_gaussian_joint_covariance_fn(
    *,
    pklin_emulator: CosmoEmulator,
    ps1loop_model: PS1LoopModel,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    survey_keys: tuple,
    ap: bool = True,
    Hz_fid: tuple[float, ...] | None = None,
    DAz_fid: tuple[float, ...] | None = None,
    z_bins: tuple[float, ...] | None = None,
    mnu_fixed: float = 0.06,
    bb_power_model: str = "kaiser",
    num: int = 256,
    n_gl: int | None = None,
    kaiser_n_gl: int = 16,
    background_mode: str = "direct",
    background_nz: int = 256,
):
    r"""Build a joint Gaussian covariance closure for ``[P0, P2, P4, B0]``.

    The ``C_PP`` block always uses the 1-loop power-spectrum multipoles from
    ``ps1loop_model``. The ``C_BB`` block defaults to Kaiser multipoles via
    ``bb_power_model='kaiser'`` and can be switched to ``'1loop'`` if desired.
    """
    pk_fn = make_pk_ell_fn(
        ells=(0, 2, 4),
        pklin_emulator=pklin_emulator,
        ps1loop_model=ps1loop_model,
        cosmo_keys=cosmo_keys,
        cosmo_sizes=cosmo_sizes,
        survey_keys=survey_keys,
        ap=ap,
        Hz_fid=Hz_fid,
        DAz_fid=DAz_fid,
        z_bins=z_bins,
        mnu_fixed=mnu_fixed,
        num=num,
        n_gl=n_gl,
        background_mode=background_mode,
        background_nz=background_nz,
    )
    evaluate_contexts, metadata = _make_theory_context_evaluator(
        pklin_emulator=pklin_emulator,
        cosmo_keys=cosmo_keys,
        cosmo_sizes=cosmo_sizes,
        survey_keys=survey_keys,
        ap=ap,
        Hz_fid=Hz_fid,
        DAz_fid=DAz_fid,
        z_bins=z_bins,
        mnu_fixed=mnu_fixed,
        background_mode=background_mode,
        background_nz=background_nz,
    )
    n_bins = metadata["n_bins"]
    has_mnu = metadata["has_mnu"]
    use_gl = n_gl is not None
    mu_gl = None
    w_proj = None
    if use_gl:
        mu_gl, w_proj = make_multipole_projector((0, 2, 4), n_gl)

    def joint_cov_fn(params, *, V_survey, k, dk, triangles=None, triangle_dk=None):
        pp_flat = pk_fn(params, k=k)
        contexts = evaluate_contexts(params)
        bb_pk_blocks = [
            _evaluate_bb_pk_ell_from_context(
                k,
                context,
                power_model=bb_power_model,
                ps1loop_model=ps1loop_model,
                num=num,
                use_gl=use_gl,
                mu_gl=mu_gl,
                w_proj=w_proj,
                kaiser_n_gl=kaiser_n_gl,
            )
            for context in contexts
        ]

        if n_bins == 1:
            return gaussian_joint_covariance(
                V_survey,
                k,
                dk,
                pp_flat.reshape(3, -1),
                triangles=triangles,
                triangle_dk=triangle_dk,
                bb_pk_ell=bb_pk_blocks[0],
            )

        return gaussian_joint_covariance_multibin(
            V_survey,
            k,
            dk,
            pp_flat.reshape(n_bins, 3, -1),
            triangles=triangles,
            triangle_dk=triangle_dk,
            bb_pk_all=jnp.stack(bb_pk_blocks, axis=0),
        )

    joint_cov_fn.ap = ap
    joint_cov_fn.n_bins = n_bins
    joint_cov_fn.z_bins = z_bins
    joint_cov_fn.has_mnu = has_mnu
    joint_cov_fn.bb_power_model = bb_power_model
    joint_cov_fn.num = num
    joint_cov_fn.n_gl = n_gl
    joint_cov_fn.kaiser_n_gl = kaiser_n_gl
    joint_cov_fn.background_mode = background_mode
    joint_cov_fn.background_nz = background_nz

    return joint_cov_fn
