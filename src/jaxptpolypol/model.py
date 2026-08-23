"""
Thin wrappers around emulators and theory codes.

These wrappers are JAX pytrees so they can be passed through ``jax.jit``
boundaries as static arguments.  They expose a uniform interface used by
both Fisher and MCMC pipelines.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["CosmoEmulator", "PS1LoopModel", "BispectrumTreeModel"]


def _contains_tracer(*values) -> bool:
    """Return ``True`` when any leaf is currently being traced by JAX."""
    return any(
        isinstance(leaf, jax.core.Tracer)
        for value in values
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _triangle_closure_tolerance(k1_np, k2_np, k3_np) -> float:
    """Match the triangle-builder closure tolerance on concrete arrays."""
    scale = float(max(np.max(k1_np), np.max(k2_np), np.max(k3_np), 1.0))
    return 10.0 * np.finfo(np.float64).eps * scale


def _validate_triangle_eager(k1, k2, k3) -> None:
    """Raise the upstream triangle-geometry errors on concrete arrays."""
    k1_np, k2_np, k3_np = np.broadcast_arrays(
        np.asarray(k1, dtype=float),
        np.asarray(k2, dtype=float),
        np.asarray(k3, dtype=float),
    )

    if np.any(k1_np <= 0) or np.any(k2_np <= 0) or np.any(k3_np <= 0):
        raise ValueError("All triangle side lengths must be strictly positive.")

    tol = _triangle_closure_tolerance(k1_np, k2_np, k3_np)
    valid = (
        (k1_np + k2_np >= k3_np - tol)
        & (k1_np + k3_np >= k2_np - tol)
        & (k2_np + k3_np >= k1_np - tol)
    )
    if not np.all(valid):
        raise ValueError("Input wavenumbers do not satisfy the triangle inequality.")


def _validate_triangle_jax_safe(_, k1, k2, k3) -> None:
    """Preserve eager validation while allowing traced triangle arrays under ``jit``."""
    if _contains_tracer(k1, k2, k3):
        jax.debug.callback(_validate_triangle_eager, k1, k2, k3)
        return
    _validate_triangle_eager(k1, k2, k3)


def _needs_ndens_jax_safe(_, stoch) -> bool:
    """Require ``ndens`` whenever stochastic amplitudes may be traced at runtime."""

    def _value_needs_ndens(value) -> bool:
        if value is None:
            return False
        if _contains_tracer(value):
            return True
        return bool(np.any(np.asarray(value) != 0.0))

    return any(_value_needs_ndens(value) for value in stoch.values())


def _get_ndens_jax_safe(_, params, stoch):
    """Mirror the upstream logic without converting traced stochastic values to NumPy."""
    has_explicit_stoch = (
        "stoch" in params
        or "P_shot" in params
        or "Pshot" in params
        or "B_shot" in params
        or "Bshot" in params
        or "A_shot" in params
        or "Ashot" in params
    )
    if not has_explicit_stoch:
        return None
    if "ndens" in params:
        return jnp.asarray(params["ndens"], dtype=float)
    if _needs_ndens_jax_safe(None, stoch):
        raise KeyError("params['ndens'] is required when bispectrum stochasticity is enabled.")
    return None


def _get_bk_shot_jax_safe(_, params):
    """Return bispectrum shot noise without Python branching on traced values."""
    stoch = params.get("stoch", {})

    if "B_shot" in stoch:
        bshot = stoch["B_shot"]
    elif "Bshot" in stoch:
        bshot = stoch["Bshot"]
    elif "Bshot" in params:
        bshot = params["Bshot"]
    else:
        bshot = 0.0

    bshot_arr = jnp.asarray(bshot, dtype=float)
    if _contains_tracer(bshot_arr):
        if "ndens" not in params:
            raise KeyError(
                "params['ndens'] is required when bispectrum shot noise is enabled."
            )
        ndens = jnp.asarray(params["ndens"], dtype=float)
        return jnp.where(bshot_arr == 0.0, 0.0, bshot_arr / ndens**2)

    if float(np.asarray(bshot_arr)) == 0.0:
        return 0.0

    if "ndens" not in params:
        raise KeyError("params['ndens'] is required when bispectrum shot noise is enabled.")

    return bshot_arr / jnp.asarray(params["ndens"], dtype=float) ** 2


# ---------------------------------------------------------------------------
# Emulator wrapper
# ---------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
class CosmoEmulator:
    """Wrapper around a ``CosmoPowerJAX`` emulator network.

    Parameters
    ----------
    probe : str
        Probe type (``'custom_log'``, ``'custom_pca'``, etc.).
    emulator_path : str
        Path to the ``.npz`` network file.
    emulator_params : list[str], optional
        Override the parameter names stored in the network.
    """

    def __init__(
        self,
        probe: str = "custom_log",
        emulator_path: str | None = None,
        emulator_params: list[str] | None = None,
    ):
        # Lazy import so the module can be loaded even if cosmopower_jax
        # is not installed (useful for testing other submodules).
        from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

        self.probe = probe
        self.path = emulator_path
        self.emulator = CosmoPowerJAX(probe=self.probe, filepath=self.path)

        if emulator_params is not None:
            self.emulator.parameters = emulator_params
        self._parameters = self.emulator.parameters
        self._modes = self.emulator.modes

    @property
    def modes(self):
        """Fourier modes (k) in the emulator's native units [Mpc⁻¹]."""
        return self._modes

    @property
    def parameters(self):
        """List of emulator input parameter names."""
        return self._parameters

    def predict(self, cosmo_dict: dict) -> jnp.ndarray:
        """Evaluate the emulator.

        Parameters
        ----------
        cosmo_dict : dict[str, jnp.ndarray]
            Parameter dictionary matching :py:attr:`parameters`.

        Returns
        -------
        prediction : jnp.ndarray
            Emulated quantity on the :py:attr:`modes` grid.
        """
        return self.emulator.predict(cosmo_dict)

    # --- JAX pytree ---------------------------------------------------------

    def tree_flatten(self):
        children = (self._parameters,)
        aux_data = (self.probe, self.path, self._modes)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        probe, path, modes = aux_data
        (parameters,) = children
        obj = cls(probe=probe, emulator_path=path, emulator_params=parameters)
        obj._modes = modes
        return obj


# ---------------------------------------------------------------------------
# 1-loop power spectrum model wrapper
# ---------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
class PS1LoopModel:
    """Wrapper around ``ps_1loop_jax.PowerSpectrum1Loop``.

    Parameters
    ----------
    do_irres : bool
        Whether to perform IR resummation (default ``True``).
    """

    def __init__(self, do_irres: bool = True):
        from ps_1loop_jax import PowerSpectrum1Loop

        self.do_irres = do_irres
        self.model = PowerSpectrum1Loop(do_irres=do_irres)
        self.pk_terms = self.model.name_pk_terms

    def get_pk_ell(self, k, ell, pk_data, params, num=256):
        """Power spectrum multipole without AP distortion."""
        return self.model.get_pk_ell(k, ell, pk_data, params, num=num)

    def get_pk_ell_ref(self, k, ell, alpha_perp, alpha_para, pk_data, params, num=256):
        """Power spectrum multipole in the reference (fiducial) frame with AP."""
        return self.model.get_pk_ell_ref(
            k, ell, alpha_perp, alpha_para, pk_data, params, num=num
        )

    def get_pkmu(self, k, mu, pk_data, params):
        """Full anisotropic power spectrum P(k, mu)."""
        return self.model.get_pkmu(k, mu, pk_data, params)

    def get_pkmu_ref(self, k, mu, alpha_perp, alpha_para, pk_data, params):
        """Anisotropic P(k, mu) in the reference frame with AP distortion."""
        return self.model.get_pkmu_ref(
            k, mu, alpha_perp, alpha_para, pk_data, params
        )

    # --- JAX pytree ---------------------------------------------------------

    def tree_flatten(self):
        return (), (self.do_irres,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (do_irres,) = aux_data
        return cls(do_irres=do_irres)


# ---------------------------------------------------------------------------
# Tree-level bispectrum model wrapper
# ---------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
class BispectrumTreeModel:
    """Wrapper around ``ps_1loop_jax.bs_tree.BispectrumTree``.

    Parameters
    ----------
    do_irres : bool
        Whether to enable the IR-resummed bispectrum path (default ``True``).
        Upstream implements this: ``BispectrumTree`` builds the no-wiggle
        spectrum via ``ir_resum.get_pk_nw_data`` and requires ``params['h']``,
        which :func:`~jaxptpolypol.theory.make_joint_pk_bk_fn` supplies. The
        flag moves only the bispectrum entries of a joint P+B vector (max
        relative change 4.8e-3 on the 7-bin test configuration); the power
        spectrum multipoles are bit-identical.
    do_AP : bool
        Whether the wrapped model should expect AP-remapped calls by default.
        ``do_AP`` is consulted only when the caller passes no alphas: the
        jaxptpolypol closures pass explicit ``(alpha_perp, alpha_para)`` when
        built with ``ap=True`` (upstream then remaps regardless of this flag)
        and ``None`` when built with ``ap=False`` -- in which case a
        ``do_AP=True`` model RAISES (upstream requires both alphas). Keep
        ``do_AP`` matched to the closure's ``ap``; every production site uses
        ``do_AP=True`` with ``ap=True``.
    rbao, ks, k_nl_rsd, kmin_fft, kmax_fft, nfft
        Static configuration forwarded to ``BispectrumTree``.
    """

    def __init__(
        self,
        do_irres: bool = True,
        do_AP: bool = False,
        rbao: float = 110.0,
        ks: float = 0.2,
        k_nl_rsd: float = 0.3,
        kmin_fft: float = 1e-5,
        kmax_fft: float = 1e3,
        nfft: int = 256,
    ):
        from ps_1loop_jax.bs_tree import BispectrumTree

        self.do_irres = do_irres
        self.do_AP = do_AP
        self.rbao = rbao
        self.ks = ks
        self.k_nl_rsd = k_nl_rsd
        self.kmin_fft = kmin_fft
        self.kmax_fft = kmax_fft
        self.nfft = nfft
        init_sig = inspect.signature(BispectrumTree.__init__)
        init_kwargs = {
            "do_irres": do_irres,
            "do_AP": do_AP,
            "rbao": rbao,
            "ks": ks,
            "kmin_fft": kmin_fft,
            "kmax_fft": kmax_fft,
            "nfft": nfft,
        }
        if "k_nl_rsd" in init_sig.parameters:
            init_kwargs["k_nl_rsd"] = k_nl_rsd
        self.model = BispectrumTree(**init_kwargs)
        self.model._validate_triangle = _validate_triangle_jax_safe.__get__(
            self.model, type(self.model)
        )
        self.model._needs_ndens = _needs_ndens_jax_safe.__get__(
            self.model, type(self.model)
        )
        self.model._get_ndens = _get_ndens_jax_safe.__get__(
            self.model, type(self.model)
        )
        if hasattr(self.model, "_get_bk_shot"):
            self.model._get_bk_shot = _get_bk_shot_jax_safe.__get__(
                self.model, type(self.model)
            )

    def get_bk0(
        self,
        k1,
        k2,
        k3,
        pk_data,
        params,
        *,
        alpha_perp=None,
        alpha_para=None,
        num_mu: int = 65,
        num_phi: int = 65,
    ):
        """Bispectrum monopole, optionally in the reference frame with AP."""
        return self.model.get_bk0(
            k1,
            k2,
            k3,
            pk_data,
            params,
            alpha_perp=alpha_perp,
            alpha_para=alpha_para,
            num_mu=num_mu,
            num_phi=num_phi,
        )

    def get_bk0_ref(
        self,
        k1,
        k2,
        k3,
        alpha_perp,
        alpha_para,
        pk_data,
        params,
        *,
        num_mu: int = 65,
        num_phi: int = 65,
    ):
        """Bispectrum monopole in the reference frame with AP."""
        if hasattr(self.model, "get_bk0_ref"):
            return self.model.get_bk0_ref(
                k1,
                k2,
                k3,
                alpha_perp,
                alpha_para,
                pk_data,
                params,
                num_mu=num_mu,
                num_phi=num_phi,
            )
        return self.model.get_bk0_tree_ref(
            k1,
            k2,
            k3,
            alpha_perp,
            alpha_para,
            pk_data,
            params,
            num_mu=num_mu,
            num_phi=num_phi,
        )

    # --- JAX pytree ---------------------------------------------------------

    def tree_flatten(self):
        return (), (
            self.do_irres,
            self.do_AP,
            self.rbao,
            self.ks,
            self.k_nl_rsd,
            self.kmin_fft,
            self.kmax_fft,
            self.nfft,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            do_irres,
            do_AP,
            rbao,
            ks,
            k_nl_rsd,
            kmin_fft,
            kmax_fft,
            nfft,
        ) = aux_data
        return cls(
            do_irres=do_irres,
            do_AP=do_AP,
            rbao=rbao,
            ks=ks,
            k_nl_rsd=k_nl_rsd,
            kmin_fft=kmin_fft,
            kmax_fft=kmax_fft,
            nfft=nfft,
        )
