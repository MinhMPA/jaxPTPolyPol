"""
Thin wrappers around emulators and theory codes.

These wrappers are JAX pytrees so they can be passed through ``jax.jit``
boundaries as static arguments.  They expose a uniform interface used by
both Fisher and MCMC pipelines.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["CosmoEmulator", "PS1LoopModel"]


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
            Parameter dictionary matching :pyattr:`parameters`.

        Returns
        -------
        prediction : jnp.ndarray
            Emulated quantity on the :pyattr:`modes` grid.
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
