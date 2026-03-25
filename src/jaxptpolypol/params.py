"""
Parameter containers for cosmology and survey specifications.

Both classes are registered as JAX pytrees so they can be used inside
``jax.jit``, ``jax.jacfwd``, ``jax.grad``, etc.

The containers are agnostic to the inference method — they work equally well
for Fisher forecasts, MCMC sampling, or any other JAX-based pipeline.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "CosmoParams",
    "SurveyParams",
    "pack_params",
    "unpack_params",
    "pack_multibin_params",
    "unpack_multibin_params",
]


# ---------------------------------------------------------------------------
# CosmoParams
# ---------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
class CosmoParams:
    """Container for cosmological parameters, registered as a JAX pytree.

    Parameters
    ----------
    param_dict : dict[str, float | array-like]
        Mapping from parameter name to value.  Each value is stored as a
        1-d ``jnp.ndarray`` (scalars are promoted via ``jnp.atleast_1d``).

    Notes
    -----
    Parameter *names* and *sizes* are static (compilation constants);
    parameter *values* are traced (differentiable).
    """

    def __init__(self, param_dict: dict):
        self.param_dict = param_dict
        self.param_keys: tuple[str, ...] = tuple(param_dict.keys())
        self.param_values: list[jnp.ndarray] = [
            jnp.atleast_1d(jnp.asarray(v)) for v in param_dict.values()
        ]
        self.param_sizes: tuple[int, ...] = tuple(v.size for v in self.param_values)

    # --- pretty printing ---------------------------------------------------

    def __str__(self) -> str:
        lines = ["CosmoParams:"]
        for k, v in self.to_dict().items():
            if v.size == 1:
                lines.append(f"  {k}: {v.item():.5g}")
            else:
                formatted = ", ".join(f"{x:.5g}" for x in v)
                lines.append(f"  {k}: [{formatted}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CosmoParams({self.to_dict()})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, CosmoParams):
            return False
        if self.param_keys != other.param_keys:
            return False
        return all(
            jnp.allclose(v1, v2)
            for v1, v2 in zip(self.param_values, other.param_values)
        )

    # --- core accessors -----------------------------------------------------

    def to_dict(self) -> dict[str, jnp.ndarray]:
        """Return ``{name: jnp.ndarray}`` dictionary (e.g. for emulator input)."""
        return {k: jnp.atleast_1d(v) for k, v in zip(self.param_keys, self.param_values)}

    def to_array(self) -> jnp.ndarray:
        """Flatten all parameter values into a single 1-d array."""
        return jnp.concatenate(self.param_values)

    def get(self, name: str) -> jnp.ndarray:
        """Look up a parameter by name."""
        idx = self.param_keys.index(name)
        return self.param_values[idx]

    # --- convenience properties (common parameter aliases) ------------------

    @property
    def z(self) -> jnp.ndarray:
        return self.get("z")

    @property
    def h(self) -> jnp.ndarray:
        return self.get("h")

    @property
    def omega_b(self) -> jnp.ndarray:
        for key in ("omega_b", "ombh2"):
            if key in self.param_keys:
                return self.get(key)
        raise KeyError("Neither 'omega_b' nor 'ombh2' found.")

    @property
    def omega_cdm(self) -> jnp.ndarray:
        for key in ("omega_cdm", "omch2"):
            if key in self.param_keys:
                return self.get(key)
        raise KeyError("Neither 'omega_cdm' nor 'omch2' found.")

    @property
    def mnu(self) -> jnp.ndarray:
        return self.get("mnu")

    @property
    def n_s(self) -> jnp.ndarray:
        for key in ("n_s", "ns"):
            if key in self.param_keys:
                return self.get(key)
        raise KeyError("Neither 'n_s' nor 'ns' found.")

    @property
    def logA(self) -> jnp.ndarray:
        for key in ("logA", "ln10^{10}A_s"):
            if key in self.param_keys:
                return self.get(key)
        raise KeyError("Neither 'logA' nor 'ln10^{10}A_s' found.")

    # --- (de)serialisation --------------------------------------------------

    @classmethod
    def from_array(
        cls, values: jnp.ndarray, keys: tuple[str, ...], sizes: tuple[int, ...]
    ) -> CosmoParams:
        """Reconstruct from a flat array given static metadata."""
        params: dict[str, jnp.ndarray] = {}
        idx = 0
        for k, s in zip(keys, sizes):
            params[k] = values[idx : idx + s]
            idx += s
        return cls(params)

    # Alias kept for backward compatibility with existing notebooks.
    unpack_array = from_array

    # --- JAX pytree ---------------------------------------------------------

    def tree_flatten(self):
        children = (self.param_values,)
        aux_data = (self.param_keys, self.param_sizes)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        keys, sizes = aux_data
        (values,) = children
        return cls({k: v.astype(float) for k, v in zip(keys, values)})


# ---------------------------------------------------------------------------
# SurveyParams
# ---------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
class SurveyParams:
    """Container for survey / nuisance parameters, registered as a JAX pytree.

    Parameters
    ----------
    param_dict : dict[str, float | dict[str, float]]
        Nested dictionary whose top-level keys are *groups*
        (``'bias'``, ``'ctr'``, ``'stoch'``, …) and whose values are either
        a scalar (stored with ``key=None``) or a sub-dict of named scalars.

    Notes
    -----
    Internally the nested structure is flattened to a list of
    ``((group, key), value)`` pairs.  The flat key list is static;
    the values array is traced.
    """

    def __init__(self, param_dict: dict):
        self.param_dict = param_dict
        flat_list = [
            ((group, k), v)
            for group, val in self.param_dict.items()
            for k, v in (val.items() if isinstance(val, dict) else [(None, val)])
        ]
        self.param_keys: tuple[tuple[str, str | None], ...] = tuple(
            kk for kk, _ in flat_list
        )
        self.param_values: jnp.ndarray = jnp.array([v for _, v in flat_list])

    # --- pretty printing ---------------------------------------------------

    def __str__(self) -> str:
        lines = ["SurveyParams:"]
        for group, val in self.to_dict().items():
            if isinstance(val, dict):
                lines.append(f"  {group}:")
                for k, v in val.items():
                    lines.append(f"    {k}: {v:.5g}")
            else:
                lines.append(f"  {group}: {val:.5g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SurveyParams({self.to_dict()})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, SurveyParams):
            return False
        return (
            self.param_keys == other.param_keys
            and jnp.allclose(self.param_values, other.param_values)
        )

    # --- core accessors -----------------------------------------------------

    def to_dict(self) -> dict:
        """Reconstruct the nested dictionary."""
        out: dict = {}
        for (group, key), val in zip(self.param_keys, self.param_values):
            if key is None:
                out[group] = val.astype(float)
            else:
                out.setdefault(group, {})[key] = val.astype(float)
        return out

    def to_array(self) -> jnp.ndarray:
        return self.param_values

    def get(self, group: str, key: str | None = None):
        """Look up a parameter by (group, key)."""
        for i, (g, k) in enumerate(self.param_keys):
            if g == group and k == key:
                return self.param_values[i]
        raise KeyError(f"Parameter '{group}:{key}' not found.")

    # --- (de)serialisation --------------------------------------------------

    @classmethod
    def from_array(cls, flat_values: jnp.ndarray, flat_keys) -> SurveyParams:
        """Reconstruct from a flat array and static key list."""
        grouped: dict = {}
        for (group, key), v in zip(flat_keys, flat_values):
            if key is None:
                grouped[group] = v
            else:
                grouped.setdefault(group, {})[key] = v
        return cls(grouped)

    unpack_array = from_array

    # --- JAX pytree ---------------------------------------------------------

    def tree_flatten(self):
        return (self.param_values,), (self.param_keys,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (flat_keys,) = aux_data
        (flat_values,) = children
        grouped: dict = {}
        for (group, key), val in zip(flat_keys, flat_values):
            if key is None:
                grouped[group] = val
            else:
                grouped.setdefault(group, {})[key] = val
        return cls(grouped)


# ---------------------------------------------------------------------------
# Pack / unpack helpers
# ---------------------------------------------------------------------------
def pack_params(
    cosmo: CosmoParams, survey: SurveyParams
) -> jnp.ndarray:
    """Pack cosmo + survey into a single flat parameter vector."""
    return jnp.concatenate([cosmo.to_array(), survey.to_array()])


def unpack_params(
    params: jnp.ndarray,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    survey_keys,
) -> tuple[CosmoParams, SurveyParams]:
    """Unpack a flat vector into ``(CosmoParams, SurveyParams)``."""
    n_cosmo = sum(cosmo_sizes)
    return (
        CosmoParams.from_array(params[:n_cosmo], cosmo_keys, cosmo_sizes),
        SurveyParams.from_array(params[n_cosmo:], survey_keys),
    )


def pack_multibin_params(
    cosmo: CosmoParams, surveys: list[SurveyParams]
) -> jnp.ndarray:
    """Pack shared cosmo + per-bin survey params into a single flat vector.

    Layout: ``[cosmo | survey_bin0 | survey_bin1 | ... | survey_binN]``
    """
    parts = [cosmo.to_array()]
    for s in surveys:
        parts.append(s.to_array())
    return jnp.concatenate(parts)


def unpack_multibin_params(
    params: jnp.ndarray,
    cosmo_keys: tuple[str, ...],
    cosmo_sizes: tuple[int, ...],
    survey_keys,
    n_bins: int,
) -> tuple[CosmoParams, list[SurveyParams]]:
    """Unpack a flat vector into ``(CosmoParams, [SurveyParams, ...])``."""
    n_cosmo = sum(cosmo_sizes)
    n_survey = len(survey_keys)
    cosmo_obj = CosmoParams.from_array(params[:n_cosmo], cosmo_keys, cosmo_sizes)
    survey_objs = []
    for i in range(n_bins):
        offset = n_cosmo + i * n_survey
        survey_objs.append(
            SurveyParams.from_array(params[offset : offset + n_survey], survey_keys)
        )
    return cosmo_obj, survey_objs
