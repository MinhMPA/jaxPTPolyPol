"""
Marginalization and projection helpers for Fisher corner plots.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .chain_analysis import DEFAULT_LATEX_LABELS
from .derived import sigma8_from_linear_pk
from .inference import marginalized_fisher_block, project_fisher_to_derived

__all__ = [
    "ParameterSpec",
    "ProjectedFisherResult",
    "native_spec",
    "H0_spec",
    "omega_m_spec",
    "sigma8_spec",
    "make_lcdm_corner_specs",
    "project_case_to_specs",
]


TransformFn = Callable[[Mapping[str, jnp.ndarray]], jnp.ndarray]


@dataclass(frozen=True)
class ParameterSpec:
    """Declarative specification of a final plotted parameter.

    Parameters
    ----------
    name
        Output parameter name.
    native_names
        Native parameter names required to compute this output.
    transform
        Callable mapping ``{native_name: scalar_value}`` to the scalar output.
        When omitted, the output is the identity map of the single native name.
    label
        Plot label. Defaults to :mod:`chain_analysis`'s LaTeX label map.
    """

    name: str
    native_names: tuple[str, ...]
    transform: TransformFn | None = None
    label: str | None = None


@dataclass(frozen=True)
class ProjectedFisherResult:
    """Result of marginalizing and projecting a Fisher case for plotting."""

    fisher_plot: np.ndarray
    fid_plot: np.ndarray
    plot_names: tuple[str, ...]
    plot_labels: tuple[str, ...]
    native_keep_names: tuple[str, ...]
    native_keep_idx: tuple[int, ...]
    jacobian: np.ndarray
    covariance_plot: np.ndarray


def _default_label(name: str) -> str:
    return DEFAULT_LATEX_LABELS.get(name, name)


def _ordered_unique(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _normalize_specs(specs: Sequence[ParameterSpec | str]) -> tuple[ParameterSpec, ...]:
    normalized: list[ParameterSpec] = []
    for spec in specs:
        if isinstance(spec, str):
            normalized.append(native_spec(spec))
        else:
            normalized.append(spec)
    if not normalized:
        raise ValueError("specs must contain at least one plotted parameter")

    plot_names = [spec.name for spec in normalized]
    if len(set(plot_names)) != len(plot_names):
        raise ValueError(f"plot parameter names must be unique, got {plot_names}")

    for spec in normalized:
        if not spec.native_names:
            raise ValueError(f"spec {spec.name!r} must list at least one native name")
        if spec.transform is None and len(spec.native_names) != 1:
            raise ValueError(
                f"identity spec {spec.name!r} must depend on exactly one native name"
            )
    return tuple(normalized)


def _make_projection_fn(
    native_keep_names: Sequence[str],
    specs: Sequence[ParameterSpec],
):
    native_keep_names = tuple(native_keep_names)
    native_positions = {name: i for i, name in enumerate(native_keep_names)}

    def derived_fn(theta_native):
        theta_native = jnp.ravel(jnp.asarray(theta_native, dtype=jnp.float64))
        native_values = {
            name: theta_native[native_positions[name]]
            for name in native_keep_names
        }
        outputs = []
        for spec in specs:
            if spec.transform is None:
                value = native_values[spec.native_names[0]]
            else:
                local_values = {name: native_values[name] for name in spec.native_names}
                value = spec.transform(local_values)
            value = jnp.ravel(jnp.asarray(value, dtype=jnp.float64))
            if value.size != 1:
                raise ValueError(
                    f"spec {spec.name!r} must return a scalar, got shape {value.shape}"
                )
            outputs.append(value[0])
        return jnp.stack(outputs)

    return derived_fn


def native_spec(name: str, *, label: str | None = None) -> ParameterSpec:
    """Return an identity spec for a native parameter."""
    return ParameterSpec(
        name=name,
        native_names=(name,),
        label=label or _default_label(name),
    )


def H0_spec(
    *,
    h_name: str = "h",
    fixed_cosmo_values: Mapping[str, float] | None = None,
    name: str = "H0",
    label: str | None = None,
) -> ParameterSpec:
    """Return the ``H0 = 100 h`` transformed parameter spec."""
    fixed_cosmo_values = dict(fixed_cosmo_values or {})
    native_names = () if h_name in fixed_cosmo_values else (h_name,)

    def transform(values: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        if h_name in values:
            h = jnp.asarray(values[h_name], dtype=jnp.float64)
        else:
            h = jnp.asarray(fixed_cosmo_values[h_name], dtype=jnp.float64)
        return 100.0 * h

    return ParameterSpec(
        name=name,
        native_names=native_names,
        transform=transform,
        label=label or _default_label(name),
    )


def omega_m_spec(
    *,
    omega_b_name: str = "ombh2",
    omega_cdm_name: str = "omch2",
    h_name: str = "h",
    mnu_name: str | None = None,
    mnu_fixed: float = 0.06,
    fixed_cosmo_values: Mapping[str, float] | None = None,
    name: str = "Omega_m",
    label: str | None = None,
) -> ParameterSpec:
    """Return the ``Omega_m`` transformed parameter spec."""
    fixed_cosmo_values = dict(fixed_cosmo_values or {})
    native_names = [
        key
        for key in (omega_b_name, omega_cdm_name, h_name)
        if key not in fixed_cosmo_values
    ]
    if mnu_name is not None and mnu_name not in fixed_cosmo_values:
        native_names.append(mnu_name)

    def transform(values: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        if omega_b_name in values:
            omega_b = jnp.asarray(values[omega_b_name], dtype=jnp.float64)
        else:
            omega_b = jnp.asarray(fixed_cosmo_values[omega_b_name], dtype=jnp.float64)
        if omega_cdm_name in values:
            omega_cdm = jnp.asarray(values[omega_cdm_name], dtype=jnp.float64)
        else:
            omega_cdm = jnp.asarray(
                fixed_cosmo_values[omega_cdm_name],
                dtype=jnp.float64,
            )
        if h_name in values:
            h = jnp.asarray(values[h_name], dtype=jnp.float64)
        else:
            h = jnp.asarray(fixed_cosmo_values[h_name], dtype=jnp.float64)
        if mnu_name is None:
            mnu = jnp.asarray(mnu_fixed, dtype=jnp.float64)
        elif mnu_name in values:
            mnu = jnp.asarray(values[mnu_name], dtype=jnp.float64)
        else:
            mnu = jnp.asarray(fixed_cosmo_values[mnu_name], dtype=jnp.float64)
        omega_nu = mnu / 93.14
        return (omega_b + omega_cdm + omega_nu) / (h**2)

    return ParameterSpec(
        name=name,
        native_names=tuple(native_names),
        transform=transform,
        label=label or _default_label(name),
    )


def sigma8_spec(
    cosmo_keys: Sequence[str],
    *,
    pklin_emulator,
    h_name: str = "h",
    fixed_cosmo_values: Mapping[str, float] | None = None,
    sigma8_redshift: float = 0.0,
    name: str = "sigma8",
    label: str | None = None,
) -> ParameterSpec:
    """Return the emulator-backed ``sigma8`` transformed parameter spec."""
    cosmo_keys = tuple(cosmo_keys)
    fixed_cosmo_values = dict(fixed_cosmo_values or {})
    emulator_parameters = getattr(pklin_emulator, "parameters", None)
    if emulator_parameters is not None:
        emulator_parameters = tuple(str(key) for key in emulator_parameters)
        required_names = [key for key in emulator_parameters if key != "z"]
    else:
        required_names = list(cosmo_keys)

    if h_name not in required_names:
        required_names.append(h_name)
    required_set = set(required_names)
    native_names = tuple(key for key in cosmo_keys if key in required_set)
    available_names = set(native_names) | set(fixed_cosmo_values)
    missing = [key for key in required_names if key not in available_names]
    if missing:
        raise KeyError(
            "sigma8_spec requires native cosmology keys "
            f"{tuple(required_names)}; provide missing fixed values for "
            f"{tuple(missing)} via fixed_cosmo_values"
        )

    modes = jnp.asarray(pklin_emulator.modes, dtype=jnp.float64)
    z_value = jnp.atleast_1d(jnp.asarray(sigma8_redshift, dtype=jnp.float64))

    def transform(values: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        if h_name in values:
            h = jnp.asarray(values[h_name], dtype=jnp.float64)
        else:
            h = jnp.asarray(fixed_cosmo_values[h_name], dtype=jnp.float64)
        emulator_input = {}
        if emulator_parameters is None:
            for key in native_names:
                emulator_input[key] = jnp.atleast_1d(
                    jnp.asarray(values[key], dtype=jnp.float64)
                )
            for key, value in fixed_cosmo_values.items():
                emulator_input[key] = jnp.atleast_1d(jnp.asarray(value, dtype=jnp.float64))
            emulator_input["z"] = z_value
        else:
            for key in emulator_parameters:
                if key == "z":
                    emulator_input[key] = z_value
                elif key in values:
                    emulator_input[key] = jnp.atleast_1d(
                        jnp.asarray(values[key], dtype=jnp.float64)
                    )
                else:
                    emulator_input[key] = jnp.atleast_1d(
                        jnp.asarray(fixed_cosmo_values[key], dtype=jnp.float64)
                    )
        pklin = jnp.ravel(
            jnp.asarray(pklin_emulator.predict(emulator_input), dtype=jnp.float64)
        )
        return sigma8_from_linear_pk(modes / h, pklin)

    return ParameterSpec(
        name=name,
        native_names=native_names,
        transform=transform,
        label=label or _default_label(name),
    )


def make_lcdm_corner_specs(
    cosmo_keys: Sequence[str],
    *,
    pklin_emulator,
    omega_b_name: str = "ombh2",
    omega_cdm_name: str = "omch2",
    h_name: str = "h",
    mnu_name: str | None = None,
    mnu_fixed: float = 0.06,
    fixed_cosmo_values: Mapping[str, float] | None = None,
    sigma8_redshift: float = 0.0,
) -> tuple[ParameterSpec, ...]:
    """Return the standard ``(Omega_m, H0, sigma8)`` corner-plot specs."""
    cosmo_keys = tuple(cosmo_keys)
    fixed_cosmo_values = {
        key: value
        for key, value in dict(fixed_cosmo_values or {}).items()
        if key not in set(cosmo_keys)
    }
    return (
        omega_m_spec(
            omega_b_name=omega_b_name,
            omega_cdm_name=omega_cdm_name,
            h_name=h_name,
            mnu_name=mnu_name,
            mnu_fixed=mnu_fixed,
            fixed_cosmo_values=fixed_cosmo_values,
        ),
        H0_spec(h_name=h_name, fixed_cosmo_values=fixed_cosmo_values),
        sigma8_spec(
            cosmo_keys,
            pklin_emulator=pklin_emulator,
            h_name=h_name,
            fixed_cosmo_values=fixed_cosmo_values,
            sigma8_redshift=sigma8_redshift,
        ),
    )


def project_case_to_specs(
    fisher,
    packed_varied,
    native_param_idx,
    native_param_names: Sequence[str],
    specs: Sequence[ParameterSpec | str],
    *,
    rcond: float = 1e-12,
) -> ProjectedFisherResult:
    """Marginalize and project a Fisher case to a final plotted parameter list.

    Parameters
    ----------
    fisher
        Fisher matrix in the full varied basis.
    packed_varied
        Fiducial values in the same varied basis.
    native_param_idx
        Positions of the native cosmology block inside the varied basis.
    native_param_names
        Names of those native parameters in the same order as
        ``native_param_idx``.
    specs
        Final plotted parameters. Entries can be :class:`ParameterSpec`
        objects or plain native-parameter names.
    rcond
        Relative cutoff used for pseudo-inverse fallbacks.
    """
    normalized_specs = _normalize_specs(specs)
    native_param_names = tuple(str(name) for name in native_param_names)
    native_param_idx = tuple(int(idx) for idx in native_param_idx)
    if len(native_param_names) != len(native_param_idx):
        raise ValueError(
            "native_param_names and native_param_idx must have the same length"
        )
    if len(set(native_param_names)) != len(native_param_names):
        raise ValueError(
            f"native_param_names must be unique, got {native_param_names}"
        )

    index_by_name = dict(zip(native_param_names, native_param_idx, strict=True))
    required_native_names = {
        name for spec in normalized_specs for name in spec.native_names
    }
    native_keep_names = tuple(
        name for name in native_param_names if name in required_native_names
    )
    missing = [name for name in required_native_names if name not in index_by_name]
    if missing:
        raise KeyError(
            f"requested native parameters are not available in this case: {missing}"
        )
    native_keep_idx = tuple(index_by_name[name] for name in native_keep_names)

    fisher_native = marginalized_fisher_block(fisher, native_keep_idx, rcond=rcond)
    fid_native = np.asarray(packed_varied, dtype=float)[np.array(native_keep_idx)]
    derived_fn = _make_projection_fn(native_keep_names, normalized_specs)
    fisher_plot, fid_plot, jacobian, covariance_plot = project_fisher_to_derived(
        fisher_native,
        fid_native,
        derived_fn,
        rcond=rcond,
    )

    return ProjectedFisherResult(
        fisher_plot=fisher_plot,
        fid_plot=fid_plot,
        plot_names=tuple(spec.name for spec in normalized_specs),
        plot_labels=tuple(spec.label or _default_label(spec.name) for spec in normalized_specs),
        native_keep_names=native_keep_names,
        native_keep_idx=native_keep_idx,
        jacobian=jacobian,
        covariance_plot=covariance_plot,
    )
