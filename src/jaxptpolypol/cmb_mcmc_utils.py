"""Shared configuration and helpers for the CMB + BAO + BBN MCMC runs.

Holds the run registry consumed by the ``example/mcmc`` notebooks and scripts:
the named probe combinations (which ``candl`` terms, whether BAO is included,
which cosmological parameters are sampled), the Planck 2018 fiducial point the
Fisher and MCMC analyses expand around, and the plotting order, colours, and
published reference values used when comparing chains.

Alongside the registry it provides the small utilities those runs need: JSON
save/load of a run artifact, the ``100theta -> H0`` solve that maps the sampled
basis onto the native ``candl`` cosmology, conversions from a sampled vector to
native :class:`~jaxptpolypol.params.CosmoParams` arrays, and a chunked map for
evaluating a derived quantity over a long chain without exhausting memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from ps_1loop_jax import background as bg

from .params import CosmoParams

ARTIFACT_STEM = "cmb_bao_bbn_LCDM"
CACHE_SUBDIR = Path("example/mcmc/cache/cmb_bao_bbn_lcdm")

COSMO_KEYS_NATIVE = ("H0", "ombh2", "omch2", "logA", "ns", "tau")
COSMO_SIZES_NATIVE = (1, 1, 1, 1, 1, 1)
NONPLANCK_SAMPLED_COSMO_KEYS = ("100theta", "ombh2", "omch2", "logA", "ns")
ACT_LENSING_ONLY_SAMPLED_COSMO_KEYS = ("H0", "ombh2", "omch2", "logA", "ns")
PLANCK_SAMPLED_COSMO_KEYS = NONPLANCK_SAMPLED_COSMO_KEYS + ("tau",)
# Planck 2018 TT,TE,EE+lowE+lensing+BAO best fit — kept in sync with the
# Fisher notebooks (example/fisher/fisher_cmb_candl_*.ipynb and
# fisher_joint_PFS_BAO_CMB_*.ipynb) so Fisher-vs-MCMC comparisons expand
# around the same point. Note: NUTS gets no tau gradient from clipy simall
# (zero-gradient table lookup; see ~/candl/clipy/fix_simall_grad.md) — the
# chain values remain exact, only tau mixing is inefficient.
DEFAULT_FIDUCIAL_NATIVE = {
    "H0": 67.66,
    "ombh2": 0.02242,
    "omch2": 0.11933,
    "logA": 3.047,
    "ns": 0.9665,
    "tau": 0.0561,
}

COMBINATION_CONFIGS = {
    "bao_bbn": {
        "label": "BAO + BBN",
        "cmb_term_names": (),
        "include_bao": True,
        "prior_mode": "bao_bbn",
        "sampled_cosmo_keys": NONPLANCK_SAMPLED_COSMO_KEYS,
    },
    "act_lensing_bbn": {
        "label": "ACT DR6 lensing + BBN",
        "cmb_term_names": ("act_dr6_lensing",),
        "include_bao": False,
        "prior_mode": "lensing_bbn",
        "sampled_cosmo_keys": ACT_LENSING_ONLY_SAMPLED_COSMO_KEYS,
    },
    "act_lensing_bao": {
        "label": "ACT DR6 lensing + BAO + BBN",
        "cmb_term_names": ("act_dr6_lensing",),
        "include_bao": True,
        "prior_mode": "lensing_bbn",
        "sampled_cosmo_keys": NONPLANCK_SAMPLED_COSMO_KEYS,
    },
    "act_planck_lensing_bao": {
        "label": "ACT + Planck lensing + BAO + BBN",
        "cmb_term_names": ("act_dr6_lensing", "planck_lensing"),
        "include_bao": True,
        "prior_mode": "lensing_bbn",
        "sampled_cosmo_keys": NONPLANCK_SAMPLED_COSMO_KEYS,
    },
    "planck_lensing_bao": {
        "label": "Planck lensing + BAO + BBN",
        "cmb_term_names": ("planck_lensing",),
        "include_bao": True,
        "prior_mode": "lensing_bbn",
        "sampled_cosmo_keys": NONPLANCK_SAMPLED_COSMO_KEYS,
    },
    "planck_primary": {
        "label": "Planck 2018 TTTEEE + lowT + lowE",
        "cmb_term_names": ("planck_highl", "planck_lowl_tt", "planck_lowl_ee"),
        "include_bao": False,
        "prior_mode": "planck_aniso",
        "sampled_cosmo_keys": PLANCK_SAMPLED_COSMO_KEYS,
    },
}

DEFAULT_PLOT_ORDER = (
    "bao_bbn",
    "act_lensing_bbn",
    "act_lensing_bao",
    "act_planck_lensing_bao",
    "planck_lensing_bao",
    "planck_primary",
)

COMBINATION_COLORS = {
    "BAO + BBN": "tab:gray",
    "ACT DR6 lensing + BBN": "tab:green",
    "ACT DR6 lensing + BAO + BBN": "tab:orange",
    "ACT + Planck lensing + BAO + BBN": "tab:red",
    "Planck lensing + BAO + BBN": "tab:purple",
    "Planck 2018 TTTEEE + lowT + lowE": "tab:blue",
    "Planck anisotropies": "tab:blue",
}

PAPER_REFERENCE_POINTS = {
    "Planck PR4 reference": {
        "sigma8": 0.811,
        "Omega_m": 0.314,
        "H0": 67.3,
        "source": "Table 2: Planck CMB aniso. (PR4 TT+TE+EE) + SRoll2 low-ell EE",
        "color": "tab:blue",
    },
    "ACT DR6 lensing + BAO + BBN": {
        "sigma8": 0.820,
        "S8": 0.840,
        "Omega_m": 0.315,
        "H0": 68.2,
        "source": "Table 2: ACT CMB lensing + BAO",
    },
    "ACT + Planck lensing + BAO + BBN": {
        "sigma8": 0.815,
        "S8": 0.830,
        "Omega_m": 0.312,
        "H0": 68.1,
        "source": "Table 2: ACT+Planck lensing + BAO",
    },
    "ACT + Planck lensing (extended) + BAO": {
        "sigma8": 0.820,
        "S8": 0.841,
        "Omega_m": 0.316,
        "H0": 68.3,
        "source": "Table 2: ACT+Planck lensing (extended) + BAO",
        "color": "black",
    },
}


def scalar_value(value: Any) -> float:
    """Coerce a 0-d array, JAX scalar, or Python number to a plain ``float``."""
    return float(np.asarray(value).reshape(()))


def ordered_union(name_lists: Sequence[Sequence[str]]) -> tuple[str, ...]:
    """Concatenate the name lists, dropping duplicates and preserving first-seen order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for names in name_lists:
        for name in names:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return tuple(ordered)


def summarize_scalar(samples: Sequence[float]) -> dict[str, float]:
    """Return median, 16/84 percentiles, one-sided errors, mean, and std of a 1-d sample."""
    arr = np.asarray(samples)
    q16, q50, q84 = np.percentile(arr, [16.0, 50.0, 84.0])
    return {
        "q16": float(q16),
        "median": float(q50),
        "q84": float(q84),
        "minus": float(q50 - q16),
        "plus": float(q84 - q50),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def get_cache_dir(repo_root: str | Path) -> Path:
    """Return (creating if needed) the run-artifact cache directory under ``repo_root``."""
    cache_dir = Path(repo_root) / CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def artifact_path_for_selector(selector: str, cache_dir: str | Path) -> Path:
    """Return the ``.npz`` artifact path for a registered combination selector."""
    if selector not in COMBINATION_CONFIGS:
        raise KeyError(f"unknown selector {selector!r}")
    return Path(cache_dir) / f"{ARTIFACT_STEM}_{selector}.npz"


def selector_to_label(selector: str) -> str:
    """Return the human-readable label registered for a combination selector."""
    return str(COMBINATION_CONFIGS[selector]["label"])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, jnp.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return str(value)


def save_run_artifact(
    path: str | Path,
    *,
    metadata: Mapping[str, Any],
    flat_samples: Any,
    flat_log_post: Any,
    whitening_scales: Any,
    fid_native: Any,
    acceptance_rate: Any,
    num_integration_steps: Any,
    is_divergent: Any,
) -> Path:
    """Write one chain (samples, log-posterior, whitening, diagnostics) to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(_jsonable(dict(metadata)), sort_keys=True)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(metadata_json),
        flat_samples=np.asarray(flat_samples),
        flat_log_post=np.asarray(flat_log_post),
        whitening_scales=np.asarray(whitening_scales),
        fid_native=np.asarray(fid_native),
        acceptance_rate=np.asarray(acceptance_rate),
        num_integration_steps=np.asarray(num_integration_steps),
        is_divergent=np.asarray(is_divergent),
    )
    return path


def load_run_artifact(path: str | Path) -> dict[str, Any]:
    """Read back an artifact written by :func:`save_run_artifact` as a plain dict."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        return {
            "path": path,
            "metadata": metadata,
            "flat_samples": np.asarray(data["flat_samples"]),
            "flat_log_post": np.asarray(data["flat_log_post"]),
            "whitening_scales": np.asarray(data["whitening_scales"]),
            "fid_native": np.asarray(data["fid_native"]),
            "acceptance_rate": np.asarray(data["acceptance_rate"]),
            "num_integration_steps": np.asarray(data["num_integration_steps"]),
            "is_divergent": np.asarray(data["is_divergent"]),
        }


def load_available_artifacts(
    cache_dir: str | Path,
    selectors: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load every artifact that exists in ``cache_dir``, keyed by selector, skipping missing ones."""
    chosen = tuple(selectors or DEFAULT_PLOT_ORDER)
    loaded: dict[str, dict[str, Any]] = {}
    for selector in chosen:
        path = artifact_path_for_selector(selector, cache_dir)
        if path.exists():
            loaded[selector] = load_run_artifact(path)
    return loaded


def solve_h_from_100theta(
    theta100: Any,
    ombh2: Any,
    omch2: Any,
    *,
    fiducial_h: float,
    mnu: float,
    neff: float,
    n_iter: int = 8,
) -> jnp.ndarray:
    """Newton-solve ``h`` from the sampled ``100*theta_star``, differentiably (fixed-iteration ``lax.scan``)."""
    target = jnp.asarray(theta100, dtype=jnp.float64) / 100.0
    ombh2 = jnp.asarray(ombh2, dtype=jnp.float64)
    omch2 = jnp.asarray(omch2, dtype=jnp.float64)

    def body(h, _):
        def objective(hh):
            return bg.theta_star(ombh2, omch2, hh, mnu=mnu, neff=neff) - target

        f = objective(h)
        df = jax.grad(objective)(h)
        safe_df = jnp.where(
            jnp.abs(df) < 1.0e-8,
            jnp.sign(df) * 1.0e-8 + (df == 0.0) * 1.0e-8,
            df,
        )
        delta = jnp.clip(f / safe_df, -0.08, 0.08)
        h_new = jnp.clip(h - delta, 0.4, 1.0)
        return h_new, None

    h0 = jnp.asarray(fiducial_h, dtype=jnp.float64)
    h_final, _ = jax.lax.scan(body, h0, xs=jnp.arange(n_iter))
    return h_final


def native_cosmo_dict_from_sampled(
    theta_cosmo: Any,
    sampled_keys: Sequence[str],
    *,
    fiducial_native: Mapping[str, Any],
    fiducial_sampled: Mapping[str, Any],
    mnu_fixed: float,
    neff_fixed: float,
) -> dict[str, jnp.ndarray]:
    """Map a sampled cosmology vector onto the native ``candl`` parameter dict, solving for ``H0`` when ``100theta`` is sampled."""
    theta_cosmo = jnp.asarray(theta_cosmo, dtype=jnp.float64)
    sampled = {key: theta_cosmo[i] for i, key in enumerate(sampled_keys)}
    native = {
        key: jnp.asarray(fiducial_native[key], dtype=jnp.float64)
        for key in COSMO_KEYS_NATIVE
    }
    for key in ("ombh2", "omch2", "logA", "ns", "tau"):
        if key in sampled:
            native[key] = jnp.asarray(sampled[key], dtype=jnp.float64)
    if "H0" in sampled:
        native["H0"] = jnp.asarray(sampled["H0"], dtype=jnp.float64)
    else:
        theta100 = sampled.get(
            "100theta",
            jnp.asarray(fiducial_sampled["100theta"], dtype=jnp.float64),
        )
        native["H0"] = 100.0 * solve_h_from_100theta(
            theta100,
            native["ombh2"],
            native["omch2"],
            fiducial_h=float(fiducial_native["H0"]) / 100.0,
            mnu=mnu_fixed,
            neff=neff_fixed,
        )
    return native


def native_cosmo_array_from_sampled(
    theta_cosmo: Any,
    sampled_keys: Sequence[str],
    *,
    fiducial_native: Mapping[str, Any],
    fiducial_sampled: Mapping[str, Any],
    mnu_fixed: float,
    neff_fixed: float,
) -> jnp.ndarray:
    """Same as :func:`native_cosmo_dict_from_sampled`, returned as a packed :class:`~jaxptpolypol.params.CosmoParams` array."""
    native = native_cosmo_dict_from_sampled(
        theta_cosmo,
        sampled_keys,
        fiducial_native=fiducial_native,
        fiducial_sampled=fiducial_sampled,
        mnu_fixed=mnu_fixed,
        neff_fixed=neff_fixed,
    )
    return CosmoParams(native).to_array()


def omega_m_from_native(native_theta: Any, *, mnu_fixed: float) -> jnp.ndarray:
    """Compute ``Omega_m`` (including the fixed neutrino density) from a native cosmology array."""
    native_theta = jnp.asarray(native_theta, dtype=jnp.float64)
    h = native_theta[0] / 100.0
    return (native_theta[1] + native_theta[2] + mnu_fixed / 93.14) / h**2


def chunked_map(values: Any, fn: Any, *, chunk_size: int) -> np.ndarray:
    """Apply ``fn`` over a long array in fixed-size chunks and concatenate the NumPy results."""
    values = np.asarray(values)
    outputs = []
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        chunk = jnp.asarray(values[start:stop], dtype=jnp.float64)
        outputs.append(np.asarray(fn(chunk)))
    if not outputs:
        return np.empty((0,), dtype=float)
    return np.concatenate(outputs, axis=0)
