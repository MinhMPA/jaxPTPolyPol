# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**jaxPTPolyPol** is a JAX-based framework for Fisher forecasts and inference using perturbation theory power spectrum and bispectrum multipoles. It wraps two upstream libraries:

- **cosmopower-jax** — neural network emulator for the linear matter power spectrum
- **ps_1loop_jax** (at `/Users/nguyenmn/ps_1loop_jax-for-pfs`) — 1-loop galaxy power spectrum and background cosmology in JAX

The pipeline: emulator predicts P_lin(k) → ps_1loop_jax computes P_ℓ(k) multipoles → jaxPTPolyPol handles Fisher matrices, covariances, priors, and plotting.

## Build and install

```bash
# jaxPTPolyPol (this repo)
cd /Users/nguyenmn/jaxPTPolyPol
pip install -e ".[full]"

# ps_1loop_jax (companion repo, required dependency)
cd /Users/nguyenmn/ps_1loop_jax-for-pfs
pip install -e .
```

## Running tests

Only ps_1loop_jax has unit tests:

```bash
cd /Users/nguyenmn/ps_1loop_jax-for-pfs
pytest tests/ -v              # all tests
pytest tests/test_ps_1loop.py # single file
pytest tests/ -k "test_name"  # single test
```

jaxPTPolyPol uses Jupyter notebooks in `example/fisher/` as integration tests. Run them manually to verify end-to-end.

## Architecture

### Two-repo structure

| Repo | Package | Role |
|------|---------|------|
| `/Users/nguyenmn/jaxPTPolyPol` | `jaxptpolypol` | Inference layer: parameters, theory factory, covariance, Fisher, priors, plotting |
| `/Users/nguyenmn/ps_1loop_jax-for-pfs` | `ps_1loop_jax` | Model layer: 1-loop power spectrum, tree bispectrum, flat νΛCDM background cosmology |

jaxPTPolyPol imports ps_1loop_jax in `model.py` (wraps `PowerSpectrum1Loop`), `theory.py` and `bao.py` (uses `background` module for H(z), D_A(z), growth factor).

### Core design pattern: factory closures

The central pattern is `make_pk_ell_fn()` in `theory.py`. It returns a closure `pk_fn(params, *, k)` where all static configuration (emulators, fiducial cosmology, AP on/off, multi-bin redshifts) is captured in the closure scope. This eliminates the need for `static_argnames` in `jax.jit` — the caller just does:

```python
pk_fn = make_pk_ell_fn(ells=(0,2,4), pklin_emulator=..., ps1loop_model=..., ap=True, ...)
jac = jax.jacfwd(jax.jit(pk_fn))(params, k=k)
```

### JAX pytree containers

`CosmoParams` and `SurveyParams` (in `params.py`) are registered JAX pytrees. Parameter **names and sizes are static** (compilation constants); parameter **values are traced** (differentiable). This is critical — never make names/keys dynamic.

- `CosmoParams`: flat dict of cosmological parameters, accessed by name (`cosmo.h`, `cosmo.omega_b`, etc.)
- `SurveyParams`: nested dict `{group: {key: val}}` flattened to `((group, key), val)` tuples internally

### Packed parameter vectors

For differentiation, parameters are packed into flat arrays:
- Single-bin: `[cosmo | survey]` via `pack_params()`
- Multi-bin: `[cosmo | survey_bin1 | survey_bin2 | ...]` via `pack_multibin_params()`

Cosmological parameters are **shared** across bins; survey/EFT parameters are **per-bin**.

### Alcock-Paczyński effect

AP distortion is split across the two packages:
- **ps_1loop_jax** takes `(alpha_perp, alpha_para)` as inputs to `get_pk_ell_ref()`
- **jaxPTPolyPol** computes alphas from cosmology: `alpha_perp = D_A_true/D_A_fid`, `alpha_para = H_fid/H_true`

Fiducial distances are computed once (outside JIT) via `compute_fiducial_distances()` and become static constants in the closure.

### Priors

`gaussian_prior_fisher()` builds a diagonal prior Fisher matrix. `build_prior_sigmas()` maps named parameters to indices. Priors are added as `F_total = fisher_matrix(cov, jac) + F_prior`. Survey priors accept either a single dict (same for all bins) or a list of dicts (per-bin).

### Neutrino mass handling

`make_pk_ell_fn` auto-detects whether `'mnu'` is in `cosmo_keys`. If present, it's traced (differentiable); if absent, uses `mnu_fixed` (default 0.06 eV) as a static constant. The background module (`ps_1loop_jax.background`) includes neutrino density via Ω_ν = m_ν/(93.14 h²).

## Module map (jaxptpolypol)

- `params.py` — CosmoParams, SurveyParams pytrees, pack/unpack utilities
- `model.py` — CosmoEmulator (wraps cosmopower-jax), PS1LoopModel (wraps ps_1loop_jax)
- `theory.py` — `make_pk_ell_fn` factory, `compute_fiducial_distances`
- `covariance.py` — Gaussian covariance for P_ℓ multipoles
- `inference.py` — `fisher_matrix`, `marginalize_fisher`, `gaussian_prior_fisher`, `build_prior_sigmas`
- `bao.py` — BAO observables and Fisher forecast
- `plotting.py` — `plot_contours`, `plot_Gaussian`, `triangle_plot`

## Key conventions

- All background quantities are in **physical units**: H(z) in km/s/Mpc, D_A(z) in Mpc
- Power spectrum k-modes are in **h/Mpc**; emulator modes are in **1/Mpc** (divided by h before use)
- Always enable 64-bit: `jax.config.update("jax_enable_x64", True)`
- Growth factor uses the Heath integral (scale-independent, background-level neutrino approximation)
- `plot_contours` draws 2 ellipses (1σ, 2σ) per call — deduplicate legends when overlaying multiple cases

## Lessons from past mistakes

- **Audit EVERY output path when adding a variant/diagnostic mode to a script.**
  (2026-08-04) Adding `--mnu-unbounded` to `desi_prior_validation.py`, the JSON
  path got the `_unbounded` tag but the chain `.npy` path did not, so the
  diagnostic run overwrote the bounded production chain (restored by
  deterministic re-run, seed 20260808). Before running a new mode at production
  scale, grep the script for ALL `save`/`write_text`/output-path expressions and
  confirm each one carries the mode tag; a smoke run does not catch paths that
  only fire in the non-smoke branch (`if not SMOKE`).
- Smoke runs cannot preview pathologies that only manifest at production scale in a
  different sampling regime (e.g. removing a prior wall: the 2k-step smoke stayed near
  the start point and looked healthy while the 200k chain escaped into an invalid
  extrapolation region and collapsed). For diagnostics that RELAX constraints, validate
  with a production-scale run before trusting the configuration.
- **Use `command grep` for any completeness claim — the shell `grep` lies here.**
  (2026-08-08) The default `grep` in this environment is a ugrep wrapper with
  `--ignore-files`, which SILENTLY skips whole directories its ignore rules cover
  (`.superpowers/` among them) and exits 0 with no output — during the fisher_joint
  CMB port it produced a first, false-empty repo-wide sweep for stale σ(m_ν) upper
  limits. Never conclude "no occurrences anywhere" from bare `grep`: use
  `command grep -rIn` (bypasses the wrapper) or `rg --no-ignore --hidden`, and state
  in the report WHICH tool the sweep used. Note zsh needs quoted globs
  (`--include='*.py'`).
