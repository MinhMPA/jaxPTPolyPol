"""
BlackJAX-based MCMC sampling with parameter whitening.

All sampled parameters are centered at zero and rescaled to have
approximately unit variance, using Fisher-matrix-derived (or user-supplied)
scales.  This improves NUTS geometry in high-dimensional parameter spaces.

Requires
--------
blackjax >= 1.0
"""

import hashlib
import json
import pathlib

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import lax

__all__ = [
    "make_transform",
    "make_cholesky_transform",
    "make_full_params_fn",
    "make_gaussian_log_prior",
    "make_log_posterior",
    "warmup_nuts",
    "run_nuts",
    "run_rwmh",
    "run_rwmh_python",
    "run_damh_python",
    "samples_to_physical",
    "chain_cache_key",
    "chain_cache_path",
    "save_chain_cache",
    "load_chain_cache",
    "run_chain_cached",
]

_DEFAULT_SCAN_CHUNK = 128


def _resolve_tree_depths(max_tree_depth):
    """Resolve NUTS tree-depth configuration.

    Accept either a single positive integer or a ``(warmup, sample)`` pair.
    """
    if isinstance(max_tree_depth, (tuple, list)):
        if len(max_tree_depth) != 2:
            raise ValueError(
                "max_tree_depth tuple/list must have length 2: "
                f"got {max_tree_depth!r}"
            )
        warmup_depth, sample_depth = max_tree_depth
    else:
        warmup_depth = sample_depth = max_tree_depth

    warmup_depth = int(warmup_depth)
    sample_depth = int(sample_depth)
    if warmup_depth <= 0 or sample_depth <= 0:
        raise ValueError(
            "max_tree_depth entries must be positive integers: "
            f"got {max_tree_depth!r}"
        )
    return warmup_depth, sample_depth


# ---------------------------------------------------------------------------
# Parameter whitening
# ---------------------------------------------------------------------------

def make_transform(center, scale):
    """Create a centering + rescaling transform pair.

    Parameters
    ----------
    center : array_like, shape (n,)
        Fiducial values (becomes the origin in whitened space).
    scale : array_like, shape (n,)
        Per-parameter scales (e.g. Fisher-based 1-sigma widths).

    Returns
    -------
    to_whitened, to_physical : pair of callables
        ``to_whitened(theta) = (theta - center) / scale``
        ``to_physical(x) = center + scale * x``
    """
    center = jnp.asarray(center, dtype=jnp.float64)
    scale = jnp.asarray(scale, dtype=jnp.float64)

    def to_whitened(theta):
        return (theta - center) / scale

    def to_physical(x):
        return center + scale * x

    return to_whitened, to_physical


def make_cholesky_transform(center, cov):
    """Create a full-covariance (Cholesky) whitening transform pair.

    Unlike :func:`make_transform`, which rescales each parameter
    independently and therefore leaves the whitened posterior with the
    original correlation structure, this uses the Cholesky factor of a
    covariance estimate (e.g. the inverse Fisher matrix) so the whitened
    posterior is approximately the isotropic unit Gaussian. That makes the
    standard random-walk proposal scale ``2.38/sqrt(d)`` correct by
    construction. Measured motivation: with diagonal whitening on the 7-bin
    P+B marginal posterior (strongly correlated cosmology block), isotropic
    proposals at that scale were rejected 60/60 times; see
    ``docs/design/perbin-compile-measurements.md``.

    Parameters
    ----------
    center : array_like, shape (n,)
        Fiducial values (origin in whitened space).
    cov : array_like, shape (n, n)
        Covariance estimate in physical space (symmetric positive definite),
        typically ``inv(F)`` for a Fisher matrix ``F``.

    Returns
    -------
    to_whitened, to_physical : pair of callables
        ``to_whitened(theta) = solve(L, theta - center)`` with ``L L^T = cov``;
        ``to_physical(x) = center + x @ L.T`` (works for a single ``(n,)``
        vector or any batch ``(..., n)``).
    """
    center = jnp.asarray(center, dtype=jnp.float64)
    cov = jnp.asarray(cov, dtype=jnp.float64)
    chol = jnp.linalg.cholesky(cov)

    def to_whitened(theta):
        return jax.scipy.linalg.solve_triangular(
            chol, jnp.asarray(theta, dtype=jnp.float64) - center, lower=True)

    def to_physical(x):
        return center + jnp.asarray(x, dtype=jnp.float64) @ chol.T

    return to_whitened, to_physical


def make_full_params_fn(fiducial_params, varied_idx):
    """Map varied parameter values into a full (fixed + varied) vector.

    Fixed parameters keep their fiducial values; varied slots are
    overwritten at each evaluation.

    Parameters
    ----------
    fiducial_params : array_like, shape ``(n_total,)``
        Full fiducial parameter vector.
    varied_idx : sequence of int
        Indices of the varied parameters.

    Returns
    -------
    to_full : callable
        ``to_full(varied_values) -> full_params``  (shape n_total)
    """
    fid = jnp.asarray(fiducial_params, dtype=jnp.float64)
    idx = jnp.array(varied_idx)

    def to_full(varied_values):
        return fid.at[idx].set(varied_values)

    return to_full


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

def make_gaussian_log_prior(n_varied, prior_entries=None):
    """Build a Gaussian log-prior in physical (un-whitened) space.

    Parameters
    ----------
    n_varied : int
        Total number of varied parameters.
    prior_entries : list of (int, float, float), optional
        Each entry is ``(index_in_varied, mean, sigma)``.
        Parameters without entries receive a flat (improper) prior.

    Returns
    -------
    log_prior : callable
        ``log_prior(theta_varied) -> scalar``
    """
    if not prior_entries:
        return lambda _theta: 0.0

    idx = jnp.array([e[0] for e in prior_entries])
    mu = jnp.array([e[1] for e in prior_entries], dtype=jnp.float64)
    inv_var = jnp.array([1.0 / e[2] ** 2 for e in prior_entries],
                        dtype=jnp.float64)

    def log_prior(theta_varied):
        return -0.5 * jnp.sum(inv_var * (theta_varied[idx] - mu) ** 2)

    return log_prior


# ---------------------------------------------------------------------------
# Log-posterior construction
# ---------------------------------------------------------------------------

def make_log_posterior(theory_fn, data, cov_inv, log_prior_fn,
                       to_physical, full_params_fn=None):
    """Build a JIT-compiled log-posterior in whitened parameter space.

    Parameters
    ----------
    theory_fn : callable
        Forward model ``theory_fn(full_params) -> theory_vector``.
        Any static arguments (e.g. *k*) should be pre-bound::

            from functools import partial
            theory_fn = partial(pk_fn, k=k)

    data : ndarray
        Data vector (mock or observed).
    cov_inv : ndarray
        Inverse covariance matrix.
    log_prior_fn : callable
        Log-prior evaluated in **physical** varied-parameter space.
    to_physical : callable
        Whitened *x* -> physical *theta_varied*.
    full_params_fn : callable, optional
        Inserts varied physical params into the full vector
        (from :func:`make_full_params_fn`).
        If *None*, ``theory_fn`` receives the varied params directly.

    Returns
    -------
    log_posterior : callable
        ``x -> scalar``, suitable for BlackJAX samplers.
    """
    data = jnp.asarray(data, dtype=jnp.float64)
    cov_inv = jnp.asarray(cov_inv, dtype=jnp.float64)

    if full_params_fn is not None:
        @jax.jit
        def log_posterior(x):
            theta = to_physical(x)
            full = full_params_fn(theta)
            theory = theory_fn(full)
            residual = data - theory
            return -0.5 * residual @ cov_inv @ residual + log_prior_fn(theta)
    else:
        @jax.jit
        def log_posterior(x):
            theta = to_physical(x)
            theory = theory_fn(theta)
            residual = data - theory
            return -0.5 * residual @ cov_inv @ residual + log_prior_fn(theta)

    return log_posterior


# ---------------------------------------------------------------------------
# RNG helpers (following fli_mf_nuts pattern for reproducible chunking)
# ---------------------------------------------------------------------------

def _make_step_keys(rng_key, start_index, num_steps):
    """Deterministic RNG keys via fold_in — safe across chunks."""
    base = jnp.asarray(start_index, dtype=jnp.uint32)
    offsets = base + jnp.arange(num_steps, dtype=jnp.uint32)
    return jax.vmap(lambda i: jr.fold_in(rng_key, i))(offsets)


# ---------------------------------------------------------------------------
# NUTS sampler
# ---------------------------------------------------------------------------

def warmup_nuts(rng_key, log_posterior_fn, initial_position,
                num_warmup=500, num_chains=4,
                adapt_mass_matrix=True, mass_matrix_type="diagonal",
                initial_inverse_mass_matrix=None,
                max_tree_depth=10,
                progress_fn=None,
                warmup_progress_fn=None):
    """Run BlackJAX NUTS window adaptation only.

    Parameters
    ----------
    rng_key : PRNGKey
        JAX random key.
    log_posterior_fn : callable
        Log-posterior in whitened space.
    initial_position : ndarray, shape (n_params,)
        Starting point in whitened space.
    num_warmup : int
        Warmup / window-adaptation steps per chain.
    num_chains : int
        Number of independent chains.
    adapt_mass_matrix : bool
        If *True* (default), adapt both step size and mass matrix.
        If *False*, only the step size is adapted and the inverse mass
        matrix is kept fixed.
    mass_matrix_type : ``"diagonal"`` or ``"dense"``
        Shape of the inverse mass matrix adapted (or provided).
    initial_inverse_mass_matrix : ndarray, optional
        Starting inverse mass matrix. Shape ``(n,)`` for diagonal or
        ``(n, n)`` for dense. If *None*, starts from the identity.
    max_tree_depth : int
        Maximum NUTS doubling depth used during warmup.
    progress_fn : callable, optional
        ``progress_fn(chain_number, num_chains)`` called after each
        chain's warmup completes.
    warmup_progress_fn : callable, optional
        Fine-grained warmup callback with signature
        ``warmup_progress_fn(chain_number, num_chains, stage, done, total)``.
        The ``stage`` is one of ``"find_reasonable_step_size"``,
        ``"dual_averaging"``, or ``"window_adaptation"``.

    Returns
    -------
    warmup_states : list
        Final sampler state for each chain after warmup.
    warmup_params : dict
        Dict containing stacked per-chain ``step_size`` and
        ``inverse_mass_matrix`` values.
    """
    import blackjax
    from blackjax.adaptation.step_size import (
        dual_averaging_adaptation,
        find_reasonable_step_size,
    )
    from blackjax.mcmc.nuts import build_kernel as _build_nuts_kernel

    if mass_matrix_type not in ("diagonal", "dense"):
        raise ValueError(
            f"mass_matrix_type must be 'diagonal' or 'dense', "
            f"got {mass_matrix_type!r}"
        )
    warmup_max_num_doublings = int(max_tree_depth)
    if warmup_max_num_doublings <= 0:
        raise ValueError(
            f"max_tree_depth must be a positive integer, got {max_tree_depth!r}"
        )
    is_diagonal = mass_matrix_type == "diagonal"

    initial_position = jnp.asarray(initial_position, dtype=jnp.float64)
    n_params = initial_position.shape[0]
    chain_keys = jax.random.split(rng_key, num_chains)

    if initial_inverse_mass_matrix is not None:
        init_inv_mass = jnp.asarray(initial_inverse_mass_matrix,
                                    dtype=jnp.float64)
    else:
        if is_diagonal:
            init_inv_mass = jnp.ones(n_params)
        else:
            init_inv_mass = jnp.eye(n_params)

    warmup_states = []
    warmup_step_sizes = []
    warmup_inv_masses = []
    nuts_kernel = _build_nuts_kernel()

    for c in range(num_chains):
        warmup_key, adapt_key = jax.random.split(chain_keys[c])

        if adapt_mass_matrix:
            if warmup_progress_fn is not None:
                warmup_progress_fn(
                    c + 1, num_chains, "window_adaptation", 0, num_warmup
                )
            warmup = blackjax.window_adaptation(
                blackjax.nuts,
                log_posterior_fn,
                is_mass_matrix_diagonal=is_diagonal,
                initial_step_size=1.0,
                progress_bar=False,
                max_num_doublings=warmup_max_num_doublings,
            )
            (state, params), _ = warmup.run(
                warmup_key, initial_position, num_steps=num_warmup,
            )
            if warmup_progress_fn is not None:
                warmup_progress_fn(
                    c + 1, num_chains, "window_adaptation",
                    num_warmup, num_warmup,
                )
        else:
            state = blackjax.nuts.init(initial_position, log_posterior_fn)

            def kernel_generator(step_size):
                def kernel(key, current_state):
                    return nuts_kernel(
                        key,
                        current_state,
                        log_posterior_fn,
                        step_size=step_size,
                        inverse_mass_matrix=init_inv_mass,
                        max_num_doublings=warmup_max_num_doublings,
                    )
                return kernel

            if warmup_progress_fn is not None:
                warmup_progress_fn(
                    c + 1, num_chains, "find_reasonable_step_size", 0, 1
                )
            step_size = find_reasonable_step_size(
                warmup_key,
                kernel_generator,
                state,
                jnp.asarray(1.0, dtype=jnp.float64),
                target_accept=0.8,
            )
            if warmup_progress_fn is not None:
                warmup_progress_fn(
                    c + 1, num_chains, "find_reasonable_step_size", 1, 1
                )
            da_init, da_update, da_final = dual_averaging_adaptation(0.8)
            da_state = da_init(step_size)
            progress_every = max(1, num_warmup // 50)

            if warmup_progress_fn is not None:
                warmup_progress_fn(
                    c + 1, num_chains, "dual_averaging", 0, num_warmup
                )

            for step, key in enumerate(jax.random.split(adapt_key, num_warmup), start=1):
                state, info = nuts_kernel(
                    key,
                    state,
                    log_posterior_fn,
                    step_size=step_size,
                    inverse_mass_matrix=init_inv_mass,
                    max_num_doublings=warmup_max_num_doublings,
                )
                da_state = da_update(da_state, info.acceptance_rate)
                step_size = jnp.exp(da_state.log_step_size)
                if warmup_progress_fn is not None and (
                    step == num_warmup or step % progress_every == 0
                ):
                    warmup_progress_fn(
                        c + 1, num_chains, "dual_averaging", step, num_warmup
                    )

            params = {
                "step_size": da_final(da_state),
                "inverse_mass_matrix": init_inv_mass,
            }

        warmup_states.append(state)
        warmup_step_sizes.append(params["step_size"])
        warmup_inv_masses.append(params["inverse_mass_matrix"])

        if progress_fn is not None:
            progress_fn(c + 1, num_chains)

    return warmup_states, {
        "step_size": jnp.stack([jnp.asarray(x) for x in warmup_step_sizes]),
        "inverse_mass_matrix": jnp.stack(
            [jnp.asarray(x) for x in warmup_inv_masses]
        ),
    }


def run_nuts(rng_key, log_posterior_fn, initial_position,
             num_warmup=500, num_samples=2000, num_chains=4,
             adapt_mass_matrix=True, mass_matrix_type="diagonal",
             initial_inverse_mass_matrix=None,
             max_tree_depth=10,
             scan_chunk_size=_DEFAULT_SCAN_CHUNK,
             parallel_chains=False,
             progress_fn=None,
             warmup_progress_fn=None,
             sample_progress_fn=None):
    """Run NUTS with BlackJAX window adaptation.

    The transition function is JIT-compiled with ``step_size`` and
    ``inverse_mass_matrix`` passed as **arguments** (not closed over),
    so changing tuning parameters never triggers recompilation.
    Production samples are drawn in chunked ``jax.lax.scan`` blocks
    (default 128 steps) to limit peak compilation memory.

    Parameters
    ----------
    rng_key : PRNGKey
        JAX random key.
    log_posterior_fn : callable
        Log-posterior in whitened space (from :func:`make_log_posterior`).
    initial_position : ndarray, shape (n_params,)
        Starting point in whitened space (typically ``jnp.zeros(n)``).
    num_warmup : int
        Warmup / window-adaptation steps per chain.
    num_samples : int
        Post-warmup samples per chain.
    num_chains : int
        Number of independent chains.
    adapt_mass_matrix : bool
        If *True* (default), use BlackJAX window adaptation to tune both
        the step size and the mass matrix during warmup.  If *False*,
        only the step size is adapted; the mass matrix is fixed to
        ``initial_inverse_mass_matrix`` (or the identity if not given).
    mass_matrix_type : ``"diagonal"`` or ``"dense"``
        Shape of the inverse mass matrix adapted (or provided).

        - ``"diagonal"``: 1-d array of length *n* (cheaper, default).
        - ``"dense"``: full *n × n* matrix (captures correlations but
          scales as O(n²) per leapfrog step).
    initial_inverse_mass_matrix : ndarray, optional
        Starting inverse mass matrix.  Shape ``(n,)`` for diagonal or
        ``(n, n)`` for dense.  Useful for seeding the adaptation from a
        Fisher-derived estimate.  If *None*, BlackJAX starts from the
        identity.
    max_tree_depth : int or (int, int)
        NUTS trajectory doubling cap. If a tuple/list of two ints is
        provided, interpret as ``(warmup_depth, sampling_depth)``.
    scan_chunk_size : int
        Number of samples per ``jax.lax.scan`` chunk.  Smaller chunks
        reduce peak compilation memory at the cost of slightly more
        Python-level overhead.  Default 128.
    parallel_chains : bool
        If *True*, production samples for all chains are drawn in
        parallel via ``jax.vmap``.  Warmup still runs sequentially
        (BlackJAX window adaptation uses Python control flow).
        This can improve GPU utilization when per-step computation
        does not saturate the device.  Default *False*.
    progress_fn : callable, optional
        ``progress_fn(chain_number, num_chains)`` called after each
        chain's warmup completes.  During parallel production sampling,
        a single call is made after all chains finish.
    warmup_progress_fn : callable, optional
        Fine-grained warmup callback forwarded to
        :func:`warmup_nuts`.  See that function for the callback
        signature.
    sample_progress_fn : callable, optional
        Progress callback for production samples, called after each
        chunk with signature
        ``sample_progress_fn(chain_number, num_chains, done, total)``.

        For sequential sampling, ``chain_number`` is 1-based and
        ``done`` counts completed draws for that chain. For parallel
        sampling, ``chain_number`` is *None* and ``done`` counts the
        shared per-chain draw count.

    Returns
    -------
    samples : ndarray, shape (num_chains, num_samples, n_params)
        Samples in **whitened** space.
    diagnostics : dict
        ``acceptance_rate``  (num_chains, num_samples)
        ``num_integration_steps``  (num_chains, num_samples)
        ``is_divergent``  (num_chains, num_samples)
    """
    import blackjax
    from blackjax.mcmc.nuts import build_kernel as _build_nuts_kernel

    if mass_matrix_type not in ("diagonal", "dense"):
        raise ValueError(
            f"mass_matrix_type must be 'diagonal' or 'dense', "
            f"got {mass_matrix_type!r}"
        )

    initial_position = jnp.asarray(initial_position, dtype=jnp.float64)
    n_params = initial_position.shape[0]
    chain_keys = jax.random.split(rng_key, num_chains)
    is_diagonal = mass_matrix_type == "diagonal"
    warmup_tree_depth, sample_tree_depth = _resolve_tree_depths(max_tree_depth)

    if initial_inverse_mass_matrix is not None:
        init_inv_mass = jnp.asarray(initial_inverse_mass_matrix,
                                    dtype=jnp.float64)
    elif is_diagonal:
        init_inv_mass = jnp.ones(n_params)
    else:
        init_inv_mass = jnp.eye(n_params)

    # ------------------------------------------------------------------
    # Build a parameterized NUTS kernel.
    # step_size and inverse_mass_matrix are JIT *arguments*, NOT
    # closed-over constants.  This means changing tuning parameters
    # (e.g. between warmup and sampling, or between chains) never
    # triggers recompilation of the XLA program.
    # ------------------------------------------------------------------
    nuts_kernel = _build_nuts_kernel()

    @jax.jit
    def _one_step(state, key, step_size, inverse_mass_matrix):
        state, info = nuts_kernel(
            key, state, log_posterior_fn,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            max_num_doublings=sample_tree_depth,
        )
        return state, (
            state.position,
            info.acceptance_rate,
            info.num_integration_steps,
            info.is_divergent,
        )

    # ------------------------------------------------------------------
    # Chunked scan: compile once, reuse across chunks and chains.
    # ------------------------------------------------------------------
    @jax.jit
    def _scan_chunk(state, keys, step_size, inverse_mass_matrix):
        def body(s, k):
            s, out = _one_step(s, k, step_size, inverse_mass_matrix)
            return s, out
        return lax.scan(body, state, keys)

    # Pre-compile the transition on a dummy call so the first real
    # chunk does not block.
    _dummy_key = jr.fold_in(rng_key, jnp.uint32(0xFFFF_FFFE))
    _dummy_keys = jr.split(_dummy_key, min(scan_chunk_size, 2))
    _init_state = blackjax.nuts.init(initial_position, log_posterior_fn)
    _ = _scan_chunk.lower(
        _init_state, _dummy_keys, jnp.float64(1.0), init_inv_mass
    ).compile()

    warmup_states, warmup_params = warmup_nuts(
        rng_key,
        log_posterior_fn,
        initial_position,
        num_warmup=num_warmup,
        num_chains=num_chains,
        adapt_mass_matrix=adapt_mass_matrix,
        mass_matrix_type=mass_matrix_type,
        initial_inverse_mass_matrix=initial_inverse_mass_matrix,
        max_tree_depth=warmup_tree_depth,
        progress_fn=progress_fn,
        warmup_progress_fn=warmup_progress_fn,
    )
    warmup_step_sizes = [x for x in warmup_params["step_size"]]
    warmup_inv_masses = [x for x in warmup_params["inverse_mass_matrix"]]

    # ------------------------------------------------------------------
    # Production sampling
    # ------------------------------------------------------------------
    if parallel_chains:
        return _sample_parallel(
            chain_keys, warmup_states, warmup_step_sizes,
            warmup_inv_masses, _scan_chunk, num_samples,
            scan_chunk_size, num_chains, sample_progress_fn,
        )
    else:
        return _sample_sequential(
            chain_keys, warmup_states, warmup_step_sizes,
            warmup_inv_masses, _scan_chunk, num_samples,
            scan_chunk_size, sample_progress_fn,
        )


def _sample_sequential(chain_keys, warmup_states, warmup_step_sizes,
                        warmup_inv_masses, scan_chunk_fn, num_samples,
                        scan_chunk_size, progress_fn=None):
    """Production sampling: one chain at a time (lower memory)."""
    all_positions = []
    all_accept = []
    all_steps = []
    all_divergent = []

    for c in range(len(warmup_states)):
        _, sample_key = jax.random.split(chain_keys[c])
        state = warmup_states[c]
        step_size = warmup_step_sizes[c]
        inv_mass = warmup_inv_masses[c]

        chunk_positions = []
        chunk_accept = []
        chunk_steps = []
        chunk_divergent = []

        n_remaining = num_samples
        key_offset = 0

        while n_remaining > 0:
            chunk_n = min(scan_chunk_size, n_remaining)
            keys = _make_step_keys(sample_key, key_offset, chunk_n)

            state, (pos, acc, stp, div) = scan_chunk_fn(
                state, keys, step_size, inv_mass,
            )

            chunk_positions.append(pos)
            chunk_accept.append(acc)
            chunk_steps.append(stp)
            chunk_divergent.append(div)

            key_offset += chunk_n
            n_remaining -= chunk_n

            if progress_fn is not None:
                progress_fn(c + 1, len(warmup_states), key_offset, num_samples)

        all_positions.append(jnp.concatenate(chunk_positions, axis=0))
        all_accept.append(jnp.concatenate(chunk_accept, axis=0))
        all_steps.append(jnp.concatenate(chunk_steps, axis=0))
        all_divergent.append(jnp.concatenate(chunk_divergent, axis=0))

    return (
        jnp.stack(all_positions),
        {
            "acceptance_rate": jnp.stack(all_accept),
            "num_integration_steps": jnp.stack(all_steps),
            "is_divergent": jnp.stack(all_divergent),
        },
    )


def _sample_parallel(chain_keys, warmup_states, warmup_step_sizes,
                      warmup_inv_masses, scan_chunk_fn, num_samples,
                      scan_chunk_size, num_chains, progress_fn=None):
    """Production sampling: all chains in parallel via vmap."""

    # Stack per-chain states into batched pytrees
    stacked_states = jax.tree.map(
        lambda *xs: jnp.stack(xs), *warmup_states
    )
    stacked_step_sizes = jnp.stack(
        [jnp.asarray(s) for s in warmup_step_sizes]
    )
    stacked_inv_masses = jnp.stack(
        [jnp.asarray(m) for m in warmup_inv_masses]
    )

    # Per-chain sample keys
    sample_keys = jnp.stack(
        [jax.random.split(chain_keys[c])[1] for c in range(num_chains)]
    )

    # Vmapped scan over a full chunk
    @jax.jit
    def _vmap_scan_chunk(states, all_keys, step_sizes, inv_masses):
        def _single_chain_chunk(state, keys, step_size, inv_mass):
            return scan_chunk_fn(state, keys, step_size, inv_mass)
        return jax.vmap(_single_chain_chunk)(
            states, all_keys, step_sizes, inv_masses
        )

    # Run production sampling in chunks, all chains simultaneously
    all_positions = []
    all_accept = []
    all_steps = []
    all_divergent = []

    n_remaining = num_samples
    key_offset = 0

    while n_remaining > 0:
        chunk_n = min(scan_chunk_size, n_remaining)

        # Generate keys for all chains: (num_chains, chunk_n, 2)
        all_chunk_keys = jnp.stack([
            _make_step_keys(sample_keys[c], key_offset, chunk_n)
            for c in range(num_chains)
        ])

        stacked_states, (pos, acc, stp, div) = _vmap_scan_chunk(
            stacked_states, all_chunk_keys,
            stacked_step_sizes, stacked_inv_masses,
        )

        all_positions.append(pos)
        all_accept.append(acc)
        all_steps.append(stp)
        all_divergent.append(div)

        key_offset += chunk_n
        n_remaining -= chunk_n

        if progress_fn is not None:
            progress_fn(None, num_chains, key_offset, num_samples)

    return (
        jnp.concatenate(all_positions, axis=1),
        {
            "acceptance_rate": jnp.concatenate(all_accept, axis=1),
            "num_integration_steps": jnp.concatenate(all_steps, axis=1),
            "is_divergent": jnp.concatenate(all_divergent, axis=1),
        },
    )


# ---------------------------------------------------------------------------
# Random-walk Metropolis-Hastings sampler (gradient-free)
# ---------------------------------------------------------------------------

def run_rwmh(rng_key, log_posterior_fn, initial_position, num_samples,
             num_chains=1, proposal_sigma=None, scan_chunk_size=1000,
             progress_fn=None, sample_progress_fn=None):
    """Run gradient-free random-walk Metropolis-Hastings (RWMH).

    .. warning::

       **For expensive posteriors (e.g. marginal-likelihood), use**
       :func:`run_rwmh_python` **instead.**  This function wraps each MH
       transition in ``jax.lax.scan``, which folds the *entire* posterior
       graph into a single XLA program.  On the production 7-bin marginal
       posterior that program never finished compiling — measured **zero
       draws in 60 min at 94 GB** (see
       ``docs/design/perbin-compile-measurements.md``, "The notebook
       blocker, resolved").  :func:`run_rwmh_python` drives the identical
       chain from a plain Python loop over an already-compiled
       ``log_posterior_fn`` and yields draws incrementally.  ``run_rwmh``
       remains fine for **cheap toy posteriors** whose graph compiles
       quickly inside a scan body.

    A forward-evaluation-only alternative to :func:`run_nuts` for
    posteriors whose gradient is intractable or explodes the XLA
    compilation graph.  This mirrors the Metropolis-
    Hastings usage in Chudaykin, Ivanov & Philcox (arXiv:2511.20757,
    §II), which drives inference with forward posterior calls only.

    The proposal is an isotropic Gaussian scaled by ``sigma_vector``.
    There is **no warmup or adaptation**: the caller is expected to work
    in a whitened parameter space where the posterior is approximately a
    unit Gaussian, so the standard optimal-scaling default
    ``(2.38 / sqrt(d)) * ones(d)`` is a good fixed proposal.

    Chains are run **sequentially** (one at a time) to keep peak memory
    low.  Each chain draws samples in chunked ``jax.lax.scan`` blocks;
    the BlackJAX transition is JIT-compiled once and reused across
    chunks and chains.

    Parameters
    ----------
    rng_key : PRNGKey
        JAX random key.
    log_posterior_fn : callable
        Log-posterior ``x -> scalar`` (typically in whitened space,
        from :func:`make_log_posterior`).
    initial_position : ndarray, shape (n_params,)
        Starting point (typically ``jnp.zeros(n)`` in whitened space).
    num_samples : int
        Samples drawn per chain.
    num_chains : int
        Number of independent chains.
    proposal_sigma : None, scalar, or ndarray, optional
        Per-parameter proposal standard deviations.

        - *None* (default): ``(2.38 / sqrt(d)) * ones(d)``.
        - scalar: broadcast to a length-*d* vector.
        - vector, shape ``(d,)``: used as-is.
    scan_chunk_size : int
        Number of steps per ``jax.lax.scan`` chunk.  Default 1000.
    progress_fn : callable, optional
        ``progress_fn(chain_number, num_chains)`` called after each
        chain completes (1-based ``chain_number``).
    sample_progress_fn : callable, optional
        Per-chunk callback with signature
        ``sample_progress_fn(chain_number, num_chains, done, total)``,
        where ``chain_number`` is 1-based and ``done`` counts completed
        draws for that chain.

    Returns
    -------
    samples : ndarray, shape (num_chains, num_samples, n_params)
        Samples in the sampled parameter space.
    diagnostics : dict
        ``acceptance_rate``  (num_chains,) mean acceptance per chain.
        ``proposal_sigma``  (n_params,) the proposal vector used.
    """
    import blackjax
    import blackjax.mcmc.random_walk as random_walk

    initial_position = jnp.asarray(initial_position, dtype=jnp.float64)
    d = initial_position.shape[0]

    # Resolve the proposal standard-deviation vector.
    if proposal_sigma is None:
        sigma_vector = (2.38 / jnp.sqrt(d)) * jnp.ones(d, dtype=jnp.float64)
    else:
        proposal_sigma = jnp.asarray(proposal_sigma, dtype=jnp.float64)
        if proposal_sigma.ndim == 0:
            sigma_vector = proposal_sigma * jnp.ones(d, dtype=jnp.float64)
        else:
            sigma_vector = proposal_sigma
    sigma_vector = jnp.asarray(sigma_vector, dtype=jnp.float64)

    rwmh = blackjax.additive_step_random_walk(
        log_posterior_fn, random_walk.normal(sigma_vector)
    )

    # JIT the transition once; reused across chunks and chains.
    @jax.jit
    def _scan_chunk(state, keys):
        def body(s, k):
            s, info = rwmh.step(k, s)
            return s, (s.position, info.is_accepted)
        return lax.scan(body, state, keys)

    chain_keys = jax.random.split(rng_key, num_chains)

    all_positions = []
    all_accept = []

    for c in range(num_chains):
        init_key, sample_key = jax.random.split(chain_keys[c])

        # Jitter the shared start point per chain.
        jitter = 0.1 * sigma_vector * jax.random.normal(init_key, (d,))
        state = rwmh.init(initial_position + jitter)

        chunk_positions = []
        chunk_accept = []

        n_remaining = num_samples
        key_offset = 0

        while n_remaining > 0:
            chunk_n = min(scan_chunk_size, n_remaining)
            keys = _make_step_keys(sample_key, key_offset, chunk_n)

            state, (pos, acc) = _scan_chunk(state, keys)

            chunk_positions.append(pos)
            chunk_accept.append(acc)

            key_offset += chunk_n
            n_remaining -= chunk_n

            if sample_progress_fn is not None:
                sample_progress_fn(c + 1, num_chains, key_offset, num_samples)

        all_positions.append(jnp.concatenate(chunk_positions, axis=0))
        all_accept.append(jnp.concatenate(chunk_accept, axis=0))

        if progress_fn is not None:
            progress_fn(c + 1, num_chains)

    samples = jnp.stack(all_positions)
    acceptance_rate = jnp.stack(
        [acc.mean() for acc in all_accept]
    )

    return samples, {
        "acceptance_rate": acceptance_rate,
        "proposal_sigma": sigma_vector,
    }


def run_rwmh_python(rng_key, log_posterior_fn, initial_position, num_samples,
                    num_chains=1, proposal_sigma=None, thin=1,
                    progress_fn=None, sample_progress_fn=None):
    """Run gradient-free random-walk MH with a Python-driven step loop.

    A same-shaped alternative to :func:`run_rwmh` that steps the chain from a
    **plain Python loop** instead of wrapping the transition in
    ``jax.lax.scan``.  It exists for a *measured* failure mode: on the
    production 7-bin marginal posterior, :func:`run_rwmh`'s ``lax.scan``
    wrapper produced **zero draws in 60 min at 94 GB** — the scan body folds
    the already-huge marginal graph into one monolithic XLA program whose
    compilation never completes.  A plain Python loop over the *same*
    already-compiled ``log_posterior_fn`` stepped at ~5.7 s/step (one cached
    forward eval), ~50% acceptance, at a flat 28.5 GB, with draws
    accumulating.

    Differences from :func:`run_rwmh` (which are the whole point)
    -------------------------------------------------------------
    - **No ``lax.scan`` / ``fori_loop``.**  ``log_posterior_fn`` is called
      once per step from Python.  It is expected to be **already
      jit-compiled** by the caller; this driver never wraps it.
    - **Proposal noise and the Metropolis accept test use a NumPy RNG**,
      seeded deterministically from the per-chain JAX key (via
      :func:`jax.random.randint`), so ``same rng_key => identical samples``.
      Only the per-chain start jitter still uses JAX random (identical to
      :func:`run_rwmh`).
    - **Thinning.**  ``thin`` keeps every ``thin``-th step; ``num_samples``
      is the number of KEPT draws, so the loop runs ``num_samples * thin``
      total steps and the returned array is always
      ``(num_chains, num_samples, d)``.

    Everything else matches :func:`run_rwmh`: the ``proposal_sigma``
    resolution (``None`` -> ``(2.38 / sqrt(d)) * ones(d)``; scalar ->
    broadcast; vector -> as-is), the ``0.1 * sigma * normal`` per-chain start
    jitter, sequential chains, and the returned ``(samples, diagnostics)``
    contract.

    Parameters
    ----------
    rng_key : PRNGKey
        JAX random key.
    log_posterior_fn : callable
        Log-posterior ``x -> scalar``.  Expected to be **already
        jit-compiled**; it is called once per step and never re-wrapped.
    initial_position : ndarray, shape (d,)
        Starting point (typically ``jnp.zeros(d)`` in whitened space).
    num_samples : int
        Number of KEPT draws per chain.
    num_chains : int
        Number of independent chains (run sequentially).
    proposal_sigma : None, scalar, or ndarray, optional
        Per-parameter proposal standard deviations.

        - *None* (default): ``(2.38 / sqrt(d)) * ones(d)``.
        - scalar: broadcast to a length-*d* vector.
        - vector, shape ``(d,)``: used as-is.
    thin : int
        Keep every ``thin``-th step.  Total steps per chain is
        ``num_samples * thin``.  Default 1 (keep every step).
    progress_fn : callable, optional
        ``progress_fn(chain_number, num_chains)`` called after each chain
        completes (1-based ``chain_number``).
    sample_progress_fn : callable, optional
        Per-draw callback with signature
        ``sample_progress_fn(chain_number, num_chains, done, total)``, where
        ``chain_number`` is 1-based and ``done`` counts KEPT draws for that
        chain (``total`` == ``num_samples``).

    Returns
    -------
    samples : ndarray, shape (num_chains, num_samples, d)
        Samples in the sampled parameter space.
    diagnostics : dict
        ``acceptance_rate``  (num_chains,) mean acceptance per chain.
        ``proposal_sigma``  (d,) the proposal vector used.
    """
    initial_position = jnp.asarray(initial_position, dtype=jnp.float64)
    d = initial_position.shape[0]

    thin = int(thin)
    if thin < 1:
        raise ValueError(f"thin must be a positive integer, got {thin!r}")

    # Resolve the proposal standard-deviation vector (identical to run_rwmh).
    if proposal_sigma is None:
        sigma_vector = (2.38 / jnp.sqrt(d)) * jnp.ones(d, dtype=jnp.float64)
    else:
        proposal_sigma = jnp.asarray(proposal_sigma, dtype=jnp.float64)
        if proposal_sigma.ndim == 0:
            sigma_vector = proposal_sigma * jnp.ones(d, dtype=jnp.float64)
        else:
            sigma_vector = proposal_sigma
    sigma_vector = jnp.asarray(sigma_vector, dtype=jnp.float64)
    sigma_np = np.asarray(sigma_vector, dtype=np.float64)

    chain_keys = jax.random.split(rng_key, num_chains)
    total_steps = num_samples * thin

    all_positions = []
    acceptance = []

    for c in range(num_chains):
        init_key, sample_key = jax.random.split(chain_keys[c])

        # Per-chain start jitter (JAX random, identical to run_rwmh).
        jitter = 0.1 * sigma_vector * jax.random.normal(init_key, (d,))
        cur = np.asarray(initial_position + jitter, dtype=np.float64)

        # Per-chain NumPy RNG, seeded deterministically from the JAX key so
        # that ``same rng_key => identical samples``.
        seed = int(jax.random.randint(sample_key, (), 0, 2 ** 31 - 1))
        rng = np.random.default_rng(seed)

        cur_lp = float(log_posterior_fn(jnp.asarray(cur)))
        kept = np.empty((num_samples, d), dtype=np.float64)
        kept_count = 0
        n_accepted = 0

        for step in range(total_steps):
            prop = cur + sigma_np * rng.normal(size=d)
            lp = float(log_posterior_fn(jnp.asarray(prop)))
            if np.log(rng.random()) < (lp - cur_lp):
                cur, cur_lp = prop, lp
                n_accepted += 1
            if (step + 1) % thin == 0:
                kept[kept_count] = cur
                kept_count += 1
                if sample_progress_fn is not None:
                    sample_progress_fn(
                        c + 1, num_chains, kept_count, num_samples
                    )

        all_positions.append(kept)
        acceptance.append(
            n_accepted / total_steps if total_steps > 0 else 0.0
        )

        if progress_fn is not None:
            progress_fn(c + 1, num_chains)

    samples = jnp.asarray(np.stack(all_positions), dtype=jnp.float64)
    return samples, {
        "acceptance_rate": jnp.asarray(acceptance, dtype=jnp.float64),
        "proposal_sigma": sigma_vector,
    }


def run_damh_python(rng_key, log_post_exact, log_post_surrogate,
                    initial_position, num_samples, num_chains=1,
                    proposal_sigma=None, thin=1,
                    progress_fn=None, sample_progress_fn=None):
    """Run Python-driven delayed-acceptance Metropolis-Hastings (DAMH).

    An **exact-target** sampler that uses a cheap surrogate posterior to
    screen proposals and evaluates the expensive exact posterior only on
    the (few) proposals that survive the first stage.  This is the two-stage
    delayed-acceptance chain of Christen & Fox (2005), "Markov chain Monte
    Carlo Using an Approximation" (J. Comput. Graph. Statist. 14, 795).

    It is the delayed-acceptance analogue of :func:`run_rwmh_python` and
    shares all of its conventions (see below).  Like that driver it is
    **Python-driven**: neither posterior is ever wrapped in ``lax.scan`` /
    ``fori_loop`` — a measured repo rule, because folding the (already huge)
    marginal-posterior graph into a monolithic scan body never finishes
    compiling.  Both ``log_post_exact`` and ``log_post_surrogate`` are
    expected to be **already jit-compiled** by the caller and are called
    directly, once at a time.

    Algorithm (one step, symmetric proposal ``q``)
    ----------------------------------------------
    Let ``s`` be the surrogate posterior density and ``p`` the exact one
    (both supplied as *log* densities).  From the current state ``x``:

    1. Propose ``y ~ N(x, sigma^2 I)``.
    2. **Stage 1** (surrogate screen): accept ``y`` for promotion with
       probability ``min(1, s(y) / s(x))``.  On rejection the step ends
       immediately with **zero** exact evaluations — the chain stays at ``x``.
    3. **Stage 2** (exact correction): only for promoted proposals, evaluate
       the exact posterior once and accept the move with probability
       ``min(1, [p(y) s(x)] / [p(x) s(y)])``.  On rejection the chain stays
       at ``x`` (but the exact eval has already been spent).

    The stage-2 ratio ``[p(y) s(x)] / [p(x) s(y)]`` is the algebraic
    simplification of Christen & Fox's second-stage ratio
    ``[alpha1(y,x) p(y)] / [alpha1(x,y) p(x)]`` with the stage-1 acceptance
    ``alpha1(x,y) = min(1, s(y)/s(x))`` and a symmetric proposal; the two
    forms are identical in both the ``s(y) >= s(x)`` and ``s(y) < s(x)``
    branches.

    Exactness and economics
    -----------------------
    The composed transition satisfies detailed balance with respect to the
    **exact** posterior ``p``, so the chain targets ``p`` *exactly* for any
    strictly positive surrogate.  The surrogate quality affects only
    **efficiency** (how often a promoted proposal is also accepted at stage
    2, i.e. the mixing rate), **never correctness**: a perfect surrogate
    (``s == p``) makes every stage-2 ratio exactly 1 and recovers plain
    RWMH on ``p``; a poor surrogate merely wastes exact evaluations and slows
    mixing but still samples ``p``.

    The exact posterior is evaluated **once per stage-1 acceptance and never
    on a stage-1 rejection**, so the number of exact evaluations is
    ``~ (stage-1 acceptance rate) x (total steps)`` (plus one for the initial
    state) — at optimal random-walk scaling the stage-1 rate is ~20-25%, so
    the expensive posterior is touched only a fraction of the time compared
    with plain RWMH, which evaluates it every step.  Both ``p(x)`` and
    ``s(x)`` for the current state are cached and reused across steps.

    Shared conventions (identical to :func:`run_rwmh_python`)
    ---------------------------------------------------------
    ``proposal_sigma`` resolution (``None`` -> ``(2.38 / sqrt(d)) * ones(d)``;
    scalar -> broadcast; vector -> as-is); the ``0.1 * sigma * normal``
    per-chain start jitter drawn with **JAX** random from a distinct key per
    chain (via :func:`jax.random.split`); sequential chains; a per-chain
    **NumPy** RNG seeded deterministically from the JAX key (so
    ``same rng_key => identical samples``) that draws both the proposal noise
    and the (up to two) Metropolis uniforms per step; ``thin`` semantics
    (``num_samples`` is the number of KEPT draws, total steps per chain is
    ``num_samples * thin``); and the ``(num_chains, num_samples, d)`` float64
    return.

    Parameters
    ----------
    rng_key : PRNGKey
        JAX random key.
    log_post_exact : callable
        Exact (expensive) log-posterior ``x -> scalar``.  Expected to be
        **already jit-compiled**; evaluated only at stage-1-accepted
        proposals.
    log_post_surrogate : callable
        Surrogate (cheap) log-posterior ``x -> scalar``.  Expected to be
        **already jit-compiled**; evaluated at every proposal.
    initial_position : ndarray, shape (d,)
        Starting point (typically ``jnp.zeros(d)`` in whitened space).
    num_samples : int
        Number of KEPT draws per chain.
    num_chains : int
        Number of independent chains (run sequentially).
    proposal_sigma : None, scalar, or ndarray, optional
        Per-parameter proposal standard deviations.

        - *None* (default): ``(2.38 / sqrt(d)) * ones(d)``.
        - scalar: broadcast to a length-*d* vector.
        - vector, shape ``(d,)``: used as-is.
    thin : int
        Keep every ``thin``-th step.  Total steps per chain is
        ``num_samples * thin``.  Default 1 (keep every step).
    progress_fn : callable, optional
        ``progress_fn(chain_number, num_chains)`` called after each chain
        completes (1-based ``chain_number``).
    sample_progress_fn : callable, optional
        Per-draw callback with signature
        ``sample_progress_fn(chain_number, num_chains, done, total)``, where
        ``chain_number`` is 1-based and ``done`` counts KEPT draws for that
        chain (``total`` == ``num_samples``).

    Returns
    -------
    samples : ndarray, shape (num_chains, num_samples, d)
        Samples from the **exact** posterior, in the sampled parameter space.
    diagnostics : dict
        ``acceptance_rate``     (num_chains,) overall move rate =
        stage-2 accepts / total steps (fraction of steps the chain moved).
        ``proposal_sigma``      (d,) the proposal vector used.
        ``stage1_acceptance``   (num_chains,) stage-1 accepts / total steps.
        ``stage2_acceptance``   (num_chains,) stage-2 accepts / stage-1
        accepts (== 1.0 exactly when the surrogate equals the exact
        posterior; 0.0 if no proposal was ever promoted).
        ``n_exact_evals``       (num_chains,) total exact-posterior
        evaluations = 1 (initial state) + stage-1 accepts.
    """
    initial_position = jnp.asarray(initial_position, dtype=jnp.float64)
    d = initial_position.shape[0]

    thin = int(thin)
    if thin < 1:
        raise ValueError(f"thin must be a positive integer, got {thin!r}")

    # Resolve the proposal standard-deviation vector (identical to
    # run_rwmh_python).
    if proposal_sigma is None:
        sigma_vector = (2.38 / jnp.sqrt(d)) * jnp.ones(d, dtype=jnp.float64)
    else:
        proposal_sigma = jnp.asarray(proposal_sigma, dtype=jnp.float64)
        if proposal_sigma.ndim == 0:
            sigma_vector = proposal_sigma * jnp.ones(d, dtype=jnp.float64)
        else:
            sigma_vector = proposal_sigma
    sigma_vector = jnp.asarray(sigma_vector, dtype=jnp.float64)
    sigma_np = np.asarray(sigma_vector, dtype=np.float64)

    chain_keys = jax.random.split(rng_key, num_chains)
    total_steps = num_samples * thin

    all_positions = []
    acceptance = []
    stage1_acceptance = []
    stage2_acceptance = []
    n_exact_evals = []

    for c in range(num_chains):
        init_key, sample_key = jax.random.split(chain_keys[c])

        # Per-chain start jitter (JAX random, identical to run_rwmh_python).
        jitter = 0.1 * sigma_vector * jax.random.normal(init_key, (d,))
        cur = np.asarray(initial_position + jitter, dtype=np.float64)

        # Per-chain NumPy RNG, seeded deterministically from the JAX key so
        # that ``same rng_key => identical samples``.
        seed = int(jax.random.randint(sample_key, (), 0, 2 ** 31 - 1))
        rng = np.random.default_rng(seed)

        # Cache the exact AND surrogate log-density at the current state.
        s_cur = float(log_post_surrogate(jnp.asarray(cur)))
        p_cur = float(log_post_exact(jnp.asarray(cur)))
        n_exact = 1  # the initial exact evaluation

        kept = np.empty((num_samples, d), dtype=np.float64)
        kept_count = 0
        n_stage1 = 0
        n_stage2 = 0

        for step in range(total_steps):
            prop = cur + sigma_np * rng.normal(size=d)
            s_prop = float(log_post_surrogate(jnp.asarray(prop)))

            # STAGE 1: screen the proposal with the cheap surrogate.
            if np.log(rng.random()) < (s_prop - s_cur):
                n_stage1 += 1
                # STAGE 2: correct with a single exact evaluation.
                p_prop = float(log_post_exact(jnp.asarray(prop)))
                n_exact += 1
                stage2_log_ratio = (p_prop + s_cur) - (p_cur + s_prop)
                if np.log(rng.random()) < stage2_log_ratio:
                    cur, s_cur, p_cur = prop, s_prop, p_prop
                    n_stage2 += 1

            if (step + 1) % thin == 0:
                kept[kept_count] = cur
                kept_count += 1
                if sample_progress_fn is not None:
                    sample_progress_fn(
                        c + 1, num_chains, kept_count, num_samples
                    )

        all_positions.append(kept)
        acceptance.append(
            n_stage2 / total_steps if total_steps > 0 else 0.0
        )
        stage1_acceptance.append(
            n_stage1 / total_steps if total_steps > 0 else 0.0
        )
        stage2_acceptance.append(
            n_stage2 / n_stage1 if n_stage1 > 0 else 0.0
        )
        n_exact_evals.append(n_exact)

        if progress_fn is not None:
            progress_fn(c + 1, num_chains)

    samples = jnp.asarray(np.stack(all_positions), dtype=jnp.float64)
    return samples, {
        "acceptance_rate": jnp.asarray(acceptance, dtype=jnp.float64),
        "proposal_sigma": sigma_vector,
        "stage1_acceptance": jnp.asarray(stage1_acceptance, dtype=jnp.float64),
        "stage2_acceptance": jnp.asarray(stage2_acceptance, dtype=jnp.float64),
        "n_exact_evals": jnp.asarray(n_exact_evals, dtype=jnp.int64),
    }


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def samples_to_physical(samples, to_physical):
    """Convert whitened samples to physical parameter values.

    Parameters
    ----------
    samples : ndarray, shape (..., n_params)
        Samples in whitened space (any leading batch dimensions).
    to_physical : callable
        Whitened -> physical transform.

    Returns
    -------
    physical : ndarray, same shape
    """
    shape = samples.shape
    flat = samples.reshape(-1, shape[-1])
    physical = jax.vmap(to_physical)(flat)
    return physical.reshape(shape)


# ---------------------------------------------------------------------------
# Production-chain caching (2026-09-01).
#
# The joint MCMC notebooks sample live on every execution, so a purely
# cosmetic plotting change used to cost a full ~15 min re-sample per notebook.
# These helpers cache (samples, diagnostics) keyed by a SEMANTIC fingerprint.
#
# The safety design differs deliberately from the Taylor-cache guard: instead
# of enforce-on-load, the fingerprint is embedded in the FILENAME
# (``<tag>_chain_<key12>.npz``), so a changed config resolves to a different
# path and is a cache MISS -- fresh sampling -- never a warn-and-load of stale
# draws (the bk_do_irres lesson) and never a diagnostic run overwriting a
# production artifact (the 2026-08-04 output-path lesson; different configs
# coexist as different files). The full canonical config is additionally
# stored inside the file and re-verified at load, so a hand-built or renamed
# path hard-fails on mismatch rather than loading.
#
# Fingerprint recommendation for posterior chains: include the sampler
# settings (seed, lengths, chains), the prior variant, the sampled dimension,
# the theory-config hash -- and ``round(lp0, 2)``, the live-computed exact
# log-posterior at the fiducial. lp0 depends on every ingredient of the
# posterior (theory, data, covariance, priors, external blocks), so anything
# that moves the target moves the fingerprint even when no named switch
# changed; 2 decimals absorb the documented ~1e-3 cross-environment
# reproducibility of the absolute value (docs/source/testing.md).
# ---------------------------------------------------------------------------

_CHAIN_CACHE_SCALARS = (str, int, float, bool, type(None))


def _canonical_chain_config(config):
    """Canonical JSON string of a chain-cache fingerprint config.

    A flat mapping of scalars only; any other value raises ``TypeError``
    because it could not round-trip as a stable fingerprint.
    """
    items = {}
    for key, val in dict(config).items():
        if not isinstance(key, str):
            raise TypeError(f"config keys must be str, got {type(key).__name__}")
        if not isinstance(val, _CHAIN_CACHE_SCALARS):
            raise TypeError(
                f"chain-cache config value for {key!r} must be a scalar "
                f"(str/int/float/bool/None), got {type(val).__name__}")
        items[key] = val
    return json.dumps(items, sort_keys=True)


def chain_cache_key(config):
    """sha256 hex digest of the canonical config."""
    return hashlib.sha256(_canonical_chain_config(config).encode()).hexdigest()


def chain_cache_path(cache_dir, tag, config):
    """Cache path with the fingerprint embedded in the filename.

    Different configs resolve to different files, so a config change is a
    cache miss (fresh sampling), never a silent reuse, and smoke/production
    runs coexist.
    """
    return pathlib.Path(cache_dir) / f"{tag}_chain_{chain_cache_key(config)[:12]}.npz"


def save_chain_cache(path, samples, *, config, diagnostics=None):
    """Persist ``(samples, diagnostics)`` with the canonical config stamped."""
    payload = {
        "samples": np.asarray(samples),
        "__config_json": np.asarray(_canonical_chain_config(config)),
    }
    for key, val in (diagnostics or {}).items():
        payload[f"diag__{key}"] = np.asarray(val)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_chain_cache(path, *, config):
    """``(samples, diagnostics)`` if ``path`` exists and matches ``config``.

    Returns ``None`` when the file is absent (the caller samples). A present
    file whose stored config differs raises -- possible only for a path not
    produced by :func:`chain_cache_path`, since the fingerprint is in the
    filename there.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        stored = z["__config_json"].item()
        expected = _canonical_chain_config(config)
        if stored != expected:
            raise RuntimeError(
                f"chain cache {path} was written under a different config.\n"
                f"  stored:   {stored}\n  expected: {expected}\n"
                "Delete the file to re-sample.")
        samples = np.array(z["samples"])
        diagnostics = {k[len("diag__"):]: np.array(z[k])
                       for k in z.files if k.startswith("diag__")}
    return samples, diagnostics


def run_chain_cached(sampler_fn, *args, cache_path, cache_config, **kwargs):
    """Load a cached chain, or run ``sampler_fn`` and cache its result.

    Only the sampling call is ever skipped: every assertion and tripwire a
    caller evaluates before or after this call stays on the live path.
    """
    cached = load_chain_cache(cache_path, config=cache_config)
    if cached is not None:
        print(f"[chain cache] loaded {pathlib.Path(cache_path).name} "
              f"shape {tuple(cached[0].shape)}; delete the file to re-sample.")
        return cached
    samples, diagnostics = sampler_fn(*args, **kwargs)
    save_chain_cache(cache_path, samples, config=cache_config,
                     diagnostics=diagnostics)
    print(f"[chain cache] wrote {pathlib.Path(cache_path).name}")
    return samples, diagnostics

