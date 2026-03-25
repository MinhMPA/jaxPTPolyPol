"""
BlackJAX-based MCMC sampling with parameter whitening.

All sampled parameters are centered at zero and rescaled to have
approximately unit variance, using Fisher-matrix-derived (or user-supplied)
scales.  This improves NUTS geometry in high-dimensional parameter spaces.

Requires
--------
blackjax >= 1.0
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

__all__ = [
    "make_transform",
    "make_full_params_fn",
    "make_gaussian_log_prior",
    "make_log_posterior",
    "warmup_nuts",
    "run_nuts",
    "samples_to_physical",
]

_DEFAULT_SCAN_CHUNK = 128


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


def make_full_params_fn(fiducial_params, varied_idx):
    """Map varied parameter values into a full (fixed + varied) vector.

    Fixed parameters keep their fiducial values; varied slots are
    overwritten at each evaluation.

    Parameters
    ----------
    fiducial_params : array_like, shape (n_total,)
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
                progress_fn=None):
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
    progress_fn : callable, optional
        ``progress_fn(chain_number, num_chains)`` called after each
        chain's warmup completes.

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
            warmup = blackjax.window_adaptation(
                blackjax.nuts,
                log_posterior_fn,
                is_mass_matrix_diagonal=is_diagonal,
                initial_step_size=1.0,
            )
            (state, params), _ = warmup.run(
                warmup_key, initial_position, num_steps=num_warmup,
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
                    )
                return kernel

            step_size = find_reasonable_step_size(
                warmup_key,
                kernel_generator,
                state,
                jnp.asarray(1.0, dtype=jnp.float64),
                target_accept=0.8,
            )
            da_init, da_update, da_final = dual_averaging_adaptation(0.8)
            da_state = da_init(step_size)

            for key in jax.random.split(adapt_key, num_warmup):
                state, info = nuts_kernel(
                    key,
                    state,
                    log_posterior_fn,
                    step_size=step_size,
                    inverse_mass_matrix=init_inv_mass,
                )
                da_state = da_update(da_state, info.acceptance_rate)
                step_size = jnp.exp(da_state.log_step_size)

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
             scan_chunk_size=_DEFAULT_SCAN_CHUNK,
             parallel_chains=False,
             progress_fn=None,
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
        progress_fn=progress_fn,
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
