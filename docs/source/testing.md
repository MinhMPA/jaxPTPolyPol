# Testing

Three layers guard this repository: a unit-test suite for the library, the companion
repository's own suite for the model layer, and the example notebooks — which are
integration tests whose committed outputs are the reference results. A fourth gate, the
documentation build, runs in CI on every push. See {doc}`installation` if the suite will
not run at all, and {doc}`usage` for what the notebooks actually do.

## The unit-test suite

```bash
cd jaxPTPolyPol
pytest tests/ -q
```

Expected: **`255 passed, 15 deselected`**. Anything else is a regression. The suite runs in
well under a minute, needs no emulator weights, no CMB data, and no cached artifacts.

### The deselected 15

Four test modules are marked `heavy` and are **deselected by default**:

- `test_marginal_pipeline`
- `test_marginal_perbin`
- `test_theory_perbin`
- `test_marginal_taylor_pipeline`

They build per-bin theory and marginal-likelihood graphs whose compiled evaluations are
not freed between modules, so collecting them in one process stacks to roughly 85 GB of
resident memory. Run them one file at a time:

```bash
pytest tests/test_theory_perbin.py
```

or opt in deliberately, on a machine that can take it:

```bash
pytest tests/ --run-heavy
```

The gating logic lives in `tests/conftest.py`.

### Useful invocations

```bash
pytest tests/test_marginal_likelihood.py     # one module
pytest tests/ -k "taylor"                    # by name
pytest tests/ -q -x                          # stop at the first failure
```

## The model layer's suite

`ps_1loop_jax` carries its own tests, and a change to the theory will show up there first:

```bash
cd ps_1loop_jax-for-pfs
pytest tests/ -v
```

Run it whenever you touch the model layer, or when a jaxPTPolyPol tripwire fires and you
suspect the theory rather than the inference code.

## Building the docs

This site is the CI gate that most often goes red on a documentation-only change, because
it is built with warnings promoted to errors. Reproduce it exactly:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W -b html docs/source docs/_build/html
```

`-W` is the whole point: a broken cross-reference, a heading not in any toctree, or a
malformed directive fails the build rather than printing a note. Read the Docs applies the
same rule through `fail_on_warning: true` in `.readthedocs.yaml`, so a green local build
with `-W` is what predicts a green RTD build.

Two habits keep it honest. Delete `docs/_build/` before a build you intend to trust —
Sphinx caches per-document, so an incremental rebuild will not re-emit a warning for a
document it did not re-read, and a cached build can pass where a clean one fails. And
install from `docs/requirements.txt` rather than relying on whatever Sphinx is already in
your environment: the pins there carry deliberate upper bounds, because Sphinx 9 tightened
ambiguous Python cross-reference resolution and turned annotations that had been building
cleanly into errors.

## Notebooks as integration tests

The notebooks under `example/fisher/` and `example/mcmc/` are the end-to-end tests. There
is no runner for them — they are executed by hand and committed **with their outputs**.

That last point is a deliberate convention, not an oversight. Committed outputs are
*evidence*:

- **They are the reference results.** A review verifies a number by reading it out of
  history — `git show <rev>:example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_LCDM.ipynb` — rather
  than by re-running a chain that takes hours.
- **Determinism gates compare them across runs.** Because the sampler seeds are pinned and
  the artifacts are content-hashed, re-executing a notebook should reproduce its outputs.
  Diffing the committed outputs against a fresh run is how a configuration drift gets
  caught.
- **They have caught real defects.** At least one genuine bug — negative marginal variances
  displayed as `---` — was found by reading a `RuntimeWarning` preserved in a committed
  cell output.

Stripping outputs would delete that audit trail, which is why this repository does *not*
use `nbstripout`. Executed notebooks of five to six megabytes are expected and allowed.

### The pre-commit hook

What must stay out of history is regenerable bulk: Taylor template caches (~21 MB), MCMC
chains, and other artifacts that a build script can recreate. A hook enforces the line:

```bash
ln -sf ../../scripts/git-hooks/pre-commit .git/hooks/pre-commit
```

It blocks any staged file over 10 MB, warns above 5 MB, and notes when the staged total
passes 25 MB. Regenerable artifacts belong in `example/mcmc/cache/`, untracked. If a large
file genuinely must be committed, `git commit --no-verify` bypasses the hook.

## Tripwires

A tripwire is a hard, exact assertion embedded in a notebook or script — a number that
cannot drift silently. There are two kinds.

**Exact log-posterior values.** Each production notebook records the log-posterior at the
fiducial in its configuration cell and asserts it. These are deterministic functions of
the configuration and the cached artifacts, not of the chain, so they hold identically in
the smoke and production branches. For example,
`example/mcmc/mcmc_joint_PFS_BAO_CMB_BBN_LCDM.ipynb` pins:

```text
log_post_joint(theta0) = -167.750608
chi2_prof(fiducial)    = 1.082e-23
```

and the $\nu\Lambda$CDM production run pins `log_post(x0) = -173.634058` with
`chi2_prof(fiducial) = 1.169e-23`.

```{admonition} Re-recorded 2026-08-23 — bispectrum IR-resummation flip
:class: note
These values were re-recorded, in the same commit as the notebook re-execution,
after the bispectrum default changed to IR-resummed
(`BispectrumTreeModel(do_irres=True)`). The flip moves the bispectrum entries of
the theory vector by up to $\sim 5\times 10^{-3}$ relative, which shifted the
log-posterior pins by $\sim 2\times 10^{-3}$ (previous values: `-167.752302` /
`-173.635756`). The Taylor template caches were rebuilt first, the surrogate
re-validated (tilt, chain, and importance-sampling gates), and all four joint
notebooks re-executed on the fresh caches.
```

**Exactness identities.** These assert a mathematical property rather than a recorded
value, so they cannot go stale:

- `chi2(fid) < 1e-10` — the data vector is a noiseless fiducial mock and every likelihood
  term is fiducial-centered, so the profile $\chi^2$ at the fiducial must be zero to the
  float64 floor. Values around $10^{-23}$ are normal.
- `abs(lp0_surr - lp0) < 1e-6` — the Taylor surrogate is exact at its expansion point, so
  it must reproduce the exact per-bin marginal posterior at the fiducial.
- `residual(2*c1) == 4*residual(c1)` — $c_1$ is the only linear-block parameter with
  self-curvature and has no cross terms with the others, so its template-reconstruction
  residual must scale exactly quadratically.
- Fiducial-centered external blocks contribute **exactly** zero at the fiducial, so
  `log_post_joint(theta0)` must equal the full-shape value at `x0`.

### What a tripwire does not prove

`lp0` is insensitive to some real mistakes. A fiducial-centered Gaussian prior contributes
exactly zero at the fiducial, so adding or removing one leaves `lp0` unchanged — the check
for that is the **prior-entry count**, not the log-posterior value. Similarly, `lp0`
confirms wiring, not correctness of the physics: it says the pieces are connected at one
point in parameter space.

### When a tripwire fires

Do not update the recorded number to match the new output. Find out what moved.

1. **Cached artifacts.** `load_taylor_templates(..., expect_meta=template_meta_for(...))`
   plus the explicit stored-hash check, and `load_cmb_fisher_block(...)`, carry hard
   guards on the theory-configuration stamp and, for the CMB block, a content-derived
   config hash. (The bare `META` dict is not a theory-config guard — it names only the
   grid keys.) If one of those raised, the artifact
   was built against a different configuration — rebuild it rather than loosening the
   guard.
2. **The configuration cell.** $k$ grid and range, triangle range, `k_nl_rsd`, per-bin
   volumes and number densities, `PRIOR_VARIANT` — all of these move `lp0`.
3. **The prior specification.** Which spec, which phase, `marginal_means="fiducial"` versus
   `"spec"`, and whether cosmology priors are double-counted against BBN or the CMB block.
4. **The model layer.** Run the `ps_1loop_jax` suite; a theory change propagates into every
   tripwire at once, which is a useful signature.

Only once you can name the cause should the value be re-recorded — in the same commit as
the change that caused it, with the cause stated.

### Smoke runs are not production runs

Every production notebook has a `SMOKE_TEST` branch that exercises the same code path with
a tiny chain. It is a wiring gate, and it is a good one — but it cannot preview a pathology
that only appears at production scale. A relaxed-constraint diagnostic is the canonical
trap: a 2000-step smoke chain stays near its start point and looks healthy while the
200000-step production chain escapes into an invalid extrapolation region. Any change that
*removes* a constraint must be validated at production scale before it is trusted.
