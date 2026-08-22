# Installation

jaxPTPolyPol is the *inference* half of a two-repository stack. The *model* half —
`ps_1loop_jax`, which supplies the 1-loop galaxy power spectrum, the tree-level
bispectrum, and the flat $\nu\Lambda$CDM background — lives in the companion repository
`ps_1loop_jax-for-pfs` and is not distributed on PyPI. Install it first, then this
package.

## Prerequisites

- Python 3.9 or newer.
- `git`, with SSH access to GitHub if you want the bundled BAO data submodule.
- A working JAX installation for your platform. The package declares `jax>=0.4` and
  `jaxlib>=0.4`; if you need GPU or TPU wheels, install them from the JAX project's own
  instructions *before* the steps below so pip does not pull a CPU-only wheel over them.

## Step 1 — install the model layer

Clone `ps_1loop_jax-for-pfs` and install it in editable mode:

```bash
git clone <ps_1loop_jax-for-pfs>
cd ps_1loop_jax-for-pfs
pip install -e .
```

Do this **first**. The `full` extra of jaxPTPolyPol lists `ps_1loop_jax` as a
requirement, and there is no PyPI distribution by that name — installing the companion
checkout up front leaves the requirement already satisfied, so pip does not try (and
fail) to resolve it from an index.

## Step 2 — install the inference layer

```bash
git clone <jaxPTPolyPol>
cd jaxPTPolyPol
git submodule update --init          # ext_data/bao_data (DESI BAO measurements)
pip install -e ".[full]"
```

The `full` extra pulls:

`cosmopower-jax`
: the neural-network emulator for the linear matter power spectrum.

`ps_1loop_jax`
: the model layer — already satisfied by Step 1.

`quadax`
: quadrature used inside the background and projection integrals.

`matplotlib`
: required by {doc}`api/plotting` and {doc}`api/chain_analysis`.

`numdifftools`
: finite-difference Jacobians, used in the example notebooks to cross-check the JAX
  derivatives.

`candl-like`, `clipy-like`
: the CMB likelihood stack. Needed only to *rebuild* the cached CMB Fisher block — see
  [CMB extras](#cmb-extras-optional) below.

The `ext_data/bao_data` submodule is the upstream `CobayaSampler/bao_data` tree. The BAO
loaders (`load_desi_dr2`, `load_desi_2024` in {doc}`api/bao`) read the measurement and
covariance text files from it, so the BAO and joint examples need it checked out.

## Step 3 — enable 64-bit precision

JAX defaults to 32-bit floats. **Every** entry point in this package assumes float64:
Fisher matrices are inverted, Cholesky factorisations are taken of near-degenerate prior
blocks, and several regression checks assert agreement at the float64 floor. Put this at
the very top of any script or notebook, before the first JAX array is created:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Step 4 — verify the installation

```bash
python3 -c "import jax; jax.config.update('jax_enable_x64', True); import jaxptpolypol; print(jaxptpolypol.__name__)"
pytest tests/ -q
```

The suite is described in {doc}`testing`; a healthy run reports
`255 passed, 15 deselected`. It exercises the library only — it needs neither the
emulator weights nor the CMB data described below.

## Step 5 — point the emulator at a network file

`CosmoEmulator` wraps a CosmoPower-JAX network stored as an `.npz` file, and takes its
path explicitly:

```python
from jaxptpolypol.model import CosmoEmulator

pklin_emulator = CosmoEmulator(
    probe="custom_log",
    emulator_path="/path/to/jense_2023_camb_lcdm_Pk_lin.npz",
)
print(pklin_emulator.parameters)   # the network's input parameter names
```

The weights file is **not** part of either repository. The example notebooks hard-code an
absolute path to a local copy of the Jense et al. (2023) CAMB $\Lambda$CDM linear-$P(k)$
network; substitute your own path. `pklin_emulator.parameters` tells you which
cosmological parameters that particular network expects, and those names must match the
keys you put in `CosmoParams` — see {doc}`usage`.

## CMB extras (optional)

Two of the CMB dependencies cannot be installed with pip, and you very probably do not
need them.

**What is not pip-installable:**

`candl_data`
: the dataset package behind `candl_data.ACT_DR6_lens`. It is not on PyPI; it ships
  inside the `candl` source tree and is installed from there:

  ```bash
  git clone https://github.com/Lbalkenhol/candl
  pip install -e candl -e candl/candl_data
  ```

Planck `.clik` likelihood trees
: roughly 2 GB of *data*, not a package — plik high-$\ell$ TTTEEE, commander low-$\ell$
  TT, simall low-$\ell$ EE, and Planck lensing. Their filesystem paths are constants
  inside `example/mcmc/scripts/build_cmb_fisher_block.py`, and their content is hashed
  into that script's `CMB_CONFIG_HASH`.

**When you need them.** Only to *rebuild* the cached CMB Fisher block
(`example/mcmc/cache/cmb_fisher_{lcdm,nulcdm}.npz`), which is what
`example/mcmc/scripts/build_cmb_fisher_block.py` produces, and to run the two
`example/fisher/fisher_cmb_candl_*.ipynb` notebooks that build a CMB Fisher matrix
directly from the likelihoods.

**When you do not.** Every notebook that *consumes* the cached block — including both
joint PFS + BAO + CMB + BBN MCMC notebooks — loads the `.npz` artifact and never imports
`candl`, `clipy`, or `candl_data`. `jaxptpolypol.cmb.load_candl_likelihood` imports
`candl` and `clipy` lazily, inside the function body, so `import jaxptpolypol` succeeds
without any of it. That laziness is the reason these are optional rather than core
dependencies.

If you install `.[full]` on a machine without the data, `candl-like` and `clipy-like`
will be present but unusable; nothing else is affected.
