# `bao`

BAO distance observables and their Fisher forecast. Implements the DESI-style
compressed distance ratios (``DM/rs``, ``DH/rs``, ``DV/rs``) as differentiable JAX
functions, loads cobaya-format measurement and covariance files, and exposes
``make_bao_theory_fn`` — a closure over the fiducial cosmology that follows the same
pattern as ``make_pk_ell_fn`` and can be added to a full-shape Fisher matrix for a
joint analysis.

```{eval-rst}
.. automodule:: jaxptpolypol.bao
   :members:
   :undoc-members:
   :show-inheritance:
```
