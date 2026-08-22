# `cmb_mcmc_utils`

Shared configuration and helpers for the CMB + BAO + BBN MCMC runs in ``example/mcmc``.
Holds the registry of named probe combinations (which ``candl`` terms, whether BAO is
included, which cosmological parameters are sampled), the Planck 2018 fiducial point,
and the plotting metadata; alongside it, the run-artifact save/load helpers and the
differentiable ``100theta -> H0`` solve that maps the sampled basis onto the native
``candl`` cosmology.

```{eval-rst}
.. automodule:: jaxptpolypol.cmb_mcmc_utils
   :members:
   :undoc-members:
   :show-inheritance:
```
