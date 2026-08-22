# `cmb_mcmc_utils`

Shared configuration and helpers for the CMB + BAO + BBN MCMC runs in ``example/mcmc``.

This page documents the module's **functions**: the run-artifact save/load helpers, the
differentiable ``100theta -> H0`` solve that maps the sampled basis onto the native
``candl`` cosmology, the conversions from a sampled vector to native ``CosmoParams``
arrays, and a chunked map for evaluating a derived quantity over a long chain.

The module also carries the run **registry** as module-level constants —
``COMBINATION_CONFIGS`` (which ``candl`` terms each named probe combination uses, whether
BAO is included, which cosmological parameters are sampled), ``DEFAULT_FIDUCIAL_NATIVE``
(the Planck 2018 expansion point), and the plotting order, colours, and published
reference points. Autodoc renders module-level data only when the assignment
carries an attribute docstring -- a ``#:`` comment above it or a string literal below it.
``:undoc-members:`` does not override that, and ``__all__`` has no bearing on it either.
These constants are plain assignments, so they are not reproduced below; read them from
the source, linked from each function via ``[source]``.

```{eval-rst}
.. automodule:: jaxptpolypol.cmb_mcmc_utils
   :members:
   :undoc-members:
   :show-inheritance:
```
