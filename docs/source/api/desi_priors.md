# `desi_priors`

Machinery for the DESI DR1-reanalysis EFT prior specification (arXiv:2511.20757,
Table I). Layer-1 values are stored both verbatim and mapped into this codebase's
coefficient convention, and loading fails unless the two reconcile; Layer-2 applies the
``theta_NL``-dependent ``A_AP * A_amp`` rescaling at runtime, including the exact
``f``-dependent rotation of the ``c0/c2/c4`` counterterm priors from the CLASS-PT
per-multipole basis into the ``mu``-space tilde basis. The convention map itself lives in
``docs/design/desi-convention-map.md``.

```{eval-rst}
.. automodule:: jaxptpolypol.desi_priors
   :members:
   :undoc-members:
   :show-inheritance:
```
