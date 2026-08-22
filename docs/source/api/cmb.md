# `cmb`

``candl``-based CMB likelihood adapters — the single CMB interface used by the rest of
the package. Loads a native ``candl`` (or wrapped ``clipy``) likelihood, builds the
``Dl`` theory callable it expects, and wraps both in packed-parameter closures that the
existing Fisher and sampler tooling can consume unchanged.

```{eval-rst}
.. automodule:: jaxptpolypol.cmb
   :members:
   :undoc-members:
   :show-inheritance:
```
