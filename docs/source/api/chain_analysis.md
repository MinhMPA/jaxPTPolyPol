# `chain_analysis`

Convergence diagnostics, summaries, and chain plots for the packed parameter vectors
the samplers emit. ``PackedParameterSpec`` is the central abstraction: it maps a flat
packed vector (full-shape ``Pk``/``Bk`` plus nuisances plus cosmology, or ``BAO`` plus
cosmology) onto named scalar components suitable for ArviZ conversion, effective-sample-size
tables, trace plots, and corner plots.

```{eval-rst}
.. automodule:: jaxptpolypol.chain_analysis
   :members:
   :undoc-members:
   :show-inheritance:
```
