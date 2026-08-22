# `marginal_taylor`

Taylor surrogate of the per-bin marginal templates, built once about the ``theta_NL``
fiducial so that each subsequent posterior evaluation is a few dense tensor contractions
instead of a full re-trace of the ``ps_1loop_jax`` graph. ``m0`` is carried to second order
and ``M`` to first — the latter is what keeps the ``ln det`` tilt — and the whole build is
forward-over-forward and column-chunked so no reverse-mode tape is ever materialised.

```{eval-rst}
.. automodule:: jaxptpolypol.marginal_taylor
   :members:
   :undoc-members:
   :show-inheritance:
```
