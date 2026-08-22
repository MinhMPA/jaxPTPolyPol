# `theory`

Factory closures for the differentiable theory vector: ``make_pk_ell_fn``,
``make_bk0_fn``, ``make_joint_pk_bk_fn``, bispectrum triangle construction, and the
fiducial AP distances. All static configuration — emulators, fiducial cosmology, AP on/off,
single- vs multi-bin — is captured in the closure, so the returned function is directly
``jit``/``grad``-able in the parameters with no ``static_argnames``.

```{eval-rst}
.. automodule:: jaxptpolypol.theory
   :members:
   :undoc-members:
   :show-inheritance:
```
