# `marginalization`

Marginalization and projection helpers for Fisher corner plots. ``ParameterSpec``
declares a plotted quantity (native or derived, with its label), and
``project_case_to_specs`` marginalizes a full Fisher matrix down to that set — including
the nonlinear maps for ``H0``, ``Omega_m``, and ``sigma8`` — returning a
``ProjectedFisherResult`` ready for {doc}`plotting`.

```{eval-rst}
.. automodule:: jaxptpolypol.marginalization
   :members:
   :undoc-members:
   :show-inheritance:
```
