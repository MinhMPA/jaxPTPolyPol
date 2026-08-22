# `derived`

Derived-parameter maps used to project a Fisher matrix or a chain out of the native
sampled basis into ``(Omega_m, H0, sigma8)``. ``sigmaR_from_linear_pk`` integrates the
emulator's linear ``P(k)`` against a spherical top-hat window, and
``make_derived_projection_fn`` builds the Jacobian-based projection consumed by
{doc}`inference`.

```{eval-rst}
.. automodule:: jaxptpolypol.derived
   :members:
   :undoc-members:
   :show-inheritance:
```
