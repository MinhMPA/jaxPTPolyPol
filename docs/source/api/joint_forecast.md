# `joint_forecast`

Composes the PFS Taylor-surrogate marginal posterior with fiducial-centered Gaussian
external blocks — the ``candl`` CMB Fisher block and the BBN prior on ``omega_b`` — on an
extended sampled vector ``theta = concat(theta_NL, [tau])``. The PFS posterior sees only
its own leading slice; the external blocks address the full vector through index maps
built by ``embed_fisher``.

```{eval-rst}
.. automodule:: jaxptpolypol.joint_forecast
   :members:
   :undoc-members:
   :show-inheritance:
```
