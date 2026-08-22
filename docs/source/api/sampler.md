# `sampler`

BlackJAX NUTS and random-walk Metropolis drivers, plus the whitening transforms they
run in. Sampled parameters are centered at zero and rescaled to approximately unit
variance using Fisher-derived (or user-supplied) scales, which is what makes the
high-dimensional geometry tractable; warmup, sampling (sequential or parallel chains),
and the transform back to physical units all live here.

```{eval-rst}
.. automodule:: jaxptpolypol.sampler
   :members:
   :undoc-members:
   :show-inheritance:
```
