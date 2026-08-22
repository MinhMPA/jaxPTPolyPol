# `inference`

The Fisher layer: ``fisher_matrix`` from a covariance and a Jacobian, Schur-complement
marginalization over nuisance blocks, diagonal Gaussian prior Fishers with named-parameter
index resolution, Gaussian log-likelihoods, and projection of a Fisher matrix into a
derived-parameter basis. The same helpers serve both Fisher forecasts and MCMC workflows.

```{eval-rst}
.. automodule:: jaxptpolypol.inference
   :members:
   :undoc-members:
   :show-inheritance:
```

## Multi-bin index bookkeeping

``fixed_and_varied_indices`` is not in the module's ``__all__``, so ``automodule`` above
does not pick it up — but {doc}`../usage` uses it in the multi-bin walkthrough, and
:func:`~jaxptpolypol.inference.marginalize_fisher` expects exactly the ``varied_idx`` it
produces. It is documented here explicitly.

```{eval-rst}
.. autofunction:: jaxptpolypol.inference.fixed_and_varied_indices
```
