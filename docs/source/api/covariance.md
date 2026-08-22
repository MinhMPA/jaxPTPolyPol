# `covariance`

Gaussian covariance blocks on the survey grid: ``C_PP`` for the power-spectrum
multipoles ``(P0, P2, P4)`` following Chudaykin et al. (2019) Eqs. (B.1)–(B.2), and
``C_BB`` for the bispectrum monopole following Eq. (B.3) with the monopole angular
average carried out analytically. Single-bin and multi-bin assembly helpers produce a
joint ``[P..., B...]`` layout that can accommodate a non-zero ``C_PB`` block.

```{eval-rst}
.. automodule:: jaxptpolypol.covariance
   :members:
   :undoc-members:
   :show-inheritance:
```
