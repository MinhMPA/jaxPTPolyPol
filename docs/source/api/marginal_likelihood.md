# `marginal_likelihood`

Analytic (Gaussian) marginalization over the EFT and stochastic parameters that enter the
theory vector linearly, following arXiv:2511.20757 §II.3. For
``t(theta_NL, theta_lin) = m0(theta_NL) + M(theta_NL) theta_lin`` the prior integral over
``theta_lin`` is closed form, leaving a ``-2 ln L`` with a residual term, a Schur term, and a
``ln det(A Sigma_p)`` tilt. This module builds the templates ``(m0, M)`` and the resulting
marginal log-posterior.

```{eval-rst}
.. automodule:: jaxptpolypol.marginal_likelihood
   :members:
   :undoc-members:
   :show-inheritance:
```
