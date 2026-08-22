# `plotting`

Fisher-forecast plotting: ``plot_contours`` draws the 1σ and 2σ ellipses of a 2-d
covariance block, ``plot_Gaussian`` the corresponding 1-d marginal, and ``triangle_plot``
assembles both into a corner figure. ``plot_contours`` emits two ellipses per call, so
deduplicate legend entries when overlaying several cases.

```{eval-rst}
.. automodule:: jaxptpolypol.plotting
   :members:
   :undoc-members:
   :show-inheritance:
```
