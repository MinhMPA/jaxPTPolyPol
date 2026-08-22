# `params`

The parameter containers. ``CosmoParams`` and ``SurveyParams`` /
``FullShapeSurveyParams`` are registered JAX pytrees whose names and sizes are static
compilation constants while their values stay traced, plus the pack/unpack utilities that
flatten them into the single- and multi-bin packed vectors used everywhere for
differentiation (cosmology shared across bins, survey/EFT parameters per bin).

```{eval-rst}
.. automodule:: jaxptpolypol.params
   :members:
   :undoc-members:
   :show-inheritance:
```
