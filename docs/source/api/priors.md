# `priors`

Loading and resolution of the packaged Gaussian prior specifications. The YAML specs
store prior metadata together with the survey/nuisance priors in the role-aware
``shared`` / ``pk`` / ``bk`` taxonomy of ``FullShapeSurveyParams``; these helpers are thin
adapters onto the index plumbing in {doc}`inference`, emitting ``{packed_index: sigma}``
for Fisher forecasts and ``[(index_in_varied, mean, sigma), ...]`` for samplers.

```{eval-rst}
.. automodule:: jaxptpolypol.priors
   :members:
   :undoc-members:
   :show-inheritance:
```
