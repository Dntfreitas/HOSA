API Reference
=============
The main statsmodels API is split into models:

* ``statsmodels.api``: Cross-sectional models and methods. Canonically imported
  using ``import statsmodels.api as sm``.
* ``statsmodels.tsa.api``: Time-series models and methods. Canonically imported
  using ``import statsmodels.tsa.api as tsa``.
* ``statsmodels.formula.api``: A convenience interface for specifying models
  using formula strings and DataFrames. This API directly exposes the ``from_formula``
  class method of models that support the formula API. Canonically imported using
  ``import statsmodels.formula.api as smf``

Regression
~~~~~~~~~~

.. automodule:: project.Callbacks

.. automodule:: project.CNN