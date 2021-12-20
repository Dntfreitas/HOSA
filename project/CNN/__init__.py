"""This module demonstrates basic Sphinx usage with Python modules.

This is a test
==============

.. autosummary::
    :toctree: _autosummary

    cnn_models.BaseCNN
    cnn_models.CNNClassification
    cnn_models.CNNRegression
"""

from .cnn_models import BaseCNN, CNNClassification, CNNRegression
