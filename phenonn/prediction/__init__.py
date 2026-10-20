# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
PhenoNN prediction package.

Public API
----------
- :func:`main`            — CLI entry point; invoked as
                             ``python -m phenonn.prediction.predict`` or via
                             ``phenonn predict``.
- :func:`_parse_year_spec` — parse ``"2010-2019"`` / ``"2010,2015"`` / ``"all"``
                             into a list of years.
- :func:`_resolve_sites`  — resolve the ``--predict_sites`` / ``--sites`` /
                             ``--selected_pixels`` CLI flags against a checkpoint
                             to a concrete list of site IDs. Public-facing
                             despite the leading underscore, because the
                             analysis modules (perturbation, pft_linearity)
                             reuse it.

The legacy flat-CSV predictor (``predict_flat``) has been moved to
``legacy/prediction/``.
"""

from .predict import (
    main,
    _parse_year_spec,
    _resolve_sites,
)

__all__ = [
    "main",
    "_parse_year_spec",
    "_resolve_sites",
]
