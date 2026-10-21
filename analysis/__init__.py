# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Post-training analyses for PhenoNN.

Sub-packages and studies (all run as
``python -m analysis.<study>.<script>``):

- ``diagnostics``: one-shot data/model diagnostics and baseline CSVs.
- ``feature_ablation``: leave-one-feature-out and grouped channel ablations.
- ``perturbation``: ceteris-paribus weather perturbation scenarios.
- ``pft_linearity``: linearity of a no-mixing model in PFT fractions.
- ``year_generalization``: chronological, random, and cold-to-hot year splits.
- ``memory_analysis``: experimental protocol only; no runnable scripts yet.

``analysis/_checkpoint_utils`` holds the shared checkpoint helpers and
``analysis/_cluster.sh`` holds the shared cluster and environment helpers.

Nothing is re-exported here. The scripts share entry-point names (main /
parse_args) and pull heavy optional dependencies, so eager aggregation is
avoided.
"""

__all__ = []
