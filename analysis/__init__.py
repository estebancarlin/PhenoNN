# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Post-training analyses for PhenoNN: diagnostics and feature ablation.

Sub-packages (all run as ``python -m analysis.<study>.<script>``):

- ``diagnostics`` — one-shot data/model diagnostics and baseline CSVs.
- ``feature_ablation`` — leave-one-feature-out and grouped channel ablations.

``analysis/_checkpoint_utils`` holds the shared checkpoint helpers.

Nothing is re-exported here — the scripts share entry-point names (main /
parse_args) and pull heavy optional deps, so eager aggregation is avoided.
"""

__all__ = []
