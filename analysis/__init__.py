# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Post-training analyses for PhenoNN: diagnostics, baselines, ablations,
perturbations and generalization studies.

Layout: ``analysis/diagnostics/`` holds standalone one-shot scripts
(``python analysis/diagnostics/<script>.py``); each study directory holds
a larger multi-step workflow runnable with
``python -m analysis.<study>.<script>`` (see its README).

Nothing is re-exported here — the scripts share entry-point names (main /
parse_args) and pull heavy optional deps, so eager aggregation is avoided.
"""

__all__ = []
