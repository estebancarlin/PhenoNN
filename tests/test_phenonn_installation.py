#!/usr/bin/env python3
"""
PhenoNN installation test.

Verifies that the package imports, exposes ``__version__`` / ``DEVICE``, and
that the core sub-modules of the current (0.1°-native) layout import cleanly.
Run with:  python -m tests.test_phenonn_installation
"""

import importlib
import sys


def _check(module_name):
    """Import a module, raising ImportError with a readable message on failure."""
    importlib.import_module(module_name)
    print(f"  OK  {module_name}")


def main():
    print("PhenoNN installation test")
    print("=" * 40)

    # Core dependencies
    for dep in ["numpy", "matplotlib", "torch", "scipy", "pandas", "xarray"]:
        _check(dep)

    # Main package + attributes
    import phenonn

    print(f"  OK  phenonn version: {phenonn.__version__}")
    print(f"  OK  device: {phenonn.DEVICE}")

    # Sub-modules of the current structure (torch-heavy but present in the
    # CI env; XGBoost / greedy baselines are skipped — xgboost is not a CI dep).
    modules = [
        "phenonn.utils.config",
        "phenonn.utils.utils",
        "phenonn.utils.logger",
        "phenonn.utils.loss",
        "phenonn.utils.wrappers",
        "phenonn.utils.model_factory",
        "phenonn.utils.diagnostics",
        "phenonn.data.lai_dataset",
        "phenonn.models",
        "phenonn.models.rnn",
        "phenonn.models.fcn",
        "phenonn.models.transformer",
        "phenonn.models.transformerbis",
        "phenonn.models.bitransformer",
        "phenonn.models.linear_baseline",
        "phenonn.training.train_full_ram",
        "phenonn.prediction.predict",
    ]
    for module in modules:
        _check(module)

    print("=" * 40)
    print("SUCCESS: PhenoNN imports cleanly.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
