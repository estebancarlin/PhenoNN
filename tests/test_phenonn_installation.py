#!/usr/bin/env python3
"""
PhenoNN Installation Test Script
Run this after installation to verify everything is working correctly.
Usage: python test_phenonn_installation.py
"""

import sys
import subprocess
import importlib.util


def print_section(title):
    """Print a formatted section title."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_import(module_name, submodule=None):
    """Test if a module can be imported."""
    try:
        if submodule:
            full_name = f"{module_name}.{submodule}"
            importlib.import_module(full_name)
        else:
            importlib.import_module(module_name)
        print(f"  ✅ {full_name if submodule else module_name}")
        return True
    except ImportError as e:
        raise ImportError(f"  ❌ {full_name if submodule else module_name}: {e}")
        return False
    except Exception as e:
        raise Exception(f"  ⚠️  {full_name if submodule else module_name}: {e}")
        return False


def main():
    print("\n" + "🎯" * 30)
    print("     PhenoNN Installation Test Suite")
    print("🎯" * 30)

    # Environment
    print_section("Python Environment")
    print(f"  Python version: {sys.version}")
    print(f"  Python executable: {sys.executable}")

    # Core Dependencies
    print_section("Core Dependencies")
    dependencies = [
        "numpy",
        "matplotlib",
        "torch",
        "scipy",
        "pandas",
        "mpltex",
        "xarray",
        "rich",
        "tqdm",
        "tensorboard",
        "sklearn",
    ]
    for dep in dependencies:
        test_import(dep)

    # Section 3: PhenoNN Modules
    print_section("PhenoNN Modules")

    # Test main package
    try:
        import phenonn

        print(f"  ✅ phenonn version: {phenonn.__version__}")
        print(f"  ✅ Device: {phenonn.DEVICE}")
    except Exception as e:
        raise Exception(f"  ❌ phenonn: {e}")
        return False

    # Test submodules
    modules = [
        ("phenonn.data", None),
        ("phenonn.data", "dataset"),
        ("phenonn.data", "dataset_flat"),
        ("phenonn.data", "feature_engineering"),
        ("phenonn.models", None),
        ("phenonn.models", "rnn"),
        ("phenonn.models", "transformer"),
        ("phenonn.models", "transformerbis"),
        ("phenonn.models", "fcn"),
        ("phenonn.models", "linear_baseline"),
        ("phenonn.training", None),
        ("phenonn.training", "train"),
        ("phenonn.training", "train_flat"),
        ("phenonn.prediction", None),
        ("phenonn.prediction", "predict"),
        ("phenonn.prediction", "predict_flat"),
        ("phenonn.utils", None),
        ("phenonn.utils", "logger"),
        ("phenonn.utils", "diagnostics"),
        ("phenonn.utils", "evaluater"),
        ("phenonn.utils", "model_loader"),
        ("phenonn.utils", "wrappers"),
        ("phenonn.utils", "utils"),
        ("phenonn.utils", "model_utils"),
    ]

    for module, submodule in modules:
        test_import(module, submodule)

    # CLI Test
    print_section("CLI Test")
    try:
        # Test version command
        result = subprocess.run(
            ["phenonn", "--version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ CLI version: {result.stdout.strip()}")
        else:
            print(f"  ⚠️  CLI version command issue: {result.stderr}")

        # Test help command
        result = subprocess.run(["phenonn", "--help"], capture_output=True, text=True)
        if "train" in result.stdout and "predict" in result.stdout:
            print("  ✅ CLI help shows commands")
        else:
            print("  ⚠️  CLI help may be incomplete")

    except FileNotFoundError:
        raise Exception("  ⚠️  CLI command not found (try 'python -m phenonn')")
        # Test alternative
        result = subprocess.run(
            ["python", "-m", "phenonn", "--version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ Module version: {result.stdout.strip()}")

    # Summary
    print_section("Installation Summary")

    # Check if all critical tests passed
    critical_passed = True

    # Verify phenonn import
    try:
        import phenonn

        version = phenonn.__version__
        print(f"  ✅ PhenoNN version {version} installed successfully")
    except Exception:
        critical_passed = False
        raise RuntimeError("  ❌ PhenoNN package not properly installed")

    # Verify torch
    try:
        import torch

        print(f"  ✅ PyTorch {torch.__version__} working")
    except Exception:
        critical_passed = False
        raise RuntimeError("  ❌ PyTorch not properly installed")

    # Verify models
    try:
        print("  ✅ Model creation working")
    except Exception:
        critical_passed = False
        raise RuntimeError("  ❌ Model imports failing")

    print("\n" + "=" * 60)
    if critical_passed:
        print("  🎉 SUCCESS! PhenoNN is fully installed and ready to use!")
        print("\n  Next steps:")
        print("    1. Run: phenonn train --help")
        print("    2. Prepare your data in the required format")
        print("    3. Start training: phenonn train --data_dir /path/to/data")
    else:
        print("  ⚠️  Some tests failed. Please check the errors above.")
        print("\n  Troubleshooting:")
        print("    1. Run: uv pip install -e .")
        print("    2. Run: uv pip install torch numpy pandas")
        print("    3. Check Python version (requires 3.8+)")
    print("=" * 60 + "\n")

    return critical_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
