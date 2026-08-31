# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""Dispatch PhenoNN training, prediction, and evaluation commands.

Run ``phenonn --help`` for the command list and ``phenonn COMMAND --help`` for
the authoritative options owned by each command parser.
"""

import sys
from .version import __version__


def main():
    # Handle version request
    if "--version" in sys.argv or "-V" in sys.argv:
        print(f"phenonn {__version__}")
        return

    # Handle no arguments
    if len(sys.argv) == 1:
        print(f"PhenoNN version {__version__}")
        print("Usage: phenonn <command> [options]")
        print("\nCommands:")
        print("  train         Train model on per-site CSVs")
        print("  train-big     Train big model with year-based split")
        print("  train-flat    Train model on flat CSVs")
        print("  train-global  Train model on selected-site NetCDF files")
        print("  predict       Run 0.1-degree pixelset inference")
        print("  predict-flat  Run predictions on flat CSVs")
        print("  evaluate-global  Evaluate a global NetCDF checkpoint")
        print("  evaluate-phenocam  Compare LAI predictions with Phenocam GCC")
        print("  compare-global  Compare global training runs and evaluations")
        print("  train-0p1     Train the 0.1-degree pixelset workflow")
        print("  predict-0p1   Predict from a 0.1-degree pixelset checkpoint")
        print("  train-xgb     Train the XGBoost baseline")
        print("  predict-xgb   Run the XGBoost baseline")
        print("\nRun 'phenonn <command> --help' for command-specific options")
        return

    # Get the command (first argument after script name)
    command = sys.argv[1]

    # Remove the 'phenonn' and command from argv, keep rest for subcommand
    # But preserve argv[0] as the script name for the subcommand
    subcommand_argv = [sys.argv[0]] + sys.argv[2:]

    # Dispatch to appropriate module, passing through all arguments
    if command == "train":
        from phenonn.training.train import run_training

        # Replace sys.argv for the subcommand
        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            run_training()
        finally:
            sys.argv = original_argv

    elif command == "train-flat":
        from phenonn.training.train_flat import run_training_flat

        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            run_training_flat()
        finally:
            sys.argv = original_argv

    elif command == "train-big":
        from phenonn.training.train_big import run_training_big

        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            run_training_big()
        finally:
            sys.argv = original_argv

    elif command == "train-global":
        from phenonn.training.train_global import run_training_global

        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            run_training_global()
        finally:
            sys.argv = original_argv

    elif command == "predict":
        from phenonn.prediction.predict import main as run_prediction

        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            run_prediction()
        finally:
            sys.argv = original_argv

    elif command == "predict-flat":
        from phenonn.prediction.predict_flat import run_prediction_flat

        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            run_prediction_flat()
        finally:
            sys.argv = original_argv

    elif command == "evaluate-global":
        from phenonn.prediction.evaluate_global import evaluate_global

        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            evaluate_global()
        finally:
            sys.argv = original_argv

    elif command == "evaluate-phenocam":
        from phenonn.prediction.evaluate_phenocam import evaluate_phenocam

        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            evaluate_phenocam()
        finally:
            sys.argv = original_argv

    elif command == "compare-global":
        from phenonn.prediction.compare_global import compare_global

        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            compare_global()
        finally:
            sys.argv = original_argv

    elif command == "train-0p1":
        from phenonn.training.train_full_ram import main as train_full_ram

        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            train_full_ram()
        finally:
            sys.argv = original_argv

    elif command == "predict-0p1":
        from phenonn.prediction.predict import main as predict_pixelset

        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            predict_pixelset()
        finally:
            sys.argv = original_argv

    elif command == "train-xgb":
        from phenonn.training.xgb_train import main as train_xgb

        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            train_xgb()
        finally:
            sys.argv = original_argv

    elif command == "predict-xgb":
        from phenonn.prediction.xgb_predict import main as predict_xgb

        original_argv = sys.argv
        sys.argv = subcommand_argv
        try:
            predict_xgb()
        finally:
            sys.argv = original_argv

    elif command in ["--help", "-h"]:
        print(f"PhenoNN version {__version__}")
        print("Usage: phenonn <command> [options]")
        print("\nCommands:")
        print("  train         Train model on per-site CSVs")
        print("  train-flat    Train model on flat CSVs")
        print("  train-big     Train big model with year-based split")
        print("  train-global  Train model on selected-site NetCDF files")
        print("  predict       Run 0.1-degree pixelset inference")
        print("  predict-flat  Run predictions on flat CSVs")
        print("  evaluate-global  Evaluate a global NetCDF checkpoint")
        print("  evaluate-phenocam  Compare LAI predictions with Phenocam GCC")
        print("  compare-global  Compare global training runs and evaluations")
        print("  train-0p1     Train the 0.1-degree pixelset workflow")
        print("  predict-0p1   Predict from a 0.1-degree pixelset checkpoint")
        print("  train-xgb     Train the XGBoost baseline")
        print("  predict-xgb   Run the XGBoost baseline")
        print("\nRun 'phenonn <command> --help' for command-specific options")

    else:
        print(f"Unknown command: {command}")
        print("Run 'phenonn --help' for usage")
        sys.exit(1)


if __name__ == "__main__":
    main()
