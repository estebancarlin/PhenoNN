# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
PhenoNN Command Line Interface

Usage examples:
    # Train LSTM model (per-site CSV format)
    phenonn train --data_dir ./data/DB/ --type lstm --hidden_size 128

    # Train on flat CSVs
    phenonn train-flat --features_csv features.csv --target_csv targets.csv

    # Train big model with year-based split
    phenonn train-big --data_dir ./data/DB/ --type transformer --split_mode year

    # Predict (per-site CSV format)
    phenonn predict --checkpoint ./runs/exp01/checkpoints/best_model.pth --data_dir ./data/DB/

    # Predict on flat CSVs
    phenonn predict-flat --checkpoint ./runs/exp_flat/checkpoints/best_model.pth

    # Show version
    phenonn --version
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
        print("  predict       Run predictions on per-site CSVs")
        print("  predict-flat  Run predictions on flat CSVs")
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

    elif command == "predict":
        from phenonn.prediction.predict import run_prediction

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

    elif command in ["--help", "-h"]:
        print(f"PhenoNN version {__version__}")
        print("Usage: phenonn <command> [options]")
        print("\nCommands:")
        print("  train         Train model on per-site CSVs")
        print("  train-flat    Train model on flat CSVs")
        print("  train-big     Train big model with year-based split")
        print("  predict       Run predictions on per-site CSVs")
        print("  predict-flat  Run predictions on flat CSVs")
        print("\nRun 'phenonn <command> --help' for command-specific options")

    else:
        print(f"Unknown command: {command}")
        print("Run 'phenonn --help' for usage")
        sys.exit(1)


if __name__ == "__main__":
    main()
