#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface for PhenoNN.

This module provides command-line access to all PhenoNN functionality:
- Predict GCC using trained models
- Train new LSTM models
- Run hyperparameter tuning

Examples:
    phenonn predict GR 4 /path/to/example
    phenonn train full GR 8 gcc_lowess 4
    phenonn hp-tuning full GR 8 gcc_lowess 4
"""

import argparse
import sys
import os

# Import from phenonn package
from .lstm_pred import run_lstm_pred
from .lstm_train import run_lstm_train
from .lstm_hp_tuning import run_lstm_hp_tuning
from . import __version__


def main_predict(args):
    """Run predictions using trained models."""
    print("=" * 60)
    print("PhenoNN - GCC Prediction")
    print("=" * 60)
    print(f"PFT: {args.pft}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model type: m{args.m}")
    print(f"Data path: {args.input_path}")
    print("-" * 60)

    # Validate input path
    if not os.path.exists(args.input_path):
        print(f"Error: Input path '{args.input_path}' does not exist.")
        print("Please provide a valid path containing:")
        print("  - testdata/ directory with climate data")
        print("  - lstm_models/ directory with trained models")
        print("  - gcc_rcc_mins_site_veg.csv file")
        sys.exit(1)

    # Check for required files/directories
    required_items = [
        os.path.join(args.input_path, "testdata"),
        os.path.join(args.input_path, "lstm_models"),
        os.path.join(args.input_path, "gcc_rcc_mins_site_veg.csv"),
    ]

    for item in required_items:
        if not os.path.exists(item):
            print(f"Warning: {item} not found. Predictions may fail.")

    try:
        # Run prediction
        run_lstm_pred(
            m=args.m,
            pft=args.pft,
            batch_size=args.batch_size,
            input_path=args.input_path,
        )
        print("-" * 60)
        print("✓ Prediction completed successfully!")
        print(f"Output saved to: gcc_pred_test_{args.pft}_m{args.m}.csv")

    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        sys.exit(1)

    print("=" * 60)


def main_train(args):
    """Train new LSTM models."""
    print("=" * 60)
    print("PhenoNN - Model Training")
    print("=" * 60)
    print(f"PFT: {args.pft}")
    print(f"Model type: m{args.m}")
    print(f"Features: {args.nr_features}")
    print(f"Target: {args.target}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Dropout: {args.dropout}")
    print(f"Patience: {args.patience}")
    print(f"Max epochs: {args.epochs}")
    print("-" * 60)

    try:
        # Run training
        run_lstm_train(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            dropout_rate=args.dropout,
            patience=args.patience,
            m=args.m,
            pft=args.pft,
            nr_features=args.nr_features,
            target=args.target,
        )
        print("-" * 60)
        print("✓ Training completed successfully!")
        print(
            f"Models saved to: lstm_models/m{args.m}_{args.pft}_{args.nr_features}f_*"
        )

    except Exception as e:
        print(f"✗ Error during training: {e}")
        sys.exit(1)

    print("=" * 60)


def main_hp_tuning(args):
    """Run hyperparameter tuning."""
    print("=" * 60)
    print("PhenoNN - Hyperparameter Tuning")
    print("=" * 60)
    print(f"PFT: {args.pft}")
    print(f"Model type: m{args.m}")
    print(f"Features: {args.nr_features}")
    print(f"Target: {args.target}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 60)

    try:
        # Run hyperparameter tuning
        run_lstm_hp_tuning(
            m=args.m,
            pft=args.pft,
            nr_features=args.nr_features,
            target=args.target,
            batch_size=args.batch_size,
        )
        print("-" * 60)
        print("✓ Hyperparameter tuning completed successfully!")
        print("Results saved to: lstm_outputs/results_lstm_hp_tuning_fixeddataset.csv")

    except Exception as e:
        print(f"✗ Error during hyperparameter tuning: {e}")
        sys.exit(1)

    print("=" * 60)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="PhenoNN - Deep learning for phenology prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Predict GCC using trained models
  phenonn predict GR 4 /path/to/example

  # Train a new model
  phenonn train full GR 8 gcc_lowess 4 --epochs 150 --learning-rate 0.01

  # Run hyperparameter tuning
  phenonn hp-tuning full GR 8 gcc_lowess 4

For more information, visit: https://github.com/kardaneh/PhenoNN
Version: {__version__}
        """,
    )

    # Add version argument
    parser.add_argument("--version", action="version", version=f"PhenoNN {__version__}")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute", required=True
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict GCC using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  phenonn predict GR 4 /path/to/example
  phenonn predict DB 8 /path/to/example --m 0
        """,
    )
    predict_parser.add_argument(
        "pft",
        choices=["DB", "EN", "GR"],
        help="Plant functional type (DB=Deciduous Broadleaf, EN=Evergreen Needleleaf, GR=Grassland)",
    )
    predict_parser.add_argument(
        "batch_size", type=int, help="Batch size (recommended: DB=8, EN=6, GR=4)"
    )
    predict_parser.add_argument(
        "input_path",
        help="Path to input data directory (must contain testdata/, lstm_models/, and gcc_rcc_mins_site_veg.csv)",
    )
    predict_parser.add_argument(
        "--m",
        default="full",
        help='Model type: "full" for mfull models or block size for m0 models (default: full)',
    )
    predict_parser.set_defaults(func=main_predict)

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train new LSTM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  phenonn train full GR 8 gcc_lowess 4 --epochs 150 --learning-rate 0.01
  phenonn train 0 DB 8 gcc_lowess 8 --dropout 0.2
        """,
    )
    train_parser.add_argument(
        "m", help='Block size for shuffling or "full" for no shuffling'
    )
    train_parser.add_argument(
        "pft", choices=["DB", "EN", "GR"], help="Plant functional type to train"
    )
    train_parser.add_argument(
        "nr_features",
        type=int,
        choices=[6, 8, 9, 14],
        help="Number of features (6=dynamic, 8=dynamic+static, 9=dynamic+snow+static, 14=all)",
    )
    train_parser.add_argument(
        "target",
        choices=[
            "gcc",
            "rcc",
            "gcc_lowess",
            "rcc_lowess",
            "gcc_norm",
            "rcc_norm",
            "gcc_lowess_norm",
            "rcc_lowess_norm",
        ],
        help="Target variable to predict",
    )
    train_parser.add_argument(
        "batch_size", type=int, help="Batch size (recommended: DB=8, EN=6, GR=4)"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Maximum number of training epochs (default: 150)",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    train_parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="LSTM hidden layer size (default: 64)",
    )
    train_parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate (default: 0.0)"
    )
    train_parser.add_argument(
        "--patience", type=int, default=30, help="Early stopping patience (default: 30)"
    )
    train_parser.add_argument(
        "--plot", action="store_true", help="Plot learning curves"
    )
    train_parser.set_defaults(func=main_train)

    # Hyperparameter tuning subcommand
    hp_parser = subparsers.add_parser(
        "hp-tuning",
        help="Run hyperparameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  phenonn hp-tuning full GR 8 gcc_lowess 4
        """,
    )
    hp_parser.add_argument(
        "m", help='Block size for shuffling or "full" for no shuffling'
    )
    hp_parser.add_argument(
        "pft", choices=["DB", "EN", "GR"], help="Plant functional type"
    )
    hp_parser.add_argument(
        "nr_features", type=int, choices=[6, 8, 9, 14], help="Number of features to use"
    )
    hp_parser.add_argument(
        "target",
        choices=["gcc", "rcc", "gcc_lowess", "rcc_lowess"],
        help="Target variable to predict",
    )
    hp_parser.add_argument("batch_size", type=int, help="Batch size for training")
    hp_parser.set_defaults(func=main_hp_tuning)

    # Parse arguments
    args = parser.parse_args()

    # Execute the appropriate function
    args.func(args)


if __name__ == "__main__":
    main()
