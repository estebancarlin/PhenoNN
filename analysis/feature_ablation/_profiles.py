"""
Hyperparameter profiles for the feature-ablation study.

One source of truth for the full (paper) runs and the short smoke runs. Values
are ``train_full_ram`` flag names; ``profile_args`` turns a profile into a CLI
argument list. Edit the profiles here, not in the generated jobs.

``paper`` standardizes on aelstm (our best performer); pass ``--model lstm`` to
the drivers when you want the LSTM baseline instead.
"""

from pathlib import Path

PAPER = {
    "type": "aelstm",
    "hidden_size": 256,
    "num_layers": 2,
    "train_years": "1992-2009",
    "val_years": "2010-2019",
    "val_fraction_of_grid": 100,
    "subset": 0.5,
    "n_sites_per_epoch": 250000,
    "n_years_per_epoch": 30,
    "n_val_sites": 50000,
    "num_epochs": 300,
    "patience": 7,
    "loss_type": "mse",
    "pft_mixing": True,
    "pft_meteo_only": True,
    "batch_size": 64,
    "amp": True,
    "num_workers": 2,
}

SMOKE = {
    "type": "lstm",
    "hidden_size": 32,
    "num_layers": 1,
    "train_years": "2008-2009",
    "val_years": "2010",
    "val_fraction_of_grid": 100,
    "n_sites_per_epoch": 5,
    "n_years_per_epoch": 1,
    "n_val_sites": 5,
    "num_epochs": 1,
    "patience": 1,
    "loss_type": "mse",
    "pft_mixing": True,
    "pft_meteo_only": True,
    "batch_size": 1,
    "num_workers": 0,
}

PROFILES = {"paper": PAPER, "smoke": SMOKE}

# Default output roots keep smoke checkpoints away from real runs.
OUTPUT_DIRS = {
    "paper": "runs/feature_ablation",
    "smoke": "runs/feature_ablation_smoke",
}

# None means "all non-PFT features"; smoke stays short on purpose.
FEATURES = {
    "paper": None,
    "smoke": ("Tmean", "tp_sum"),
}

# Dataset inputs relative to $DATA. Keep this list in one place: generated
# jobs and the smoke driver both need the exact same paths.
DATASET_FILES = {
    "features_dir": "era5_pixelset",
    "target_dir": "LAI_pixelset",
    "pft_dir": "PFT_pixelset",
    "selected_pixels": "selected_pixels005_1.nc",
    "parent_map": "selected_pixels01_1.nc",
    "stats_path": "norm_stats_1992_2019.json",
    "co2_path": "CO2_1700_2023_TRENDYv2024.txt",
}


def default_data_dir(project):
    """Resolve the dataset root, preferring the in-repo ``pixelset1`` link."""
    project = Path(project)
    for candidate in (project / "pixelset1", project.parent / "pixelset1"):
        if candidate.is_dir():
            return str(candidate)
    return str(project.parent / "pixelset1")


def profile_args(profile):
    """Return ``[(flag, value_or_True), ...]`` for a training profile."""
    return list(profile.items())


def cli_args(profile):
    """Render a profile as a list of command-line arguments (with values)."""
    out = []
    for key, value in profile.items():
        flag = f"--{key}"
        if value is True:
            out.append(flag)
        elif value is False or value is None:
            continue
        else:
            out.extend([flag, str(value)])
    return out


def data_cli_args(data):
    """Render the dataset paths under ``data`` as a CLI argument list."""
    root = Path(data)
    out = []
    for key, filename in DATASET_FILES.items():
        out.extend([f"--{key}", str(root / filename)])
    return out
