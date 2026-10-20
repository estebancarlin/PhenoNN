"""
Shared naming / feature sets for the feature-ablation study.

Keeping the feature list, the ablation groups and the experiment names here
means ``gen_jobs.py`` (which writes the jobs), the drivers and
``aggregate.py`` (which reads the results back) can never drift.
"""

import runpy
from pathlib import Path

# Load config.py directly so job generation / aggregation naming does not import
# the phenonn package (which eagerly imports torch). The file is pure stdlib.
_CONFIG = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "phenonn" / "utils" / "config.py")
)
ALL_FEATURES = _CONFIG["ALL_FEATURES"]
PFT_START = _CONFIG["PFT_START"]

# Canonical order is [dynamic, cyclic, co2, pft]; PFT_START is where the PFT
# block begins, so ALL_FEATURES[:PFT_START] is exactly the non-PFT set.
NON_PFT_FEATURES = list(ALL_FEATURES[:PFT_START])

# Experiment folder names under --output_dir (runs/<exp>/checkpoints/).
BASELINE_EXP = "abl_baseline"

# Combined ablation shared by the group sweep and the model/size sweep.
COMBO_FEATURES = ["tp_sum", "ssrd_sum", "strd_sum", "VPD_max", "Tmean"]

# ── Ablation groups (name, feature list). Related meteo features go together. ──
_RAW_GROUPS = [
    ("baseline", []),
    ("temperature", ["Tmin", "Tmax", "Tmean"]),
    ("radiation", ["ssrd_sum", "strd_sum"]),  # the two downward fluxes
    ("net_radiation", ["Rn_tot"]),
    ("vpd", ["VPD_max", "VPD_mean"]),
    ("precip", ["tp_sum"]),
    ("smi", ["SMI"]),
    ("pet", ["PET"]),
    ("daylength", ["daylength"]),
    ("co2", ["co2"]),
    # Combined ablation (removed together): precip + both radiation fluxes +
    # VPD_max + Tmean. Runs on the same GPU as co2 (last slice).
    ("combo_rad_tp_vpdmax_tmean", COMBO_FEATURES),
]
# Keep only features present in this build; drop empty non-baseline groups.
GROUPS = [(n, [f for f in fs if f in ALL_FEATURES]) for n, fs in _RAW_GROUPS]
GROUPS = [(n, fs) for n, fs in GROUPS if n == "baseline" or fs]

# (model, hidden_size, stress_dim|None) for the fixed-ablation model sweep.
COMBO_CONFIGS = [
    ("lstm", 256, None),
    ("lstm", 512, None),
    ("bitransformer_v2", 256, 1),
    ("bitransformer_v2", 512, 1),
]


def exp_name(feature: str) -> str:
    """Experiment name for the per-feature run that ablates `feature`."""
    return f"abl_{feature}"


def group_exp_name(model: str, group: str) -> str:
    """Experiment name for one group in the in-process sweep."""
    return f"ablate_{model}_{group}"


def combo_suffix(model: str, hidden_size: int, stress_dim=None) -> str:
    """Stable suffix shared by a combo run and its baseline."""
    suffix = f"{model}_h{hidden_size}"
    return suffix + (f"_s{stress_dim}" if stress_dim is not None else "")


def combo_exp_name(model: str, hidden_size: int, stress_dim=None) -> str:
    """Experiment name for the ablated member of a combo config."""
    return f"ablate_combo_{combo_suffix(model, hidden_size, stress_dim)}"


def combo_baseline_exp_name(model: str, hidden_size: int, stress_dim=None) -> str:
    """Experiment name for the no-ablation baseline of a combo config."""
    return f"{combo_exp_name(model, hidden_size, stress_dim)}_baseline"
