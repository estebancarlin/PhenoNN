"""Fast unit tests for the feature-ablation workflow (no data / GPU / runs)."""

import csv
import io
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import _checkpoint_utils
from analysis.feature_ablation import (
    _common,
    _profiles,
    aggregate,
    gen_jobs,
    train_ablate,
)
from analysis.feature_ablation._hooks import ZeroChannels
from analysis.feature_ablation.run_ablate_inproc import (
    _is_done,
    _parse_indices,
    _specs,
    _split_features,
)
from phenonn.utils.config import ALL_FEATURES, FEATURE_CHANNELS, PFT_START

REPO = Path(__file__).resolve().parents[1]


class TestCommon(unittest.TestCase):
    def test_groups_are_non_pft(self):
        self.assertEqual(_common.GROUPS[0], ("baseline", []))
        for name, features in _common.GROUPS:
            with self.subTest(group=name):
                if name != "baseline":
                    self.assertTrue(features)
                for feature in features:
                    self.assertIn(feature, _common.NON_PFT_FEATURES)

    def test_names(self):
        self.assertEqual(_common.BASELINE_EXP, "abl_baseline")
        self.assertEqual(_common.exp_name("Tmean"), "abl_Tmean")
        self.assertEqual(
            _common.group_exp_name("aelstm", "radiation"),
            "ablate_aelstm_radiation",
        )
        self.assertEqual(
            _common.combo_baseline_exp_name("lstm", 256),
            "ablate_combo_lstm_h256_baseline",
        )
        self.assertEqual(
            _common.combo_exp_name("bitransformer_v2", 512, 1),
            "ablate_combo_bitransformer_v2_h512_s1",
        )


class TestProfiles(unittest.TestCase):
    def test_cli_args_renders_flags(self):
        args = _profiles.cli_args(_profiles.SMOKE)
        self.assertIn("--type", args)
        self.assertIn("lstm", args)
        self.assertIn("--pft_mixing", args)
        self.assertNotIn("--amp", args)
        self.assertEqual(args.count("--type"), 1)

    def test_profile_defaults(self):
        self.assertEqual(_profiles.FEATURES["smoke"], ("Tmean", "tp_sum"))
        self.assertIsNone(_profiles.FEATURES["paper"])
        self.assertNotEqual(
            _profiles.OUTPUT_DIRS["paper"], _profiles.OUTPUT_DIRS["smoke"]
        )

    def test_default_data_dir_prefers_repo_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "repo"
            repo_data = project / "pixelset1"
            parent_data = Path(tmp) / "pixelset1"
            repo_data.mkdir(parents=True)
            parent_data.mkdir()
            self.assertEqual(_profiles.default_data_dir(project), str(repo_data))
            repo_data.rmdir()
            self.assertEqual(_profiles.default_data_dir(project), str(parent_data))

    def test_data_cli_args_cover_all_inputs(self):
        args = _profiles.data_cli_args("/data/pixelset1")
        for flag, filename in _profiles.DATASET_FILES.items():
            self.assertIn(f"--{flag}", args)
            self.assertIn(f"/data/pixelset1/{filename}", args)


class TestSpecs(unittest.TestCase):
    def test_feature_specs(self):
        specs = _specs("features", "lstm", "Tmean,tp_sum")
        self.assertEqual(specs[0], ("baseline", [], _common.BASELINE_EXP))
        self.assertEqual(specs[1], ("Tmean", ["Tmean"], "abl_Tmean"))
        self.assertEqual(specs[2][2], "abl_tp_sum")

    def test_feature_specs_reject_pft(self):
        with self.assertRaises(SystemExit):
            _specs("features", "lstm", "pft1_frac")

    def test_group_specs_match_groups(self):
        specs = _specs("groups", "aelstm", "")
        self.assertEqual(len(specs), len(_common.GROUPS))
        self.assertEqual(specs[0][2], "ablate_aelstm_baseline")
        self.assertTrue(specs[-1][1])  # combo group

    def test_parse_indices(self):
        self.assertEqual(_parse_indices("0-2", 5), [0, 1, 2])
        self.assertEqual(_parse_indices("0,3", 5), [0, 3])
        self.assertEqual(_parse_indices("", 3), [0, 1, 2])
        with self.assertRaises(SystemExit):
            _parse_indices("9", 3)

    def test_split_features(self):
        self.assertEqual(_split_features(" Tmean, tp_sum "), ["Tmean", "tp_sum"])
        self.assertEqual(_split_features(""), [])


class TestHook(unittest.TestCase):
    def test_zeroes_selected_channel_only(self):
        hook = ZeroChannels([1])
        x = torch.ones(2, 28, 3)
        (out,) = hook(None, (x,))
        self.assertEqual(out[:, 1, :].sum().item(), 0.0)
        self.assertEqual(out[:, 0, :].sum().item(), 6.0)
        self.assertEqual(x[:, 1, :].sum().item(), 6.0)  # input untouched

    def test_pickle_roundtrip(self):
        hook = pickle.loads(pickle.dumps(ZeroChannels([2, 4])))
        x = torch.ones(1, 28, 2)
        (out,) = hook(None, (x,))
        self.assertEqual(out[:, 2, :].sum().item(), 0.0)
        self.assertEqual(out[:, 4, :].sum().item(), 0.0)

    def test_rejects_wrong_channel_count(self):
        hook = ZeroChannels([0])
        with self.assertRaises(ValueError):
            hook(None, (torch.ones(1, FEATURE_CHANNELS - 1, 3),))

    def test_rejects_out_of_range_channel(self):
        with self.assertRaises(ValueError):
            ZeroChannels([FEATURE_CHANNELS])


class TestTrainAblate(unittest.TestCase):
    def test_pop_ablate_feature_forms(self):
        argv = ["train", "--ablate_feature", "Tmean", "--epochs", "1"]
        self.assertEqual(train_ablate._pop_ablate_feature(argv), "Tmean")
        self.assertEqual(argv, ["train", "--epochs", "1"])

        argv = ["train", "--ablate_feature=tp_sum"]
        self.assertEqual(train_ablate._pop_ablate_feature(argv), "tp_sum")
        self.assertEqual(argv, ["train"])

    def test_pop_ablate_feature_requires_value(self):
        with self.assertRaises(SystemExit):
            train_ablate._pop_ablate_feature(["train", "--ablate_feature"])

    def test_resolve_channel(self):
        with redirect_stdout(io.StringIO()):
            self.assertIsNone(train_ablate._resolve_channel("none"))
            self.assertEqual(
                train_ablate._resolve_channel("Tmean"),
                ALL_FEATURES.index("Tmean"),
            )
            with self.assertRaises(SystemExit):
                train_ablate._resolve_channel("not_a_feature")
            with self.assertRaises(SystemExit):
                train_ablate._resolve_channel(ALL_FEATURES[PFT_START])


class TestCheckpointUtils(unittest.TestCase):
    def test_load_state_dict(self):
        self.assertEqual(
            _checkpoint_utils.load_state_dict({"state_dict": {"a": 1}}),
            {"a": 1},
        )
        self.assertEqual(
            _checkpoint_utils.load_state_dict({"model_state_dict": {"b": 2}}),
            {"b": 2},
        )
        self.assertEqual(_checkpoint_utils.load_state_dict({"state_dict": {}}), {})
        with self.assertRaises(KeyError):
            _checkpoint_utils.load_state_dict({"optimizer_state_dict": {}})

    def test_find_best_checkpoint_prefers_current_tar(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "checkpoints"
            checkpoints.mkdir()
            legacy = checkpoints / "best_model.pth"
            current = checkpoints / "run_best_model.pth.tar"
            legacy.write_text("legacy")
            current.write_text("current")
            self.assertEqual(
                _checkpoint_utils.find_best_checkpoint(tmp),
                current,
            )

    def test_find_best_checkpoint_ignores_full_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "best_model_full.pth").write_text("full")
            self.assertIsNone(_checkpoint_utils.find_best_checkpoint(tmp))

    def test_has_completed_run_requires_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "best_model.pth.tar").write_text("best")
            self.assertFalse(_checkpoint_utils.has_completed_run(tmp))
            (checkpoints / "final_model_epoch1.pth.tar").write_text("final")
            self.assertTrue(_checkpoint_utils.has_completed_run(tmp))


class TestInprocSweep(unittest.TestCase):
    def test_done_requires_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output_dir=tmp, experiment="run")
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "config.json").write_text("{}")
            self.assertFalse(_is_done(args))
            checkpoints = run / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "best_model.pth.tar").write_text("checkpoint")
            self.assertFalse(_is_done(args))
            (checkpoints / "final_model_epoch1.pth.tar").write_text("final")
            self.assertTrue(_is_done(args))


class TestAggregateMasks(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(aggregate.normalize_mask(None), None)
        self.assertEqual(aggregate.normalize_mask(""), ())
        self.assertEqual(aggregate.normalize_mask("tp_sum, Tmean"), ("Tmean", "tp_sum"))
        self.assertEqual(
            aggregate.normalize_mask(["tp_sum", "Tmean"]), ("Tmean", "tp_sum")
        )

    def test_match_ignores_order(self):
        self.assertTrue(aggregate.masks_match("a,b", "b,a"))
        self.assertFalse(aggregate.masks_match("a", "b"))
        self.assertTrue(aggregate.masks_match(None, None))
        self.assertFalse(aggregate.masks_match(None, ()))

    def test_validate(self):
        good = {"experiment": "x", "val_r2": 0.5, "mask": ("Tmean",), "has_mask": True}
        self.assertTrue(aggregate._validate(good, ("Tmean",), False, False))
        self.assertFalse(aggregate._validate(good, ("tp_sum",), False, False))
        with redirect_stdout(io.StringIO()):
            self.assertTrue(aggregate._validate(good, ("tp_sum",), False, True))

    def test_validate_rejects_non_finite_r2(self):
        bad = {
            "experiment": "x",
            "val_r2": float("nan"),
            "mask": ("Tmean",),
            "has_mask": True,
        }
        with redirect_stdout(io.StringIO()):
            self.assertFalse(aggregate._validate(bad, ("Tmean",), False, False))

    def test_read_run_skips_unreadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "run" / "checkpoints"
            checkpoints.mkdir(parents=True)
            (checkpoints / "best_model.pth.tar").write_text("broken")
            with mock.patch.object(
                aggregate.torch, "load", side_effect=RuntimeError("corrupt")
            ), redirect_stdout(io.StringIO()):
                self.assertIsNone(aggregate._read_run(tmp, "run"))

    def test_read_run_skips_non_mapping_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "run" / "checkpoints"
            checkpoints.mkdir(parents=True)
            (checkpoints / "best_model.pth.tar").write_text("junk")
            with mock.patch.object(
                aggregate.torch, "load", return_value=["not", "a", "dict"]
            ), redirect_stdout(io.StringIO()):
                self.assertIsNone(aggregate._read_run(tmp, "run"))

    def test_write_outputs_round_trips_comma_masks(self):
        rows = [
            {
                "label": "temperature",
                "experiment": "ablate_aelstm_temperature",
                "ablate_features": "Tmax,Tmean,Tmin",
                "val_r2": 0.5,
                "delta_r2": -0.1,
                "val_rmse": 0.2,
                "delta_rmse": 0.01,
                "best_epoch": 7,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            out = Path(tmp) / "feature_ablation.png"
            aggregate._write_outputs(rows, out, "test")
            with open(out.with_suffix(".csv"), newline="") as f:
                parsed = list(csv.reader(f))
        self.assertEqual(len(parsed), 2)
        self.assertEqual(len(parsed[0]), 8)
        self.assertEqual(parsed[1][2], "Tmax,Tmean,Tmin")

    def test_missing_baseline_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                aggregate._collect_features(tmp, "abl_baseline", [], False, False)

    def test_feature_and_group_inputs(self):
        feats = aggregate._feature_inputs("Tmean,tp_sum")
        self.assertEqual([f[0] for f in feats], ["Tmean", "tp_sum"])
        self.assertEqual(feats[0][2], ("Tmean",))
        groups = aggregate._group_inputs("aelstm", "temperature,co2")
        self.assertEqual([g[0] for g in groups], ["temperature", "co2"])
        self.assertEqual(
            aggregate.normalize_mask(groups[0][2]),
            ("Tmax", "Tmean", "Tmin"),
        )
        combos = aggregate._combo_inputs()
        self.assertEqual(len(combos), len(_common.COMBO_CONFIGS))
        self.assertTrue(combos[0][1].endswith("_h256"))

    def test_combo_uses_per_config_baseline(self):
        mask = tuple(_common.COMBO_FEATURES)
        runs = {
            "base1": {
                "experiment": "base1",
                "val_r2": 0.2,
                "val_rmse": 0.3,
                "best_epoch": 1,
                "mask": (),
                "has_mask": True,
            },
            "abl1": {
                "experiment": "abl1",
                "val_r2": 0.1,
                "val_rmse": 0.4,
                "best_epoch": 2,
                "mask": mask,
                "has_mask": True,
            },
        }
        inputs = [("c1", "abl1", "base1", mask)]
        with mock.patch.object(
            aggregate, "_read_run", side_effect=lambda _d, exp: runs[exp]
        ):
            rows, baselines = aggregate._collect_combo("unused", inputs, False, False)
        self.assertEqual(baselines["c1"][0], "base1")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["delta_r2"], -0.1)


class TestGenJobs(unittest.TestCase):
    def test_generate_smoke_jobs_is_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp) / "jobs"
            out = Path(tmp) / "runs"
            jobs.mkdir(parents=True)
            (jobs / "run_stale.sh").write_text("#!/bin/sh\n")
            argv = [
                "gen_jobs.py",
                "--profile",
                "smoke",
                "--study",
                "all",
                "--project",
                str(REPO),
                "--data",
                str(Path(tmp) / "data"),
                "--jobs_dir",
                str(jobs),
                "--output_dir",
                str(out),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                gen_jobs.main()

            names = sorted(p.name for p in jobs.glob("run_*.sh"))
            self.assertNotIn("run_stale.sh", names)
            self.assertIn("run_baseline.sh", names)
            self.assertIn("run_ablate_Tmean.sh", names)
            self.assertIn("run_ablate_tp_sum.sh", names)
            self.assertIn("run_combo.sh", names)
            self.assertEqual(len(list(jobs.glob("run_groups_*.sh"))), 4)
            self.assertTrue((out / "logs").is_dir())
            self.assertIn("mkdir -p", (jobs / "submit_all.sh").read_text())

            data_root = Path(tmp) / "data"
            for script in jobs.glob("run_*.sh"):
                text = script.read_text()
                for forbidden in (
                    "#SBATCH --partition",
                    "#SBATCH --account",
                    "#SBATCH --qos",
                    "#SBATCH --mem",
                ):
                    self.assertNotIn(forbidden, text)
                for flag, filename in _profiles.DATASET_FILES.items():
                    self.assertIn(
                        f"--{flag} {data_root / filename}",
                        text,
                        f"{script.name} is missing --{flag}",
                    )

            feature = (jobs / "run_ablate_Tmean.sh").read_text()
            self.assertIn("--ablate_feature Tmean", feature)
            self.assertIn("--experiment abl_Tmean", feature)
            self.assertIn(str(out), feature)
            self.assertIn("--type lstm", feature)

    def test_common_args_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp) / "jobs"
            argv = [
                "gen_jobs.py",
                "--profile",
                "paper",
                "--study",
                "features",
                "--features",
                "Tmean",
                "--common_args",
                "--loss_type mse --batch_size 7",
                "--jobs_dir",
                str(jobs),
                "--output_dir",
                str(Path(tmp) / "runs"),
                "--project",
                str(REPO),
                "--data",
                str(Path(tmp) / "data"),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                gen_jobs.main()
            text = (jobs / "run_ablate_Tmean.sh").read_text()
            self.assertIn("--loss_type mse --batch_size 7", text)
            self.assertNotIn("--type aelstm", text)
            self.assertIn(
                f"--selected_pixels {Path(tmp) / 'data' / 'selected_pixels005_1.nc'}",
                text,
            )

    def test_combo_skip_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp) / "jobs"
            argv = [
                "gen_jobs.py",
                "--profile",
                "smoke",
                "--study",
                "combo",
                "--skip-baseline",
                "--jobs_dir",
                str(jobs),
                "--output_dir",
                str(Path(tmp) / "runs"),
                "--project",
                str(REPO),
                "--data",
                str(Path(tmp) / "data"),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                gen_jobs.main()
            self.assertIn("--skip-baseline", (jobs / "run_combo.sh").read_text())
            self.assertFalse((jobs / "run_baseline.sh").exists())

    def test_submit_all_preflights_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp) / "jobs"
            argv = [
                "gen_jobs.py",
                "--profile",
                "smoke",
                "--study",
                "features",
                "--jobs_dir",
                str(jobs),
                "--output_dir",
                str(Path(tmp) / "runs"),
                "--project",
                str(REPO),
                "--data",
                str(Path(tmp) / "missing-data"),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                gen_jobs.main()
            env = dict(
                os.environ,
                PHENONN_SITE="hal",
                PHENONN_LOCAL="1",
                PHENONN_DRY_RUN="0",
            )
            proc = subprocess.run(
                ["bash", str(jobs / "submit_all.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("missing:", proc.stderr)

    def test_submit_all_dry_run_site_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = Path(tmp) / "jobs"
            argv = [
                "gen_jobs.py",
                "--profile",
                "smoke",
                "--study",
                "features",
                "--jobs_dir",
                str(jobs),
                "--output_dir",
                str(Path(tmp) / "runs"),
                "--project",
                str(REPO),
                "--data",
                str(Path(tmp) / "data"),
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                gen_jobs.main()
            (Path(tmp) / "runs" / "logs").rmdir()
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            fake_sbatch = bin_dir / "sbatch"
            fake_sbatch.write_text("#!/bin/sh\n")
            fake_sbatch.chmod(0o755)
            env = dict(
                os.environ,
                PATH=f"{bin_dir}:{os.environ.get('PATH', '')}",
                PHENONN_SITE="hal",
                PHENONN_DRY_RUN="1",
                PHENONN_LOCAL="0",
                PHENONN_DEPENDENCY="afterok:12345",
            )
            proc = subprocess.run(
                ["bash", str(jobs / "submit_all.sh")],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertIn("--partition=batch", proc.stdout)
            self.assertIn("--dependency=afterok:12345", proc.stdout)
            self.assertIn("run_baseline.sh", proc.stdout)
            self.assertTrue((Path(tmp) / "runs" / "logs").is_dir())


class TestClusterScript(unittest.TestCase):
    def test_bash_syntax(self):
        scripts = [
            REPO / "analysis" / "_cluster.sh",
            REPO / "analysis" / "feature_ablation" / "jobs" / "submit_ablate.sh",
            REPO / "analysis" / "feature_ablation" / "jobs" / "submit_combo.sh",
            REPO / "scripts" / "smoke_feature_ablation.sh",
        ]
        for script in scripts:
            with self.subTest(script=script.name):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_module_activation_cannot_abort_under_set_u(self):
        script = r"""
set -euo pipefail
module() {
    if [[ "${1:-}" == "load" ]]; then
        printf '%s\n' "$MAGPLUS_HOME" >/dev/null
    fi
}
source "$1/analysis/_cluster.sh"
export PHENONN_ROOT="$2"
export PHENONN_VENV=/dev/null
export PHENONN_SITE=hal
export PHENONN_MODULES=broken-python
phenonn_cluster_init
printf 'survived\n'
"""
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env.pop("MAGPLUS_HOME", None)
            env.pop("PHENONN_SKIP_MODULES", None)
            env.pop("PHENONN_LOCAL", None)
            env.pop("PHENONN_DRY_RUN", None)
            proc = subprocess.run(
                ["bash", "-c", script, "bash", str(REPO), tmp],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(proc.stdout.strip(), "survived")

    def test_no_modules_by_default(self):
        script = r"""
set -euo pipefail
module() { printf 'called\n' >> "$3"; }
source "$1/analysis/_cluster.sh"
export PHENONN_ROOT="$2"
export PHENONN_VENV=/dev/null
export PHENONN_SITE=hal
unset PHENONN_MODULES
phenonn_cluster_init
"""
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "module-called"
            proc = subprocess.run(
                ["bash", "-c", script, "bash", str(REPO), tmp, str(marker)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertFalse(marker.exists())

    def test_data_default_prefers_repo_pixelset1(self):
        script = r"""
set -euo pipefail
source "$1/analysis/_cluster.sh"
export PHENONN_ROOT="$2"
export PHENONN_VENV=/dev/null
export PHENONN_SITE=hal
unset DATA
phenonn_cluster_init
printf '%s' "$DATA"
"""
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "repo"
            data = project / "pixelset1"
            data.mkdir(parents=True)
            proc = subprocess.run(
                ["bash", "-c", script, "bash", str(REPO), str(project)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.stdout, str(data))

    def test_data_default_falls_back_to_parent(self):
        script = r"""
set -euo pipefail
source "$1/analysis/_cluster.sh"
export PHENONN_ROOT="$2"
export PHENONN_VENV=/dev/null
export PHENONN_SITE=hal
unset DATA
phenonn_cluster_init
printf '%s' "$DATA"
"""
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "pixelset1"
            data.mkdir()
            project = Path(tmp) / "repo"
            project.mkdir()
            proc = subprocess.run(
                ["bash", "-c", script, "bash", str(REPO), str(project)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.stdout, str(data))


if __name__ == "__main__":
    unittest.main()
