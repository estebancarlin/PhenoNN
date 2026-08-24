import json
import tempfile
import unittest
from pathlib import Path

from phenonn.prediction.compare_global import summarize_run


class TestCompareGlobal(unittest.TestCase):
    def test_summary_selects_best_epoch_and_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run-a"
            run.mkdir()
            (run / "config.json").write_text(
                json.dumps({"type": "lstm", "normalize": False}), encoding="utf-8"
            )
            history = [
                {
                    "epoch": 1,
                    "train": {"rmse": 0.8},
                    "validation": {"mse": 0.49, "rmse": 0.7, "r2": 0.5},
                },
                {
                    "epoch": 2,
                    "train": {"rmse": 0.6},
                    "validation": {"mse": 0.36, "rmse": 0.6, "r2": 0.7},
                },
            ]
            (run / "history.json").write_text(json.dumps(history), encoding="utf-8")
            evaluation = {"metrics": {"rmse": 0.55, "r2": 0.75}}
            (run / "evaluation_spatial_validation.json").write_text(
                json.dumps(evaluation), encoding="utf-8"
            )

            summary = summarize_run(run)

            self.assertEqual(summary["best_epoch"], 2)
            self.assertEqual(summary["joint_validation_rmse"], 0.6)
            self.assertEqual(summary["spatial_validation_rmse"], 0.55)


if __name__ == "__main__":
    unittest.main()
