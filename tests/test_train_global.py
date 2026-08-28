import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from phenonn.training.train_global import (
    build_model,
    initialize_wandb,
    nan_safe_mse,
    sample_sites_by_chunk,
    wandb_epoch_metrics,
)


class TestTrainGlobal(unittest.TestCase):
    def test_existing_models_keep_sparse_output_contract(self):
        settings = {
            "lstm": 4,
            "gru": 4,
            "transformer": 4,
            "bitransformer": 8,
            "fcn": 4,
        }
        for model_type, hidden_size in settings.items():
            model = build_model(
                model_type,
                27,
                hidden_size=hidden_size,
                num_layers=1,
                seq_length=365,
                embed_size=4,
                nhead=1,
            )
            output = model(torch.randn(2, 27, 365))
            self.assertEqual(output.shape, (2, 1, 36))

    def test_incompatible_scalar_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported global model"):
            build_model("linear", 27, hidden_size=4, num_layers=1)

    def test_nan_safe_mse_ignores_missing_targets(self):
        prediction = torch.tensor([[[1.0, 4.0]]])
        target = torch.tensor([[[2.0, float("nan")]]])
        self.assertEqual(float(nan_safe_mse(prediction, target)), 1.0)

    def test_chunk_sampling_is_reproducible(self):
        sites = np.array(["a", "b", "c", "d"])
        chunks = np.array([1, 1, 2, 3])
        first = sample_sites_by_chunk(
            sites, chunks, n_chunks=2, max_sites=0, rng=np.random.default_rng(42)
        )
        second = sample_sites_by_chunk(
            sites, chunks, n_chunks=2, max_sites=0, rng=np.random.default_rng(42)
        )
        self.assertEqual(first, second)

    def test_wandb_metrics_are_flattened(self):
        record = {
            "epoch": 3,
            "train_sites": 100,
            "train_years": [2001, 2002],
            "train": {"mse": 0.25, "rmse": 0.5, "n_valid": 10},
            "validation": {
                "mse": 0.36,
                "rmse": 0.6,
                "r2": 0.7,
                "n_valid": 8,
            },
        }
        metrics = wandb_epoch_metrics(record, 1e-3, 12.0)
        self.assertEqual(metrics["train/rmse"], 0.5)
        self.assertEqual(metrics["validation/r2"], 0.7)
        self.assertEqual(metrics["runtime/epoch_seconds"], 12.0)

    def test_wandb_configuration_excludes_local_paths(self):
        args = Namespace(
            wandb=True,
            wandb_project="phenonn-test",
            wandb_entity="",
            wandb_group="",
            wandb_tags="lstm, smoke",
            wandb_mode="offline",
            experiment="smoke",
        )
        run = Mock()
        wandb = Mock()
        wandb.init.return_value = run
        model = torch.nn.Linear(2, 1)
        configuration = {"type": "lstm", "era_dir": "D:/private/data"}
        with patch.dict("sys.modules", {"wandb": wandb}):
            result = initialize_wandb(args, configuration, Path("runs"), model)
        self.assertIs(result, run)
        sent_configuration = wandb.init.call_args.kwargs["config"]
        self.assertEqual(sent_configuration, {"type": "lstm"})


if __name__ == "__main__":
    unittest.main()
