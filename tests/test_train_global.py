import unittest

import numpy as np
import torch

from phenonn.training.train_global import (
    build_model,
    nan_safe_mse,
    sample_sites_by_chunk,
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


if __name__ == "__main__":
    unittest.main()
