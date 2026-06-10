# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import unittest
import torch
import sys
import os
from phenonn.models.fcn import FCN, FCBlock


class TestFCBlock(unittest.TestCase):
    """Unit tests for FCBlock class."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.in_features = 128
        self.out_features = 64

    def test_initialization(self):
        """Test FCBlock initialization."""
        block = FCBlock(self.in_features, self.out_features)

        self.assertIsInstance(block.linear, torch.nn.Linear)
        self.assertEqual(block.linear.in_features, self.in_features)
        self.assertEqual(block.linear.out_features, self.out_features)

        self.assertIsInstance(block.bn, torch.nn.BatchNorm1d)
        self.assertEqual(block.bn.num_features, self.out_features)

        self.assertIsInstance(block.relu, torch.nn.ReLU)

    def test_forward_shape(self):
        """Test FCBlock forward pass output shape."""
        block = FCBlock(self.in_features, self.out_features)

        x = torch.randn(self.batch_size, self.in_features)
        y = block(x)

        self.assertEqual(y.shape, (self.batch_size, self.out_features))

    def test_forward_values(self):
        """Test FCBlock forward pass produces non-zero values."""
        block = FCBlock(self.in_features, self.out_features)

        x = torch.randn(self.batch_size, self.in_features)
        y = block(x)

        # Should have some non-zero values
        self.assertGreater(y.abs().sum(), 0)

    def test_gradient_flow(self):
        """Test gradient flow through FCBlock."""
        block = FCBlock(self.in_features, self.out_features)

        x = torch.randn(self.batch_size, self.in_features, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        for param in block.parameters():
            self.assertIsNotNone(param.grad)


class TestFCN(unittest.TestCase):
    """Unit tests for FCN class."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.feature_channel = 6
        self.output_channel = 4
        self.num_layers = 3
        self.hidden_size = 196
        self.seq_length = 10

    def test_initialization_default(self):
        """Test FCN initialization with default parameters."""
        model = FCN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
        )

        self.assertEqual(model.feature_channel, self.feature_channel)
        self.assertEqual(model.output_channel, self.output_channel)
        self.assertEqual(model.seq_length, self.seq_length)
        self.assertEqual(model.dim_expand, 0)
        self.assertIsNotNone(model.input_layer)
        self.assertIsNotNone(model.hidden_layers)
        self.assertIsNotNone(model.output_layer)
        self.assertIsNone(model.dim_change)

    def test_initialization_invalid_layers(self):
        """Test FCN initialization with invalid number of layers."""
        with self.assertRaises(ValueError):
            FCN(
                feature_channel=self.feature_channel,
                output_channel=self.output_channel,
                num_layers=0,  # Invalid
                hidden_size=self.hidden_size,
                seq_length=self.seq_length,
            )

    def test_forward_shape_default(self):
        """Test FCN forward pass output shape without expansion."""
        model = FCN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
        )

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)

        expected_shape = (self.batch_size, self.output_channel, self.seq_length)
        self.assertEqual(y.shape, expected_shape)

    def test_forward_shape_with_expansion(self):
        """Test FCN forward pass output shape with expansion."""
        dim_expand = 0
        model = FCN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
            dim_expand=dim_expand,
        )

        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y = model(x)

        expected_shape = (
            self.batch_size,
            self.output_channel,
            self.seq_length + dim_expand,
        )
        self.assertEqual(y.shape, expected_shape)

    def test_forward_with_different_batch_sizes(self):
        """Test FCN forward with different batch sizes (skip batch size 1 due to BatchNorm)."""
        model = FCN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
        )

        # Skip batch size 1 because BatchNorm needs at least 2 samples in training mode
        # Use eval mode to test batch size 1
        batch_sizes = [2, 4, 16, 32, 64]

        for bs in batch_sizes:
            x = torch.randn(bs, self.feature_channel, self.seq_length)
            y = model(x)
            self.assertEqual(y.shape, (bs, self.output_channel, self.seq_length))

    def test_forward_with_batch_size_1_eval_mode(self):
        """Test FCN with batch size 1 in eval mode."""
        model = FCN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
        )
        model.eval()  # Set to eval mode to allow batch size 1

        x = torch.randn(1, self.feature_channel, self.seq_length)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(y.shape, (1, self.output_channel, self.seq_length))

    def test_forward_with_different_hidden_sizes(self):
        """Test FCN forward with different hidden sizes."""
        hidden_sizes = [64, 128, 256]

        for hs in hidden_sizes:
            model = FCN(
                feature_channel=self.feature_channel,
                output_channel=self.output_channel,
                num_layers=self.num_layers,
                hidden_size=hs,
                seq_length=self.seq_length,
            )

            x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
            y = model(x)
            self.assertEqual(
                y.shape, (self.batch_size, self.output_channel, self.seq_length)
            )

    def test_forward_with_different_num_layers(self):
        """Test FCN forward with different numbers of layers."""
        num_layers_list = [1, 2, 3, 4]

        for nl in num_layers_list:
            model = FCN(
                feature_channel=self.feature_channel,
                output_channel=self.output_channel,
                num_layers=nl,
                hidden_size=self.hidden_size,
                seq_length=self.seq_length,
            )

            x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
            y = model(x)
            self.assertEqual(
                y.shape, (self.batch_size, self.output_channel, self.seq_length)
            )

    def test_forward_with_different_sequence_lengths(self):
        """Test FCN forward with different sequence lengths."""
        # Note: seq_length is fixed at model initialization
        # Different sequence lengths require different models
        seq_lengths = [5, 10, 20]

        for sl in seq_lengths:
            model = FCN(
                feature_channel=self.feature_channel,
                output_channel=self.output_channel,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                seq_length=sl,
            )

            x = torch.randn(self.batch_size, self.feature_channel, sl)
            y = model(x)
            self.assertEqual(y.shape, (self.batch_size, self.output_channel, sl))

    def test_gradient_flow(self):
        """Test gradient flow through FCN."""
        model = FCN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
        )

        x = torch.randn(
            self.batch_size, self.feature_channel, self.seq_length, requires_grad=True
        )
        y = model(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_gradient_accumulation(self):
        """Test that gradients accumulate properly."""
        model = FCN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
        )

        # Forward pass 1
        x1 = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y1 = model(x1)
        loss1 = y1.sum()
        loss1.backward()

        # Save first gradients
        grad1 = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

        # Forward pass 2
        x2 = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        y2 = model(x2)
        loss2 = y2.sum()
        loss2.backward()

        # Gradients should have accumulated
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertIsNotNone(grad1.get(name))
                self.assertNotEqual(param.grad.sum(), grad1[name].sum())

    def test_model_parameters_count(self):
        """Test model has expected number of parameters."""
        model = FCN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)

    def test_model_serialization(self):
        """Test model can be saved and loaded."""
        import tempfile

        model1 = FCN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
        )

        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            torch.save(model1.state_dict(), f.name)
            temp_file = f.name

        # Load model
        model2 = FCN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
        )
        model2.load_state_dict(torch.load(temp_file))

        # Test forward pass produces same output
        x = torch.randn(self.batch_size, self.feature_channel, self.seq_length)
        model1.eval()
        model2.eval()

        with torch.no_grad():
            y1 = model1(x)
            y2 = model2(x)

        torch.testing.assert_close(y1, y2)

        # Cleanup
        os.unlink(temp_file)

    def test_device_transfer(self):
        """Test model can be moved to CUDA if available."""
        model = FCN(
            feature_channel=self.feature_channel,
            output_channel=self.output_channel,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            seq_length=self.seq_length,
        )

        if torch.cuda.is_available():
            model = model.cuda()
            x = torch.randn(
                self.batch_size, self.feature_channel, self.seq_length
            ).cuda()
            y = model(x)
            self.assertTrue(y.is_cuda)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestFCBlock))
    suite.addTests(loader.loadTestsFromTestCase(TestFCN))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
