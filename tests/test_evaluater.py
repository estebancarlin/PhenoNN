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
from unittest.mock import Mock
import numpy as np

from phenonn.utils.evaluater import (
    NMSELoss,
    NMAELoss,
    MetricTracker,
    get_loss_function,
    mse_all,
    mbe_all,
    mae_all,
    r2_all,
    nmae_all,
    nmse_all,
    mare_all,
    gmrae_all,
)


class TestNMSELoss(unittest.TestCase):
    """Unit tests for NMSELoss class."""

    def setUp(self):
        """Set up test fixtures."""
        self.criterion = NMSELoss(eps=1e-8)

    def test_forward(self):
        """Test NMSELoss forward pass."""
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        loss = self.criterion(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_non_zero_loss(self):
        """Test NMSELoss with non-zero loss."""
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = self.criterion(pred, target)
        self.assertGreater(loss.item(), 0.0)

    def test_normalization(self):
        """Test that loss is normalized by target variance."""
        pred = torch.tensor([2.0, 4.0, 6.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        mse = torch.mean((pred - target) ** 2)
        norm = torch.mean(target**2)
        expected = mse / norm
        loss = self.criterion(pred, target)
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)


class TestNMAELoss(unittest.TestCase):
    """Unit tests for NMAELoss class."""

    def setUp(self):
        """Set up test fixtures."""
        self.criterion = NMAELoss(eps=1e-8)

    def test_forward(self):
        """Test NMAELoss forward pass."""
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        loss = self.criterion(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_non_zero_loss(self):
        """Test NMAELoss with non-zero loss."""
        pred = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = self.criterion(pred, target)
        self.assertGreater(loss.item(), 0.0)


class TestMetricTracker(unittest.TestCase):
    """Unit tests for MetricTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = MetricTracker()

    def test_metric_tracker_init(self):
        """Test MetricTracker initialization."""

        self.assertEqual(self.tracker.value, 0.0)
        self.assertEqual(self.tracker.count, 0)

    def test_metric_tracker_reset(self):
        """Test MetricTracker reset method."""

        tracker = MetricTracker()
        tracker.value = 10.5
        tracker.count = 5
        tracker.reset()
        self.assertEqual(tracker.value, 0.0)
        self.assertEqual(tracker.count, 0)

    def test_metric_tracker_update(self):
        """Test MetricTracker update method."""

        tracker = MetricTracker()

        # First update
        tracker.update(10.0, 5)
        self.assertEqual(tracker.value, 50.0)  # 10 * 5
        self.assertEqual(tracker.count, 5)

        # Second update
        tracker.update(20.0, 3)
        self.assertEqual(tracker.value, 110.0)  # 50 + 20*3
        self.assertEqual(tracker.count, 8)  # 5 + 3

        # Third update with zero count
        tracker.update(30.0, 0)
        self.assertEqual(tracker.value, 110.0)  # Unchanged
        self.assertEqual(tracker.count, 8)  # Unchanged

    def test_metric_tracker_getmean(self):
        """Test MetricTracker getmean method."""

        tracker = MetricTracker()

        # Test with valid updates
        tracker.update(10.0, 5)
        tracker.update(20.0, 3)

        mean = tracker.getmean()
        expected_mean = 110.0 / 8  # (10*5 + 20*3) / (5+3) = 110/8 = 13.75
        self.assertAlmostEqual(mean, expected_mean, places=6)

        # Test with zero count (should raise ZeroDivisionError)
        tracker.reset()
        with self.assertRaises(ZeroDivisionError):
            tracker.getmean()

    def test_metric_tracker_getstd(self):
        """Test MetricTracker getstd method."""

        tracker = MetricTracker()

        # Known values
        # Values: [10 (×5), 20 (×3)]
        # mean = 13.75
        # E[x^2] = (10^2 * 5 + 20^2 * 3) / 8 = (500 + 1200) / 8 = 212.5
        # variance = 212.5 - 13.75^2 = 23.4375
        # std = sqrt(23.4375) ≈ 4.841229
        tracker.update(10.0, 5)
        tracker.update(20.0, 3)

        std = tracker.getstd()
        expected_std = np.sqrt(212.5 - 13.75**2)

        self.assertAlmostEqual(std, expected_std, places=6)

        # Test with zero count (should raise ZeroDivisionError)
        tracker.reset()
        with self.assertRaises(ZeroDivisionError):
            tracker.getstd()

    def test_metric_tracker_getsqrtmean(self):
        """Test MetricTracker getsqrtmean method."""

        tracker = MetricTracker()

        tracker.update(16.0, 2)  # mean = 16, sqrt = 4
        tracker.update(4.0, 2)  # mean = (16*2 + 4*2)/4 = 10, sqrt = sqrt(10)

        sqrtmean = tracker.getsqrtmean()
        expected_sqrtmean = np.sqrt(10.0)  # sqrt(10) ≈ 3.16227766
        self.assertAlmostEqual(sqrtmean, expected_sqrtmean, places=6)

        # Test with zero count (should raise ZeroDivisionError)
        tracker.reset()
        with self.assertRaises(ZeroDivisionError):
            tracker.getsqrtmean()


class TestGetLossFunction(unittest.TestCase):
    """Unit tests for get_loss_function factory."""

    def test_mse_loss(self):
        """Test MSE loss creation."""
        args = Mock()
        loss = get_loss_function("mse", args)
        self.assertIsInstance(loss, torch.nn.MSELoss)

    def test_mae_loss(self):
        """Test MAE loss creation."""
        args = Mock()
        loss = get_loss_function("mae", args)
        self.assertIsInstance(loss, torch.nn.L1Loss)

    def test_nmae_loss(self):
        """Test NMAE loss creation."""
        args = Mock()
        loss = get_loss_function("nmae", args)
        self.assertIsInstance(loss, NMAELoss)

    def test_nmse_loss(self):
        """Test NMSE loss creation."""
        args = Mock()
        loss = get_loss_function("nmse", args)
        self.assertIsInstance(loss, NMSELoss)

    def test_huber_loss(self):
        """Test Huber loss creation."""
        args = Mock()
        args.beta_delta = 1.0
        loss = get_loss_function("huber", args)
        self.assertIsInstance(loss, torch.nn.HuberLoss)

    def test_smoothl1_loss(self):
        """Test SmoothL1 loss creation."""
        args = Mock()
        args.beta_delta = 1.0
        loss = get_loss_function("smoothl1", args)
        self.assertIsInstance(loss, torch.nn.SmoothL1Loss)

    def test_invalid_loss_type(self):
        """Test invalid loss type raises error."""
        args = Mock()
        with self.assertRaises(ValueError):
            get_loss_function("invalid_loss", args)

    def test_huber_without_beta_delta(self):
        """Test Huber loss without beta_delta raises error."""
        args = Mock()
        delattr(args, "beta_delta")
        with self.assertRaises(ValueError):
            get_loss_function("huber", args)


class TestMetricFunctions(unittest.TestCase):
    """Unit tests for metric functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.pred = torch.tensor([2.0, 3.0, 4.0])
        self.true = torch.tensor([1.0, 2.0, 3.0])

    def test_mse_all(self):
        """Test MSE metric."""
        count, value = mse_all(self.pred, self.true)
        expected_mse = (1**2 + 1**2 + 1**2) / 3
        self.assertEqual(count, 3)
        self.assertAlmostEqual(value.item(), expected_mse, places=6)

    def test_mbe_all(self):
        """Test MBE metric."""
        count, value = mbe_all(self.pred, self.true)
        expected_mbe = (1 + 1 + 1) / 3
        self.assertEqual(count, 3)
        self.assertAlmostEqual(value.item(), expected_mbe, places=6)

    def test_mae_all(self):
        """Test MAE metric."""
        count, value = mae_all(self.pred, self.true)
        expected_mae = (1 + 1 + 1) / 3
        self.assertEqual(count, 3)
        self.assertAlmostEqual(value.item(), expected_mae, places=6)

    def test_r2_all(self):
        """Test R² metric."""
        pred = torch.tensor([2.0, 3.0, 4.0])
        true = torch.tensor([2.0, 3.0, 4.0])
        count, value = r2_all(pred, true)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(value.item(), 1.0, places=6)

    def test_r2_all_perfect(self):
        """Test R² metric with perfect predictions."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([1.0, 2.0, 3.0])
        count, value = r2_all(pred, true)
        self.assertAlmostEqual(value.item(), 1.0, places=6)

    def test_nmae_all(self):
        """Test NMAE metric."""
        count, value = nmae_all(self.pred, self.true)
        self.assertEqual(count, 3)
        self.assertGreater(value.item(), 0)

    def test_nmse_all(self):
        """Test NMSE metric."""
        count, value = nmse_all(self.pred, self.true)
        self.assertEqual(count, 3)
        self.assertGreater(value.item(), 0)

    def test_mare_all(self):
        """Test MARE metric."""
        count, value = mare_all(self.pred, self.true)
        self.assertEqual(count, 3)
        self.assertGreater(value.item(), 0)

    def test_gmrae_all(self):
        """Test GMRAE metric."""
        count, value = gmrae_all(self.pred, self.true)
        self.assertEqual(count, 3)
        self.assertGreater(value.item(), 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestNMSELoss))
    suite.addTests(loader.loadTestsFromTestCase(TestNMAELoss))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestGetLossFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricFunctions))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
