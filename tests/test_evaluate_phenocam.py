import unittest

import numpy as np
import pandas as pd

from phenonn.prediction.evaluate_phenocam import evaluate_frames


class TestEvaluatePhenocam(unittest.TestCase):
    def make_curves(self):
        dates = pd.date_range("2018-01-05", periods=36, freq="10D")
        phase = np.linspace(0, 2 * np.pi, len(dates), endpoint=False)
        shape = 0.5 - 0.5 * np.cos(phase)
        predictions = pd.DataFrame(
            {"site": "site-a", "date": dates, "lai_pred": 0.3 + 4 * shape}
        )
        observations = pd.DataFrame(
            {"site": "site-a", "date": dates, "gcc": 0.25 + 0.15 * shape}
        )
        return predictions, observations

    def test_identical_shape_has_perfect_association_and_timing(self):
        predictions, observations = self.make_curves()
        aligned, annual, summary = evaluate_frames(predictions, observations)

        self.assertEqual(len(aligned), 36)
        self.assertEqual(summary["n_site_years"], 1)
        self.assertAlmostEqual(annual.loc[0, "pearson"], 1.0)
        self.assertGreater(annual.loc[0, "spearman"], 0.999)
        self.assertAlmostEqual(annual.loc[0, "shape_rmse"], 0.0)
        self.assertEqual(annual.loc[0, "peak_error_days"], 0)

    def test_low_amplitude_gcc_suppresses_timing_only(self):
        predictions, observations = self.make_curves()
        observations["gcc"] = 0.3 + 0.0001 * np.arange(len(observations))
        _, annual, summary = evaluate_frames(predictions, observations)

        self.assertFalse(annual.loc[0, "timing_valid"])
        self.assertTrue(np.isnan(annual.loc[0, "start_error_days"]))
        self.assertEqual(summary["n_timing_site_years"], 0)

    def test_quality_filter_and_roi_are_preserved(self):
        predictions, observations = self.make_curves()
        observations["roi"] = "grass"
        observations["quality"] = "good"
        observations.loc[:20, "quality"] = "bad"
        aligned, annual, _ = evaluate_frames(
            predictions,
            observations,
            roi_column="roi",
            quality_column="quality",
            valid_quality=["good"],
            min_points=10,
        )

        self.assertEqual(len(aligned), 15)
        self.assertEqual(annual.loc[0, "roi"], "grass")

    def test_duplicate_predictions_are_rejected(self):
        predictions, observations = self.make_curves()
        predictions = pd.concat([predictions, predictions.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "duplicate site/date"):
            evaluate_frames(predictions, observations)

    def test_short_year_is_excluded(self):
        predictions, observations = self.make_curves()
        _, annual, summary = evaluate_frames(
            predictions.iloc[:10], observations.iloc[:10]
        )
        self.assertTrue(annual.empty)
        self.assertEqual(summary["n_site_years"], 0)


if __name__ == "__main__":
    unittest.main()
