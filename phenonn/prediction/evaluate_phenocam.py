# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh

"""Compare global LAI predictions with independent Phenocam GCC curves."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _correlation(left, right):
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _robust_scale(values):
    lower, upper = np.quantile(values, [0.05, 0.95])
    amplitude = float(upper - lower)
    if amplitude <= 0:
        return np.full_like(values, np.nan, dtype=np.float64), amplitude
    scaled = np.clip((values - lower) / amplitude, 0.0, 1.0)
    return scaled, amplitude


def _curve_dates(dates, values, threshold):
    if not np.isfinite(values).all():
        return None, None, None
    peak_index = int(np.argmax(values))
    peak = dates.iloc[peak_index]
    rising = np.flatnonzero(values[: peak_index + 1] >= threshold)
    falling = np.flatnonzero(values[peak_index:] <= threshold)
    start = dates.iloc[int(rising[0])] if rising.size else None
    end = dates.iloc[peak_index + int(falling[0])] if falling.size else None
    return start, peak, end


def _day_difference(left, right):
    if left is None or right is None:
        return np.nan
    return int((left - right).days)


def evaluate_frames(
    predictions,
    observations,
    *,
    site_column="site",
    date_column="date",
    prediction_column="lai_pred",
    gcc_column="gcc",
    roi_column=None,
    quality_column=None,
    valid_quality=None,
    min_points=18,
    threshold=0.2,
    min_lai_amplitude=0.1,
    min_gcc_amplitude=0.02,
):
    """Align annual curves and compute association, shape, and timing metrics."""
    prediction_columns = [site_column, date_column, prediction_column]
    observation_columns = [site_column, date_column, gcc_column]
    group_columns = [site_column]
    if roi_column:
        observation_columns.append(roi_column)
        group_columns.append(roi_column)
    if quality_column:
        observation_columns.append(quality_column)

    predictions = predictions[prediction_columns].copy()
    observations = observations[observation_columns].copy()
    predictions[date_column] = pd.to_datetime(predictions[date_column], errors="raise")
    observations[date_column] = pd.to_datetime(
        observations[date_column], errors="raise"
    )
    if quality_column and valid_quality is not None:
        observations = observations[
            observations[quality_column].astype(str).isin(set(valid_quality))
        ]

    duplicate_prediction = predictions.duplicated([site_column, date_column])
    duplicate_observation = observations.duplicated(group_columns + [date_column])
    if duplicate_prediction.any():
        raise ValueError("Predictions contain duplicate site/date rows")
    if duplicate_observation.any():
        raise ValueError("Observations contain duplicate site/ROI/date rows")

    aligned = observations.merge(
        predictions,
        on=[site_column, date_column],
        how="inner",
        validate="many_to_one" if roi_column else "one_to_one",
    )
    aligned[gcc_column] = pd.to_numeric(aligned[gcc_column], errors="coerce")
    aligned[prediction_column] = pd.to_numeric(
        aligned[prediction_column], errors="coerce"
    )
    aligned = aligned[
        np.isfinite(aligned[gcc_column]) & np.isfinite(aligned[prediction_column])
    ].copy()
    aligned["year"] = aligned[date_column].dt.year

    records = []
    scaled_groups = []
    annual_groups = group_columns + ["year"]
    for keys, group in aligned.groupby(annual_groups, sort=True, dropna=False):
        group = group.sort_values(date_column).copy()
        if len(group) < min_points:
            continue
        lai = group[prediction_column].to_numpy(dtype=np.float64)
        gcc = group[gcc_column].to_numpy(dtype=np.float64)
        lai_scaled, lai_amplitude = _robust_scale(lai)
        gcc_scaled, gcc_amplitude = _robust_scale(gcc)
        group["lai_shape"] = lai_scaled
        group["gcc_shape"] = gcc_scaled
        scaled_groups.append(group)

        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(annual_groups, keys))
        record.update(
            {
                "n_matched": int(len(group)),
                "pearson": _correlation(lai, gcc),
                "spearman": _correlation(
                    pd.Series(lai).rank().to_numpy(),
                    pd.Series(gcc).rank().to_numpy(),
                ),
                "shape_rmse": float(np.sqrt(np.mean((lai_scaled - gcc_scaled) ** 2))),
                "shape_mae": float(np.mean(np.abs(lai_scaled - gcc_scaled))),
                "lai_amplitude": lai_amplitude,
                "gcc_amplitude": gcc_amplitude,
                "timing_valid": bool(
                    lai_amplitude >= min_lai_amplitude
                    and gcc_amplitude >= min_gcc_amplitude
                ),
            }
        )
        if record["timing_valid"]:
            lai_dates = _curve_dates(group[date_column], lai_scaled, threshold)
            gcc_dates = _curve_dates(group[date_column], gcc_scaled, threshold)
            record.update(
                {
                    "start_error_days": _day_difference(lai_dates[0], gcc_dates[0]),
                    "peak_error_days": _day_difference(lai_dates[1], gcc_dates[1]),
                    "end_error_days": _day_difference(lai_dates[2], gcc_dates[2]),
                }
            )
        else:
            record.update(
                {
                    "start_error_days": np.nan,
                    "peak_error_days": np.nan,
                    "end_error_days": np.nan,
                }
            )
        records.append(record)

    annual = pd.DataFrame.from_records(records)
    aligned = (
        pd.concat(scaled_groups, ignore_index=True) if scaled_groups else aligned[:0]
    )
    metric_columns = [
        "pearson",
        "spearman",
        "shape_rmse",
        "shape_mae",
        "start_error_days",
        "peak_error_days",
        "end_error_days",
    ]
    summary = {
        "n_aligned_rows": int(len(aligned)),
        "n_site_years": int(len(annual)),
        "n_timing_site_years": (
            int(annual["timing_valid"].sum()) if not annual.empty else 0
        ),
        "macro_mean": {
            column: (
                float(annual[column].mean())
                if not annual.empty and annual[column].notna().any()
                else None
            )
            for column in metric_columns
        },
        "macro_median": {
            column: (
                float(annual[column].median())
                if not annual.empty and annual[column].notna().any()
                else None
            )
            for column in metric_columns
        },
    }
    return aligned, annual, summary


def evaluate_phenocam(args=None):
    if args is None:
        args = parse_args()
    prediction_path = Path(args.predictions)
    observation_path = Path(args.observations)
    predictions = pd.read_csv(prediction_path)
    observations = pd.read_csv(observation_path, skiprows=args.observation_skiprows)
    valid_quality = args.valid_quality.split(",") if args.valid_quality else None
    aligned, annual, summary = evaluate_frames(
        predictions,
        observations,
        site_column=args.site_column,
        date_column=args.date_column,
        prediction_column=args.prediction_column,
        gcc_column=args.gcc_column,
        roi_column=args.roi_column or None,
        quality_column=args.quality_column or None,
        valid_quality=valid_quality,
        min_points=args.min_points,
        threshold=args.threshold,
        min_lai_amplitude=args.min_lai_amplitude,
        min_gcc_amplitude=args.min_gcc_amplitude,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_dir / "phenocam_aligned.csv", index=False)
    annual.to_csv(output_dir / "phenocam_site_year_metrics.csv", index=False)
    result = {
        "protocol": {
            "comparison": "GCC association, independently scaled shape, and timing",
            "physical_lai_error": False,
            "date_join": "exact",
            "robust_scale_quantiles": [0.05, 0.95],
            "timing_threshold": args.threshold,
            "min_points": args.min_points,
            "min_lai_amplitude": args.min_lai_amplitude,
            "min_gcc_amplitude": args.min_gcc_amplitude,
        },
        "inputs": {
            "predictions": str(prediction_path.resolve()),
            "predictions_sha256": _sha256(prediction_path),
            "observations": str(observation_path.resolve()),
            "observations_sha256": _sha256(observation_path),
        },
        "summary": summary,
    }
    output = output_dir / "phenocam_evaluation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--site-column", default="site")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--prediction-column", default="lai_pred")
    parser.add_argument("--gcc-column", default="gcc")
    parser.add_argument("--roi-column", default="")
    parser.add_argument("--quality-column", default="")
    parser.add_argument("--valid-quality", default="")
    parser.add_argument("--observation-skiprows", type=int, default=0)
    parser.add_argument("--min-points", type=int, default=18)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--min-lai-amplitude", type=float, default=0.1)
    parser.add_argument("--min-gcc-amplitude", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_phenocam()
