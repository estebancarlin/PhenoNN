# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Stefan Barbu, Kazem Ardaneh

"""Summarize and compare global LAI training runs and evaluations."""

import argparse
import csv
import json
from pathlib import Path


def summarize_run(run_path):
    run_path = Path(run_path)
    configuration = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
    history = json.loads((run_path / "history.json").read_text(encoding="utf-8"))
    if not history:
        raise ValueError(f"Training history is empty: {run_path}")
    best = min(history, key=lambda record: record["validation"]["mse"])
    row = {
        "run": run_path.name,
        "path": str(run_path.resolve()),
        "architecture": configuration["type"],
        "normalized": bool(configuration.get("normalize", False)),
        "epochs_completed": len(history),
        "best_epoch": int(best["epoch"]),
        "train_rmse_at_best": float(best["train"]["rmse"]),
        "joint_validation_rmse": float(best["validation"]["rmse"]),
        "joint_validation_r2": float(best["validation"].get("r2", float("nan"))),
    }
    for evaluation in sorted(run_path.glob("evaluation_*.json")):
        result = json.loads(evaluation.read_text(encoding="utf-8"))
        label = evaluation.stem[len("evaluation_") :]
        row[f"{label}_rmse"] = float(result["metrics"]["rmse"])
        row[f"{label}_r2"] = float(result["metrics"].get("r2", float("nan")))
    return row


def compare_global(args=None):
    if args is None:
        args = parse_args()
    rows = [summarize_run(path) for path in args.runs]
    rows.sort(key=lambda row: row["joint_validation_rmse"])
    result = {"ranked_by": "joint_validation_rmse", "runs": rows}
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with open(
            output.with_suffix(".csv"), "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    compare_global()
