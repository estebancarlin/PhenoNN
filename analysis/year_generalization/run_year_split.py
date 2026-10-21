#!/usr/bin/env python3
"""
run_year_split.py: train one YEAR-SPLIT of the temporal-generalization study.

Same sites, OVERLAP mode; only the train/val YEAR lists change (18 train / 10 val,
matching the chronological baseline). Splits:
  chrono         : train 1992-2009, val 2010-2019 (baseline: validate on the future)
  rand1 ... rand4  : 18 random train years, the other 10 for val (seeds 1001..1004)
  coldhot        : train on the 18 coldest years, val on the 10 hottest
                   (needs year_temp_rank.json from rank_years_by_temp.py)

Compare Delta R2 = R2(split) - R2(chrono) to see the effect of predicting the future
vs random years vs a hotter climate.

    python -m analysis.year_generalization.run_year_split --split rand1 \\
        [--all_years 1992-2019] \\
        <train_full_ram flags: fixed model config, --val_fraction_of_grid 100 ...>

The full year pool defaults to the union of the supplied ``--train_years`` and
``--val_years``; pass ``--all_years`` explicitly when those are not supplied.
"""

import argparse
import json
import os
import random

from phenonn.prediction.predict import _parse_year_spec as parse_year_spec
from phenonn.training import train_full_ram as T

N_TRAIN = 18  # -> 10 val, like the chrono baseline


def _rand_split(all_years, seed):
    """Split a year pool into 18 random training years and 10 validation years."""
    yy = list(all_years)
    random.Random(seed).shuffle(yy)
    return sorted(yy[:N_TRAIN]), sorted(yy[N_TRAIN:])


def _years_from_specs(*specs):
    """Return the sorted union of non-empty year specifications."""
    years = set()
    for spec in specs:
        if spec:
            years.update(parse_year_spec(spec) or [])
    return sorted(years)


def resolve_split(name, temp_rank_json, all_years):
    """Resolve one named split into training and validation year lists.

    Parameters
    ----------
    name : str
        Split name: ``chrono``, ``rand1`` to ``rand4``, or ``coldhot``.
    temp_rank_json : str
        JSON file written by ``rank_years_by_temp.py``. Used by ``coldhot``.
    all_years : sequence of int
        Full sorted year pool.

    Returns
    -------
    tuple of list of int
        ``(train_years, val_years)``.
    """
    if name == "chrono":
        return list(all_years[:N_TRAIN]), list(all_years[N_TRAIN:])
    if name.startswith("rand") and name[4:].isdigit():
        return _rand_split(all_years, 1000 + int(name[4:]))
    if name == "coldhot":
        if not os.path.exists(temp_rank_json):
            raise SystemExit(
                f"coldhot needs {temp_rank_json!r}: run rank_years_by_temp.py first."
            )
        with open(temp_rank_json) as f:
            order = json.load(f)["years_cold_to_hot"]
        missing = sorted(set(all_years) - set(order))
        if missing:
            raise SystemExit(
                f"coldhot ranking {temp_rank_json!r} is missing years: {missing}"
            )
        pool = set(all_years)
        ranked = [year for year in order if year in pool]
        if len(ranked) <= N_TRAIN:
            raise SystemExit(
                f"coldhot ranking has {len(ranked)} pool years; "
                f"need more than {N_TRAIN}."
            )
        return sorted(ranked[:N_TRAIN]), sorted(ranked[N_TRAIN:])
    raise SystemExit(f"unknown split {name!r} (chrono | rand1..4 | coldhot)")


def main():
    """Resolve one year split and start training through ``train_full_ram``."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit",
    )
    ap.add_argument("--split", required=True)
    ap.add_argument(
        "--temp_rank_json", default="runs/year_generalization/year_temp_rank.json"
    )
    ap.add_argument(
        "--all_years",
        default="",
        help="Full year pool (e.g. 1992-2019). Defaults to the union of "
        "--train_years and --val_years.",
    )
    known, rest = ap.parse_known_args()

    args = T.parse_args(rest)
    all_years = (
        parse_year_spec(known.all_years)
        if known.all_years
        else _years_from_specs(args.train_years, args.val_years)
    )
    if not all_years:
        ap.error(
            "no year pool: pass --all_years (e.g. --all_years 1992-2019) or "
            "supply --train_years / --val_years"
        )
    if len(all_years) <= N_TRAIN:
        ap.error(f"--all_years has {len(all_years)} years; need more than {N_TRAIN}")

    tr, va = resolve_split(known.split, known.temp_rank_json, all_years)
    args.train_years = ",".join(map(str, tr))
    args.val_years = ",".join(map(str, va))
    args.experiment = f"yeargen_{known.split}"

    paths, logger = T.setup_directories_and_logging(args)
    device = T.setup_device_and_seed(args, logger)
    logger.show_header(f"Year-split [{known.split}]: {args.type}")
    logger.info(f"Year pool   : {all_years[0]}-{all_years[-1]} ({len(all_years)})")
    logger.info(f"Train years : {args.train_years}")
    logger.info(f"Val years   : {args.val_years}")

    if os.path.exists(os.path.join(paths.exp, "config.json")):
        logger.info(f"[skip] {args.experiment}: already completed.")
        return

    shared = T.load_shared(args, logger, device)
    T.train_one_config(args, shared, logger, paths)


if __name__ == "__main__":
    main()
