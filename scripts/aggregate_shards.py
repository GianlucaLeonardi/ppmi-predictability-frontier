#!/usr/bin/env python3
"""Concatenate per-task shard CSVs into the canonical frontier output files,
cross-checking the shard set against the manifest before aggregating.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import PROCESSED_DATA_DIR, RESULTS_DIR
from evaluation.frontier import aggregate_cv_results
from utils.logging_utils import add_file_handler, get_logger

log = get_logger("aggregate_shards")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",
                        default=str(PROCESSED_DATA_DIR / "task_manifest.csv"))
    parser.add_argument("--shards-dir",
                        default=str(RESULTS_DIR / "shards"))
    parser.add_argument("--allow-missing", action="store_true",
                        help="Skip missing shards instead of aborting (for partial reruns)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Debug: aggregate only the first N tasks from the manifest. "
                             "Must match MAX_TASKS used for shard submission.")
    args = parser.parse_args()

    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(log, log_dir / "aggregate_shards.log")

    manifest = pd.read_csv(args.manifest)
    if args.max_tasks is not None and args.max_tasks < len(manifest):
        log.info("aggregating first %d of %d tasks (--max-tasks debug)",
                 args.max_tasks, len(manifest))
        manifest = manifest.head(args.max_tasks)
    shards_dir = Path(args.shards_dir)
    n_tasks = len(manifest)

    expected = {
        i: shards_dir / f"{i:04d}__{row['task_key']}.csv"
        for i, row in manifest.iterrows()
    }

    missing = [i for i, p in expected.items() if not p.exists()]
    if missing:
        msg = f"{len(missing)}/{n_tasks} shard files missing (e.g. idx={missing[:5]})"
        if args.allow_missing:
            log.warning(msg + " — continuing with partial aggregate")
        else:
            log.error(msg)
            log.error("Re-run the missing shards (indices %s) or pass --allow-missing.",
                      ",".join(str(i) for i in missing[:20]))
            return 2

    frames = []
    for i, p in expected.items():
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            log.warning("shard %d (%s) is empty", i, p.name)
            continue
        frames.append(df)

    if not frames:
        log.error("no non-empty shards found in %s", shards_dir)
        return 3

    fold_df = pd.concat(frames, ignore_index=True, sort=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = RESULTS_DIR / "frontier_results_per_fold.csv"
    fold_df.to_csv(fold_path, index=False)
    log.info("wrote %s (%d rows from %d shards)",
             fold_path, len(fold_df), len(frames))

    agg_df = aggregate_cv_results(fold_df)
    agg_path = RESULTS_DIR / "frontier_results.csv"
    agg_df.to_csv(agg_path, index=False)
    log.info("wrote %s (%d aggregated rows)", agg_path, len(agg_df))

    if not agg_df.empty and {"r2", "spearman"}.issubset(agg_df.columns):
        from evaluation.metrics import detect_r2_spearman_divergence
        divs = detect_r2_spearman_divergence(agg_df)
        if divs:
            div_path = RESULTS_DIR / "r2_spearman_divergences.csv"
            pd.DataFrame(divs).to_csv(div_path, index=False)
            log.info("wrote %s (%d divergence cases)", div_path, len(divs))

    if "error" in fold_df.columns:
        n_err = fold_df["error"].notna().sum()
        if n_err > 0:
            log.warning("%d model fits had errors across all shards", n_err)

    return 0


if __name__ == "__main__":
    sys.exit(main())
