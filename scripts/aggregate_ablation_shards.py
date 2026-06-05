#!/usr/bin/env python3
"""Concatenate per-target ablation shard CSVs into the canonical
results/tables/ablation_perfold_xgboost.csv read by the Step 5 stat tests.
Use --allow-missing to skip absent shards for partial reruns.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import PROCESSED_DATA_DIR, RESULTS_DIR
from scripts.run_ablation_perfold_shard import shard_task_keys
from utils.logging_utils import add_file_handler, get_logger

log = get_logger("aggregate_ablation_shards")


SHARDS_DIR = RESULTS_DIR / "ablation_shards"
OUT_PATH = RESULTS_DIR / "tables" / "ablation_perfold_xgboost.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",
                        default=str(PROCESSED_DATA_DIR / "task_manifest.csv"))
    parser.add_argument("--shards-dir", default=str(SHARDS_DIR))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--allow-missing", action="store_true",
                        help="Skip missing shards instead of aborting "
                             "(for partial reruns)")
    args = parser.parse_args()

    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(log, log_dir / "aggregate_ablation_shards.log")

    targets = shard_task_keys(Path(args.manifest))
    shards_dir = Path(args.shards_dir)
    out_path = Path(args.out)

    log.info("Expected shards: %d (one per regression ablation target)", len(targets))
    log.info("Shards dir: %s", shards_dir)

    found = []
    missing = []
    empty = []
    for i, target in enumerate(targets):
        p = shards_dir / f"{i:04d}__{target}.csv"
        if not p.exists():
            missing.append((i, target, p))
            continue
        if p.stat().st_size == 0:
            empty.append((i, target, p))
            continue
        found.append((i, target, p))

    if missing or empty:
        msg = (f"Missing {len(missing)} ablation shard(s); "
               f"empty {len(empty)} shard(s). Found {len(found)}/{len(targets)}.")
        if missing:
            log.warning("Missing shards: %s",
                        ", ".join(f"{i}:{t}" for i, t, _ in missing))
        if empty:
            log.warning("Empty shards: %s",
                        ", ".join(f"{i}:{t}" for i, t, _ in empty))
        if not args.allow_missing:
            log.error(msg + " Re-run the missing/empty shards:")
            for i, t, _ in missing + empty:
                log.error("  shard %d (%s)", i, t)
            return 2
        log.warning(msg + " --allow-missing set; continuing with partial aggregate.")

    if not found:
        log.error("No non-empty ablation shards found in %s", shards_dir)
        return 3

    frames = []
    for i, target, p in found:
        try:
            df = pd.read_csv(p)
            log.info("  shard %d (%s): %d rows", i, target, len(df))
            frames.append(df)
        except Exception as e:
            log.warning("  shard %d (%s): read failed (%s) -- skipping", i, target, e)

    if not frames:
        log.error("All shards failed to read; nothing to aggregate")
        return 4

    out_df = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows from %d shards)", out_path, len(out_df), len(frames))
    return 0


if __name__ == "__main__":
    sys.exit(main())
