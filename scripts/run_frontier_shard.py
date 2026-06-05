#!/usr/bin/env python3
"""Run the frontier benchmark for ONE task (shard), identified by index into
the task manifest. Writes per-fold rows to results/shards/{NNNN}__{task_key}.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import PROCESSED_DATA_DIR, RESULTS_DIR
from evaluation.frontier import run_single_task
from utils.logging_utils import add_file_handler, get_logger

log = get_logger("frontier_shard")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-idx", type=int, required=True,
                        help="0-based index into task_manifest.csv")
    parser.add_argument("--manifest",
                        default=str(PROCESSED_DATA_DIR / "task_manifest.csv"))
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--shards-dir",
                        default=str(RESULTS_DIR / "shards"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.task_idx < 0 or args.task_idx >= len(manifest):
        log.error("task_idx %d out of range [0, %d)", args.task_idx, len(manifest))
        return 2

    row = manifest.iloc[args.task_idx]
    task_key = row["task_key"]
    task_meta = row.to_dict()

    shards_dir = Path(args.shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    out_path = shards_dir / f"{args.task_idx:04d}__{task_key}.csv"

    log_dir = RESULTS_DIR / "logs" / "shards"
    log_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(log, log_dir / f"{args.task_idx:04d}__{task_key}.log")

    log.info("shard start: idx=%d task=%s type=%s regime=%s horizon=%s",
             args.task_idx, task_key, row["task_type"], row["regime"], row["horizon"])

    results = run_single_task(
        task_key=task_key,
        task_meta=task_meta,
        model_names=None,  # let run_single_task pick per task_type
        save_predictions=not args.no_predictions,
        max_seeds=args.max_seeds,
    )

    if not results:
        log.warning("shard produced 0 results (empty or <20 rows): %s", task_key)

    pd.DataFrame(results).to_csv(out_path, index=False)
    log.info("shard done: idx=%d task=%s rows=%d -> %s",
             args.task_idx, task_key, len(results), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
