#!/usr/bin/env python3
"""Per-task shard wrapper for the per-fold modality ablation.

Runs the full per-fold ablation for one target (by 0-based index) across all
(regime, horizon) combos, writing one CSV per shard.

Run one shard per invocation (target selected by its 0-based index); dispatch
the shards across workers with your own job scheduler.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import PROCESSED_DATA_DIR, RESULTS_DIR
from utils.logging_utils import add_file_handler, get_logger

log = get_logger("ablation_shard")


# Same canonical grid as `run_all.py` Step 3 -- keep these in sync.
ABLATION_GRID = [
    ("baseline_multimodal", "V06"),
    ("baseline_multimodal", "V08"),
    ("baseline_plus_12m", "V08"),
    ("rolling", "V08"),
]


def shard_task_keys(manifest_path: Path) -> list:
    """Return the de-duplicated regression target list that participates in the ablation grid."""
    manifest = pd.read_csv(manifest_path)
    sub = manifest[manifest["task_type"] == "regression"].copy()
    targets = (
        sub[sub.apply(lambda r: (r["regime"], r["horizon"]) in ABLATION_GRID, axis=1)]
        ["target"].drop_duplicates().sort_values().tolist()
    )
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-idx", type=int, required=True,
                        help="0-based index into the ablation target list")
    parser.add_argument("--manifest",
                        default=str(PROCESSED_DATA_DIR / "task_manifest.csv"))
    parser.add_argument("--no-incremental", action="store_true",
                        help="Skip per-(regime, horizon) flush (only final write)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    targets = shard_task_keys(manifest_path)

    if args.task_idx < 0 or args.task_idx >= len(targets):
        log.error("task_idx %d out of range [0, %d) -- known targets: %s",
                  args.task_idx, len(targets), targets)
        return 2

    target = targets[args.task_idx]

    log_dir = RESULTS_DIR / "logs" / "shards"
    log_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(log, log_dir / f"ablation_{args.task_idx:04d}__{target}.log")

    # Defer import: ablation_perfold imports back from this module
    # (ABLATION_GRID), so doing it after we've defined the constant avoids
    # any circular-import hazard.
    from evaluation.ablation_perfold import run_shard

    out_path = run_shard(
        target=target,
        shard_idx=args.task_idx,
        manifest_path=manifest_path,
        incremental=not args.no_incremental,
    )
    log.info("ablation shard %d done -> %s", args.task_idx, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
