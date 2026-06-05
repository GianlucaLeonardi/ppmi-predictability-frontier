#!/usr/bin/env python3
"""Run the predictability frontier benchmark serially (non-sharded path).

CV dims are driven by configs/config.py (N_OUTER_FOLDS, N_SEEDS). For a
parallel / sharded launch, dispatch the shards across workers with your own
job scheduler.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.frontier import run_frontier

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Limit number of CV seeds (default: N_SEEDS from config)")
    parser.add_argument("--no-predictions", action="store_true")
    args = parser.parse_args()
    run_frontier(
        max_tasks=args.max_tasks,
        max_seeds=args.max_seeds,
        save_predictions=not args.no_predictions,
    )
