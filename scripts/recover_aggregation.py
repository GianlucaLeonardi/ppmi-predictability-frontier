#!/usr/bin/env python3
"""
Recover `frontier_results.csv` + `r2_spearman_divergences.csv` from the
already-saved `frontier_results_per_fold.csv`, without re-running Step 2.

Use this when Step 2 (frontier benchmark) completed and saved the per-fold
CSV but crashed during the aggregation step. Safe to run repeatedly.

Usage:
    cd /path/to/ppmi-predictability-frontier
    python scripts/recover_aggregation.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import RESULTS_DIR
from evaluation.frontier import aggregate_cv_results


def main():
    fold_path = RESULTS_DIR / "frontier_results_per_fold.csv"
    if not fold_path.exists():
        print(f"ERROR: {fold_path} not found. You must run Step 2 first.")
        sys.exit(1)

    print(f"Loading per-fold results from {fold_path} ...")
    fold_df = pd.read_csv(fold_path)
    print(f"Loaded {len(fold_df)} rows across {fold_df['task_key'].nunique()} tasks")

    print("Aggregating across folds/seeds ...")
    agg_df = aggregate_cv_results(fold_df)
    agg_path = RESULTS_DIR / "frontier_results.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved: {agg_path} ({len(agg_df)} rows)")

    # R²/Spearman divergences (mirrors the block in run_frontier)
    if not agg_df.empty and "r2" in agg_df.columns and "spearman" in agg_df.columns:
        from evaluation.metrics import detect_r2_spearman_divergence
        divergences = detect_r2_spearman_divergence(agg_df)
        if divergences:
            div_df = pd.DataFrame(divergences)
            div_path = RESULTS_DIR / "r2_spearman_divergences.csv"
            div_df.to_csv(div_path, index=False)
            print(f"Saved: {div_path} ({len(divergences)} divergence cases)")

    if "error" in fold_df.columns:
        n_err = fold_df["error"].notna().sum()
        if n_err > 0:
            print(f"NOTE: {n_err} model fits had errors (non-fatal).")

    print("\nDone. You can now run downstream steps:")
    print("  python run_all.py --step 3")


if __name__ == "__main__":
    main()
