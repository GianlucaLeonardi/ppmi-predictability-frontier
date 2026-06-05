#!/usr/bin/env python3
"""
Sensitivity analysis: CV split stability across random seeds, for regression models.

Usage:
    python scripts/sensitivity_splits.py               # 6 representative tasks (~10 min)
    python scripts/sensitivity_splits.py --all-tasks   # all tasks (~3-5 hours)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    CV_SEEDS,
    MODELS_REGRESSION,
    MIN_NON_NAN_FRAC,
    N_OUTER_FOLDS,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    TARGETS,
    VISIT_SCHEDULE,
)
from data_preprocessing.build_dataset import (
    ColumnNormalizer,
    create_cv_folds,
    sanity_check_no_leakage,
)
from evaluation.frontier import load_task, _pop_raw_cols
from evaluation.metrics import regression_metrics
from models.ml_models import get_model
from utils.logging_utils import get_logger

log = get_logger("sensitivity_splits")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use the main pipeline CV_SEEDS, extended to >= 5 distinct seeds for stability.
_EXTRA_SEEDS = [123, 456, 789, 1024, 7, 2024]
SEEDS = list(CV_SEEDS) + [s for s in _EXTRA_SEEDS if s not in CV_SEEDS]
SEEDS = SEEDS[:max(5, len(CV_SEEDS))]

REPRESENTATIVE_TASKS = [
    # task_key values (target__regime__horizon); regression tasks only
    "updrs3_total__baseline_only__V08",
    "moca_total__baseline_only__V08",
    "updrs1_total__baseline_only__V08",
    "ortho_sys__baseline_only__V08",
    "moca_delayed__baseline_only__V08",
    "updrs3_pigd__baseline_only__V08",
]

MODEL_NAMES = list(MODELS_REGRESSION.keys())  # all 7 regression models


def _get_xy(df: pd.DataFrame):
    """Extract feature matrix X, target vector y, and feature column names."""
    feat_cols = [c for c in df.columns if c not in ("PATNO", "target")
                 and not c.startswith("__raw__")]
    X_nan = df[feat_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    X = np.nan_to_num(X_nan.copy(), nan=0.0)
    return X, X_nan, y, feat_cols


def _normalize_fold(train_df, test_df, feat_cols):
    """Fit normalizer on train, transform both. Drop high-missing columns."""
    missing_fracs = train_df[feat_cols].isna().mean()
    keep_cols = missing_fracs[missing_fracs < (1 - MIN_NON_NAN_FRAC)].index.tolist()

    normalizer = ColumnNormalizer()
    normalizer.fit(train_df[keep_cols])

    train_normed = normalizer.transform(train_df[keep_cols])
    test_normed = normalizer.transform(test_df[keep_cols])

    train_out = pd.concat([
        train_df[["PATNO", "target"]].reset_index(drop=True),
        train_normed.reset_index(drop=True),
    ], axis=1)
    test_out = pd.concat([
        test_df[["PATNO", "target"]].reset_index(drop=True),
        test_normed.reset_index(drop=True),
    ], axis=1)
    return train_out, test_out, keep_cols


def _load_all_tasks_from_manifest():
    """Load all regression tasks from the task manifest."""
    manifest_path = PROCESSED_DATA_DIR / "task_manifest.csv"
    if not manifest_path.exists():
        log.warning("task_manifest.csv not found; falling back to representative tasks")
        return None
    manifest = pd.read_csv(manifest_path)
    reg_tasks = manifest[manifest["task_type"] == "regression"]
    return reg_tasks["task_key"].tolist()


def _target_spec_by_name(name: str):
    """Look up a TargetSpec by short name."""
    for tspec in TARGETS:
        if tspec.name == name:
            return tspec
    return None


def main(all_tasks: bool = False):
    import argparse
    parser = argparse.ArgumentParser(description="Sensitivity split analysis")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Run all tasks (not just representative)")
    args, _ = parser.parse_known_args()
    all_tasks = all_tasks or args.all_tasks

    task_keys = list(REPRESENTATIVE_TASKS)
    if all_tasks:
        loaded = _load_all_tasks_from_manifest()
        if loaded:
            task_keys = loaded
            log.info("Full sensitivity mode: %d tasks from manifest", len(task_keys))

    t_start = time.time()
    log.info("=== Sensitivity analysis: CV split stability (v2) ===")
    log.info("Seeds: %s", SEEDS)
    log.info("Tasks: %d tasks (%s mode)", len(task_keys),
             "all" if all_tasks else "representative")
    log.info("Models: %s", MODEL_NAMES)
    log.info("CV: %d-fold per seed", N_OUTER_FOLDS)

    all_rows = []
    total_fits = len(SEEDS) * len(task_keys) * len(MODEL_NAMES) * N_OUTER_FOLDS
    fit_count = 0

    for task_key in task_keys:
        # Load preprocessed full task dataset
        full_df = load_task(task_key)
        if full_df is None or len(full_df) < 20:
            log.warning("  SKIP %s: insufficient data", task_key)
            continue

        # Parse task_key for target info
        parts = task_key.split("__")
        target_name = parts[0] if len(parts) >= 1 else ""
        horizon = parts[2] if len(parts) >= 3 else ""
        tspec = _target_spec_by_name(target_name)
        task_type = tspec.task_type if tspec else "regression"

        # Separate raw columns
        raw_df = _pop_raw_cols(full_df)
        feat_cols = [c for c in full_df.columns if c not in ("PATNO", "target")]
        sanity_check_no_leakage(feat_cols, horizon)

        patnos = full_df["PATNO"].values
        y_all = full_df["target"].values.astype(np.float32)

        for seed in SEEDS:
            # Create CV folds with this seed (same protocol as main pipeline)
            folds = create_cv_folds(
                patnos, y_all, task_type,
                n_folds=N_OUTER_FOLDS, seed=seed,
            )

            for fold_idx, fold in enumerate(folds):
                train_idx = fold["train_idx"]
                test_idx = fold["test_idx"]

                train_df = full_df.iloc[train_idx].reset_index(drop=True)
                test_df = full_df.iloc[test_idx].reset_index(drop=True)
                raw_train = raw_df.iloc[train_idx].reset_index(drop=True) if not raw_df.empty else pd.DataFrame()
                raw_test = raw_df.iloc[test_idx].reset_index(drop=True) if not raw_df.empty else pd.DataFrame()

                # Normalize (fit on train fold only)
                train_normed, test_normed, keep_feat_cols = _normalize_fold(
                    train_df, test_df, feat_cols
                )

                X_train, X_train_nan, y_train, fn = _get_xy(train_normed)
                X_test, X_test_nan, y_test, _ = _get_xy(test_normed)
                n_test = len(y_test)
                if n_test < 5:
                    continue

                for model_name in MODEL_NAMES:
                    fit_count += 1
                    model_cfg = MODELS_REGRESSION.get(model_name, {})
                    t0 = time.time()

                    try:
                        model = get_model(model_name, task_type="regression",
                                          config=model_cfg)

                        if model_name == "locf":
                            model.fit(y_train)
                            y_pred = model.predict(raw_test)
                        elif model_name == "lme":
                            target_col = tspec.column if tspec else ""
                            horizon_months = float(VISIT_SCHEDULE.get(horizon, 36))
                            model.fit(X_train, y_train, feature_names=fn,
                                      target_column=target_col,
                                      horizon_months=horizon_months,
                                      X_train_raw=X_train_nan,
                                      raw_target_train=raw_train)
                            y_pred = model.predict(X_test, X_test_raw=X_test_nan,
                                                   raw_target_test=raw_test)
                        elif model_name == "xgboost":
                            model.fit(X_train_nan, y_train)
                            y_pred = model.predict(X_test_nan)
                        elif model_name == "population_mean":
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        elapsed = time.time() - t0

                        m = regression_metrics(y_test, y_pred)

                        all_rows.append({
                            "seed": seed,
                            "fold": fold_idx,
                            "task_key": task_key,
                            "model": model_name,
                            "r2": m["r2"],
                            "spearman": m["spearman"],
                            "mae": m["mae"],
                            "n_test": n_test,
                        })

                        if fit_count % 100 == 0:
                            log.info("  [%d/%d] seed=%d fold=%d %s %s  "
                                     "r2=%.3f  spearman=%.3f  (%.1fs)",
                                     fit_count, total_fits, seed, fold_idx,
                                     task_key, model_name,
                                     m["r2"], m["spearman"], elapsed)

                    except Exception as e:
                        log.error("  FAILED seed=%d fold=%d %s %s: %s",
                                  seed, fold_idx, task_key, model_name, e)
                        all_rows.append({
                            "seed": seed,
                            "fold": fold_idx,
                            "task_key": task_key,
                            "model": model_name,
                            "r2": float("nan"),
                            "spearman": float("nan"),
                            "mae": float("nan"),
                            "n_test": n_test,
                        })

    # ------------------------------------------------------------------
    # Save per-seed per-fold results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    out_path = TABLES_DIR / "sensitivity_splits.csv"
    results_df.to_csv(out_path, index=False)
    log.info("Saved per-seed/fold results to %s (%d rows)", out_path, len(results_df))

    # ------------------------------------------------------------------
    # Compute and save summary (mean, std, CV across seeds×folds per task)
    # ------------------------------------------------------------------
    summary_rows = []
    for (task_key, model_name), grp in results_df.groupby(["task_key", "model"]):
        r2 = grp["r2"].dropna()
        sp = grp["spearman"].dropna()

        def _cv(vals):
            """Coefficient of variation."""
            m = vals.mean()
            s = vals.std()
            return (s / abs(m)) if (abs(m) > 1e-10 and len(vals) > 1) else float("nan")

        summary_rows.append({
            "task_key": task_key,
            "model": model_name,
            "n_evals": len(grp),
            "r2_mean": round(r2.mean(), 4) if len(r2) > 0 else float("nan"),
            "r2_std": round(r2.std(), 4) if len(r2) > 1 else float("nan"),
            "r2_cv": round(_cv(r2), 4) if len(r2) > 1 else float("nan"),
            "spearman_mean": round(sp.mean(), 4) if len(sp) > 0 else float("nan"),
            "spearman_std": round(sp.std(), 4) if len(sp) > 1 else float("nan"),
            "spearman_cv": round(_cv(sp), 4) if len(sp) > 1 else float("nan"),
            "mae_mean": round(grp["mae"].mean(), 4),
            "mae_std": round(grp["mae"].std(), 4) if len(grp) > 1 else float("nan"),
            "n_test_mean": round(grp["n_test"].mean(), 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES_DIR / "sensitivity_splits_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Saved summary to %s (%d rows)", summary_path, len(summary_df))

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t_start
    log.info("=== Sensitivity analysis complete in %.1f s ===", elapsed_total)
    log.info("\nSummary (R^2 and Spearman across %d seeds x %d folds):",
             len(SEEDS), N_OUTER_FOLDS)
    log.info("%-45s %-12s %8s %8s %8s %8s",
             "task_key", "model", "r2_mean", "r2_std", "sp_mean", "sp_std")
    log.info("-" * 100)
    for _, row in summary_df.iterrows():
        log.info("%-45s %-12s %8.4f %8.4f %8.4f %8.4f",
                 row["task_key"], row["model"],
                 row["r2_mean"], row["r2_std"],
                 row["spearman_mean"], row["spearman_std"])

    return results_df, summary_df


if __name__ == "__main__":
    main()
