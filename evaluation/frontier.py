"""Predictability Frontier benchmark runner: patient-level stratified
repeated CV with nested hyperparameter tuning across all tasks."""

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    MODELS_REGRESSION,
    MODELS_CLASSIFICATION,
    N_OUTER_FOLDS,
    N_SEEDS,
    CV_SEEDS,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    VISIT_SCHEDULE,
)
from data_preprocessing.build_dataset import (
    ColumnNormalizer,
    create_cv_folds,
)
from evaluation.metrics import (
    classification_metrics,
    regression_metrics,
    ranking_metrics,
    target_variance_context,
)
from models.ml_models import get_model
from utils.logging_utils import get_logger

log = get_logger("frontier_runner")


def load_task(task_key: str) -> Optional[pd.DataFrame]:
    """Load the full (unsplit, unnormalized) task dataset."""
    task_dir = PROCESSED_DATA_DIR / "tasks" / task_key
    path = task_dir / "full.csv.gz"
    if path.exists():
        return pd.read_csv(path, compression="gzip")
    return None


def _pop_raw_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and remove __raw__ columns from a split DataFrame."""
    raw_cols = sorted([c for c in df.columns if c.startswith("__raw__")])
    if not raw_cols:
        return pd.DataFrame(index=df.index)
    raw_df = df[raw_cols].copy()
    df.drop(columns=raw_cols, inplace=True)
    return raw_df


def _get_xy(df: pd.DataFrame):
    """Extract feature matrix X, target y, and feature names (with nan->0 imputed X)."""
    feat_cols = [c for c in df.columns if c not in ("PATNO", "target")]
    assert not any(c.startswith("__raw__") for c in feat_cols), \
        f"__raw__ columns in feature matrix: {[c for c in feat_cols if c.startswith('__raw__')]}"
    X = df[feat_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    X_imputed = np.nan_to_num(X, nan=0.0)
    return X_imputed, X, y, feat_cols  # X_imputed, X_with_nan, y, feat_names


def _qc_no_leakage(feat_names: List[str], horizon: str) -> None:
    horizon_months = VISIT_SCHEDULE[horizon]
    for fname in feat_names:
        parts = fname.split("__")
        if len(parts) >= 2 and parts[1] in VISIT_SCHEDULE:
            if VISIT_SCHEDULE[parts[1]] >= horizon_months:
                raise ValueError(
                    f"LEAKAGE: feature '{fname}' from {parts[1]} "
                    f"({VISIT_SCHEDULE[parts[1]]}m) >= horizon {horizon} ({horizon_months}m)"
                )


def _normalize_fold(train_df, test_df, feat_cols, min_non_nan_frac=0.10):
    """Fit normalizer on train fold, transform both, and drop high-missing columns."""
    from configs.config import MIN_NON_NAN_FRAC
    min_non_nan_frac = MIN_NON_NAN_FRAC

    # Drop columns with >90% missing in train
    missing_fracs = train_df[feat_cols].isna().mean()
    keep_cols = missing_fracs[missing_fracs < (1 - min_non_nan_frac)].index.tolist()

    normalizer = ColumnNormalizer()
    normalizer.fit(train_df[keep_cols])

    train_normed = normalizer.transform(train_df[keep_cols])
    test_normed = normalizer.transform(test_df[keep_cols])

    # Reassemble with PATNO and target
    train_out = pd.concat([
        train_df[["PATNO", "target"]].reset_index(drop=True),
        train_normed.reset_index(drop=True),
    ], axis=1)

    test_out = pd.concat([
        test_df[["PATNO", "target"]].reset_index(drop=True),
        test_normed.reset_index(drop=True),
    ], axis=1)

    return train_out, test_out, keep_cols


def run_single_task(
    task_key: str,
    task_meta: Dict[str, Any],
    model_names: Optional[List[str]] = None,
    save_predictions: bool = True,
    max_seeds: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run N_OUTER_FOLDS-fold x N_SEEDS CV for a single task, one result dict per (model, seed, fold)."""
    full_df = load_task(task_key)
    if full_df is None or len(full_df) < 20:
        return []

    task_type = task_meta.get("task_type", "regression")
    target_col = task_meta.get("target_column", "")
    horizon = task_meta.get("horizon", "")

    # Separate raw columns before anything else
    raw_df = _pop_raw_cols(full_df)

    feat_cols = [c for c in full_df.columns if c not in ("PATNO", "target")]
    _qc_no_leakage(feat_cols, horizon)

    n_classes = 2

    # Select model set
    if model_names is None:
        if task_type == "regression":
            model_names = list(MODELS_REGRESSION.keys())
        elif task_type == "classification":
            model_names = list(MODELS_CLASSIFICATION.keys())
        else:  # ranking
            # Ranking targets are derived; only feature-based models apply.
            model_names = [m for m in MODELS_REGRESSION.keys()
                           if m not in ("population_mean", "locf", "lme")]

    # Predictions directory
    pred_dir = RESULTS_DIR / "predictions" / task_key
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    patnos = full_df["PATNO"].values
    y_all = full_df["target"].values.astype(np.float32)

    results = []
    n_seeds = max_seeds if max_seeds is not None else N_SEEDS
    seeds_to_run = CV_SEEDS[:n_seeds]
    model_configs = MODELS_REGRESSION if task_type in ("regression", "ranking") else MODELS_CLASSIFICATION

    for seed_idx in seeds_to_run:
        # Create stratified folds for this seed
        folds = create_cv_folds(
            patnos, y_all, task_type,
            n_folds=N_OUTER_FOLDS, seed=seed_idx,
        )

        for fold_idx, fold in enumerate(folds):
            train_idx = fold["train_idx"]
            test_idx = fold["test_idx"]

            # Split the data
            train_df = full_df.iloc[train_idx].reset_index(drop=True)
            test_df = full_df.iloc[test_idx].reset_index(drop=True)

            # Raw columns for LOCF/LME
            raw_train = raw_df.iloc[train_idx].reset_index(drop=True) if not raw_df.empty else pd.DataFrame()
            raw_test = raw_df.iloc[test_idx].reset_index(drop=True) if not raw_df.empty else pd.DataFrame()

            # Normalize (fit on train fold only)
            train_normed, test_normed, keep_feat_cols = _normalize_fold(
                train_df, test_df, feat_cols
            )

            # Extract X, y
            X_train_imp, X_train_nan, y_train, fn = _get_xy(train_normed)
            X_test_imp, X_test_nan, y_test, _ = _get_xy(test_normed)

            n_test = len(y_test)
            if n_test < 5:
                continue

            var_ctx = target_variance_context(y_test) if task_type == "regression" else {}

            for model_name in model_names:
                model_cfg = model_configs.get(model_name, {})
                t0 = time.time()
                log.info("    seed=%d fold=%d model=%s ...", seed_idx, fold_idx, model_name)
                try:
                    model = get_model(model_name, task_type=task_type, config=model_cfg, seed=seed_idx)

                    # Fit
                    if model_name == "locf":
                        model.fit(y_train)
                        y_pred = model.predict(raw_test)
                    elif model_name == "lme":
                        horizon_months = float(VISIT_SCHEDULE.get(horizon, 36))
                        model.fit(X_train_imp, y_train, feature_names=fn,
                                  target_column=target_col, horizon_months=horizon_months,
                                  X_train_raw=X_train_nan,
                                  raw_target_train=raw_train)
                        y_pred = model.predict(X_test_imp, X_test_raw=X_test_nan,
                                               raw_target_test=raw_test)
                    elif model_name in ("population_mean", "majority_class"):
                        model.fit(X_train_imp, y_train)
                        y_pred = model.predict(X_test_imp)
                    elif model_name in ("xgboost", "xgboost_clf"):
                        # XGBoost handles NaN natively
                        model.fit(X_train_nan, y_train)
                        y_pred = model.predict(X_test_nan)
                    else:
                        model.fit(X_train_imp, y_train)
                        y_pred = model.predict(X_test_imp)
                    elapsed = time.time() - t0

                    # Save predictions per fold
                    if save_predictions:
                        fold_dir = pred_dir / f"seed{seed_idx}_fold{fold_idx}"
                        fold_dir.mkdir(parents=True, exist_ok=True)
                        np.save(fold_dir / "y_test.npy", y_test)
                        np.save(fold_dir / f"{model_name}.npy", y_pred)

                    # Compute metrics
                    if task_type == "classification":
                        X_for_proba = X_test_nan if model_name in ("xgboost_clf",) else X_test_imp
                        y_prob = model.predict_proba(X_for_proba) if hasattr(model, "predict_proba") else None
                        m = classification_metrics(y_test, y_pred, y_prob, n_classes=n_classes)

                        if save_predictions and y_prob is not None:
                            np.save(fold_dir / f"{model_name}_proba.npy", y_prob)
                    elif task_type == "ranking":
                        m = ranking_metrics(y_test, y_pred)
                    else:
                        m = regression_metrics(y_test, y_pred)

                    # CIs are computed in aggregate_cv_results() from the CV distribution.

                    # Store hyperparameters if available
                    best_params = {}
                    if hasattr(model, "best_params_"):
                        best_params = model.best_params_

                    result = {
                        "task_key": task_key,
                        "target": task_meta["target"],
                        "target_display": task_meta["target_display"],
                        "target_domain": task_meta["target_domain"],
                        "task_type": task_type,
                        "horizon": task_meta["horizon"],
                        "horizon_months": task_meta["horizon_months"],
                        "regime": task_meta["regime"],
                        "regime_display": task_meta["regime_display"],
                        "model": model_name,
                        "seed": seed_idx,
                        "fold": fold_idx,
                        "n_train": len(y_train),
                        "n_test": n_test,
                        "n_features": len(fn),
                        "fit_time_s": round(elapsed, 2),
                        **m,
                        **var_ctx,
                        "best_params": json.dumps(best_params) if best_params else "",
                    }
                    results.append(result)

                except Exception as e:
                    log.error("FAILED %s / %s / seed=%d fold=%d: %s",
                              task_key, model_name, seed_idx, fold_idx, e)
                    log.debug(traceback.format_exc())
                    results.append({
                        "task_key": task_key,
                        "target": task_meta["target"],
                        "target_display": task_meta["target_display"],
                        "target_domain": task_meta["target_domain"],
                        "task_type": task_type,
                        "horizon": task_meta["horizon"],
                        "horizon_months": task_meta["horizon_months"],
                        "regime": task_meta["regime"],
                        "regime_display": task_meta["regime_display"],
                        "model": model_name,
                        "seed": seed_idx,
                        "fold": fold_idx,
                        "n_train": len(y_train),
                        "n_test": n_test,
                        "n_features": len(fn),
                        "error": str(e),
                    })

    return results


def aggregate_cv_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold results to per-task-model summaries (mean, std, CI across seeds/folds)."""
    if results_df.empty:
        return results_df

    # Identify metric columns
    meta_cols = ["task_key", "target", "target_display", "target_domain",
                 "task_type", "horizon", "horizon_months", "regime",
                 "regime_display", "model"]
    non_metric = set(meta_cols + ["seed", "fold", "n_train", "n_test",
                                    "n_features", "fit_time_s", "error",
                                    "best_params"])
    metric_prefixes = ("r2", "spearman", "mae", "rmse", "pearson", "auroc",
                       "auprc", "f1", "mcc", "balanced_accuracy", "sensitivity",
                       "specificity", "ppv", "npv", "brier", "accuracy",
                       "prevalence", "kendall_tau")
    metric_cols = [c for c in results_df.columns
                   if c not in non_metric and c.startswith(metric_prefixes)]

    # Filter out error rows (no `error` column when every task succeeded).
    if "error" in results_df.columns:
        valid = results_df[results_df["error"].isna()]
    else:
        valid = results_df
    if valid.empty:
        return pd.DataFrame()

    agg_rows = []
    for keys, grp in valid.groupby(meta_cols):
        row = dict(zip(meta_cols, keys))
        row["n_cv_results"] = len(grp)
        row["n_train_mean"] = grp["n_train"].mean()
        row["n_test_mean"] = grp["n_test"].mean()
        row["n_features"] = grp["n_features"].iloc[0]

        for mc in metric_cols:
            if mc in grp.columns:
                vals = grp[mc].dropna()
                if len(vals) > 0:
                    row[f"{mc}"] = vals.mean()
                    row[f"{mc}_std"] = vals.std()
                    row[f"{mc}_ci_lo"] = np.percentile(vals, 2.5)
                    row[f"{mc}_ci_hi"] = np.percentile(vals, 97.5)

        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def run_frontier(
    manifest_path: Optional[Path] = None,
    model_names: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    max_seeds: Optional[int] = None,
    save_predictions: bool = True,
) -> pd.DataFrame:
    if manifest_path is None:
        manifest_path = PROCESSED_DATA_DIR / "task_manifest.csv"

    manifest = pd.read_csv(manifest_path)
    log.info("Loaded manifest with %d tasks", len(manifest))

    if max_tasks is not None:
        manifest = manifest.head(max_tasks)
        log.info("Limiting to %d tasks (debug mode)", max_tasks)

    all_results = []
    n_tasks = len(manifest)

    for i, (_, row) in enumerate(manifest.iterrows()):
        task_key = row["task_key"]
        task_meta = row.to_dict()

        # Select model names by task type
        task_model_names = model_names
        if task_model_names is None:
            if row["task_type"] == "regression":
                task_model_names = list(MODELS_REGRESSION.keys())
            elif row["task_type"] == "classification":
                task_model_names = list(MODELS_CLASSIFICATION.keys())
            else:  # ranking (see run_single_task for rationale)
                task_model_names = [m for m in MODELS_REGRESSION.keys()
                                     if m not in ("population_mean", "locf", "lme")]

        log.info("[%d/%d] Running %s (%s, %d-fold x %d-seed) ...",
                 i + 1, n_tasks, task_key, row["task_type"],
                 N_OUTER_FOLDS, max_seeds or N_SEEDS)
        results = run_single_task(
            task_key, task_meta, task_model_names,
            save_predictions, max_seeds=max_seeds,
        )
        all_results.extend(results)

        if (i + 1) % 5 == 0:
            log.info("  Progress: %d/%d tasks complete", i + 1, n_tasks)

    # Save per-fold results
    fold_df = pd.DataFrame(all_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_path = RESULTS_DIR / "frontier_results_per_fold.csv"
    save_fold_df = fold_df.copy()
    save_fold_df.to_csv(fold_path, index=False)
    log.info("Saved per-fold results to %s (%d rows)", fold_path, len(save_fold_df))

    # Aggregate across folds/seeds
    agg_df = aggregate_cv_results(fold_df)
    agg_path = RESULTS_DIR / "frontier_results.csv"
    agg_df.to_csv(agg_path, index=False)
    log.info("Saved aggregated results to %s (%d rows)", agg_path, len(agg_df))

    # Detect R^2/Spearman divergences
    if not agg_df.empty and "r2" in agg_df.columns and "spearman" in agg_df.columns:
        from evaluation.metrics import detect_r2_spearman_divergence
        divergences = detect_r2_spearman_divergence(agg_df)
        if divergences:
            div_df = pd.DataFrame(divergences)
            div_path = RESULTS_DIR / "r2_spearman_divergences.csv"
            div_df.to_csv(div_path, index=False)
            log.info("Found %d R^2/Spearman divergence cases -> %s", len(divergences), div_path)

    if "error" in fold_df.columns:
        n_err = fold_df["error"].notna().sum()
        if n_err > 0:
            log.warning("%d model fits had errors", n_err)

    return agg_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Limit number of CV seeds (default: N_SEEDS)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to run (default: all)")
    parser.add_argument("--no-predictions", action="store_true",
                        help="Skip saving prediction arrays")
    args = parser.parse_args()
    run_frontier(
        model_names=args.models,
        max_tasks=args.max_tasks,
        max_seeds=args.max_seeds,
        save_predictions=not args.no_predictions,
    )
