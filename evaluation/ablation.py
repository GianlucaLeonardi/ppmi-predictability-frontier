"""
Leave-one-modality-out ablation: re-run the best model per target dropping
one modality family at a time and report the metric drop.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    MODALITY_FAMILIES,
    MODELS_REGRESSION,
    MODELS_CLASSIFICATION,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    N_OUTER_FOLDS,
)
from data_preprocessing.build_dataset import (
    create_cv_folds,
)
from evaluation.frontier import load_task, _pop_raw_cols, _get_xy, _normalize_fold
from evaluation.metrics import regression_metrics, classification_metrics, ranking_metrics
from models.ml_models import get_model
from utils.logging_utils import get_logger

log = get_logger("ablation")


def identify_modality_columns(feat_names: List[str], modality_name: str) -> List[int]:
    return [i for i, name in enumerate(feat_names) if name.split("__")[0] == modality_name]


def run_ablation_for_task(
    task_key: str,
    task_meta: Dict,
    model_name: str = "xgboost",
) -> List[Dict]:
    """Run leave-one-modality-out ablation for a single task."""
    full_df = load_task(task_key)
    if full_df is None or len(full_df) < 20:
        return []

    task_type = task_meta.get("task_type", "regression")
    target_col = task_meta.get("target_column", "")
    n_classes = 2
    raw_df = _pop_raw_cols(full_df)
    feat_cols = [c for c in full_df.columns if c not in ("PATNO", "target")]
    y_all = full_df["target"].values.astype(np.float32)
    patnos = full_df["PATNO"].values

    # Single seed, single fold for ablation (speed)
    folds = create_cv_folds(patnos, y_all, task_type, n_folds=N_OUTER_FOLDS, seed=0)
    fold = folds[0]  # Use first fold only for ablation

    train_df = full_df.iloc[fold["train_idx"]].reset_index(drop=True)
    test_df = full_df.iloc[fold["test_idx"]].reset_index(drop=True)

    train_normed, test_normed, keep_cols = _normalize_fold(train_df, test_df, feat_cols)
    X_train, X_train_nan, y_train, fn = _get_xy(train_normed)
    X_test, X_test_nan, y_test, _ = _get_xy(test_normed)

    is_xgb = model_name in ("xgboost", "xgboost_clf")
    # Ranking tasks reuse the regression model families (they predict a
    # continuous rank value and are scored by Spearman/Kendall tau).
    model_configs = (
        MODELS_REGRESSION
        if task_type in ("regression", "ranking")
        else MODELS_CLASSIFICATION
    )

    # Full model baseline
    model_cfg = model_configs.get(model_name, {})
    model = get_model(model_name, task_type=task_type, config=model_cfg)
    if is_xgb:
        model.fit(X_train_nan, y_train)
        y_pred_full = model.predict(X_test_nan)
    else:
        model.fit(X_train, y_train)
        y_pred_full = model.predict(X_test)

    if task_type == "regression":
        baseline_metrics = regression_metrics(y_test, y_pred_full)
    elif task_type == "ranking":
        baseline_metrics = ranking_metrics(y_test, y_pred_full)
    else:
        y_prob_full = None
        if hasattr(model, "predict_proba"):
            X_proba = X_test_nan if is_xgb else X_test
            y_prob_full = model.predict_proba(X_proba)
        baseline_metrics = classification_metrics(y_test, y_pred_full, y_prob_full, n_classes=n_classes)

    results = []
    for mf in MODALITY_FAMILIES:
        drop_indices = identify_modality_columns(fn, mf.name)
        if not drop_indices:
            continue
        keep_indices = [i for i in range(X_train.shape[1]) if i not in drop_indices]
        if not keep_indices:
            continue

        try:
            model = get_model(model_name, task_type=task_type, config=model_cfg)
            if is_xgb:
                model.fit(X_train_nan[:, keep_indices], y_train)
                y_pred_abl = model.predict(X_test_nan[:, keep_indices])
            else:
                model.fit(X_train[:, keep_indices], y_train)
                y_pred_abl = model.predict(X_test[:, keep_indices])

            if task_type == "regression":
                abl_metrics = regression_metrics(y_test, y_pred_abl)
            elif task_type == "ranking":
                abl_metrics = ranking_metrics(y_test, y_pred_abl)
            else:
                y_prob_abl = None
                if hasattr(model, "predict_proba"):
                    X_proba_abl = X_test_nan[:, keep_indices] if is_xgb else X_test[:, keep_indices]
                    y_prob_abl = model.predict_proba(X_proba_abl)
                abl_metrics = classification_metrics(y_test, y_pred_abl, y_prob_abl, n_classes=n_classes)
        except Exception as e:
            log.warning("Ablation failed for %s / drop %s: %s", task_key, mf.name, e)
            continue

        # Report appropriate metrics per task_type.
        # Convention for `delta`: positive = dropping the modality HURTS
        # (i.e., lowers a higher-is-better metric, or raises an error metric).
        if task_type == "regression":
            metric_names = ["r2", "spearman", "mae", "rmse"]
            higher_is_better = {"r2", "spearman"}
        elif task_type == "ranking":
            metric_names = ["spearman", "kendall_tau"]
            higher_is_better = {"spearman", "kendall_tau"}
        else:  # classification
            metric_names = ["mcc", "balanced_accuracy", "f1"]
            higher_is_better = {"mcc", "balanced_accuracy", "f1"}

        for metric_name in metric_names:
            bv = baseline_metrics.get(metric_name, float("nan"))
            av = abl_metrics.get(metric_name, float("nan"))
            delta = (bv - av) if metric_name in higher_is_better else (av - bv)
            results.append({
                "task_key": task_key,
                "target": task_meta["target"],
                "target_display": task_meta["target_display"],
                "target_domain": task_meta["target_domain"],
                "horizon": task_meta["horizon"],
                "horizon_months": task_meta["horizon_months"],
                "regime": task_meta["regime"],
                "task_type": task_type,
                "model": model_name,
                "dropped_modality": mf.name,
                "dropped_modality_display": mf.display,
                "n_dropped_features": len(drop_indices),
                "metric": metric_name,
                f"full_{metric_name}": bv,
                f"ablated_{metric_name}": av,
                f"delta_{metric_name}": delta,
            })

    return results


def run_ablation_suite(
    manifest_path: Optional[Path] = None,
    regime: str = "baseline_only",
    horizon: str = "V08",
    model_name: str = "xgboost",
) -> pd.DataFrame:
    if manifest_path is None:
        manifest_path = PROCESSED_DATA_DIR / "task_manifest.csv"
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        log.error("Ablation: task_manifest not found at %s — skipping suite", manifest_path)
        return pd.DataFrame()

    manifest = pd.read_csv(manifest_path)
    subset = manifest[
        (manifest["regime"] == regime) &
        (manifest["horizon"] == horizon)
    ]
    log.info("Running ablation for %d tasks (regime=%s, horizon=%s)", len(subset), regime, horizon)

    # Model selection per task_type: regression/ranking -> regression model,
    # classification -> xgboost_clf.
    def _model_for(tt: str) -> str:
        if tt == "classification":
            return "xgboost_clf"
        if tt == "ranking":
            return "xgboost"
        return model_name  # regression (or unknown → regression default)

    all_results = []
    n_failed = 0
    for _, row in subset.iterrows():
        tt = row.get("task_type", "regression")
        mn = _model_for(tt)
        task_key = row["task_key"]
        log.info("  Ablation: %s (%s, model=%s)", task_key, tt, mn)
        try:
            results = run_ablation_for_task(task_key, row.to_dict(), mn)
            all_results.extend(results)
        except Exception as e:
            # Never let one bad task kill the whole suite — the ablation
            # summary is aggregated across all retained tasks, so a partial
            # table is still useful.
            n_failed += 1
            log.warning("Ablation failed for %s (%s, model=%s): %s",
                        task_key, tt, mn, e)
            continue

    if n_failed:
        log.warning("Ablation suite %s/%s: %d of %d tasks failed",
                    regime, horizon, n_failed, len(subset))

    ablation_df = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / f"ablation_{regime}_{horizon}_{model_name}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(out_path, index=False)
    log.info("Saved ablation results: %s (%d rows)", out_path, len(ablation_df))
    return ablation_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", default="baseline_only")
    parser.add_argument("--horizon", default="V08")
    parser.add_argument("--model", default="xgboost")
    args = parser.parse_args()
    run_ablation_suite(regime=args.regime, horizon=args.horizon, model_name=args.model)
