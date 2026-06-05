#!/usr/bin/env python3
"""
SHAP feature importance analysis: retrains XGBoost on representative tasks and
computes SHAP values.

Outputs:
    results/tables/shap_importance.csv       -- top features per task
    results/figures/shap_summary_*.pdf/png   -- SHAP summary plots
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    MODELS_REGRESSION,
    FIGURES_DIR,
    TABLES_DIR,
)
from evaluation.frontier import load_task, _pop_raw_cols
from data_preprocessing.build_dataset import create_cv_folds
from models.ml_models import get_model
from utils.logging_utils import get_logger

log = get_logger("shap_analysis")

try:
    import shap
    HAS_SHAP = True

    # Monkey-patch SHAP to handle XGBoost >= 2.0 base_score format.
    import shap.explainers._tree as _shap_tree
    _orig_xgb_init = _shap_tree.XGBTreeModelLoader.__init__

    def _safe_xgb_init(self, xgb_model):
        _builtin_float = float
        def _safe_float(x):
            if isinstance(x, str) and ("[" in x or "]" in x):
                return _builtin_float(x.strip("[]"))
            return _builtin_float(x)
        import builtins
        builtins.float = _safe_float
        try:
            _orig_xgb_init(self, xgb_model)
        finally:
            builtins.float = _builtin_float

    _shap_tree.XGBTreeModelLoader.__init__ = _safe_xgb_init

except ImportError:
    HAS_SHAP = False

SHAP_TASKS = [
    ("updrs3_total__baseline_only__V08",   "regression"),
    ("moca_total__baseline_only__V08",     "regression"),
    ("updrs1_total__baseline_only__V08",   "regression"),
    ("updrs3_total__rolling__V08",         "regression"),
    ("moca_total__rolling__V08",           "regression"),
]

N_TOP_FEATURES = 20


def run_shap_analysis():
    """Compute SHAP values for representative tasks."""
    if not HAS_SHAP:
        log.warning("shap package not installed; skipping SHAP analysis.")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_importance_rows = []
    t_start = time.time()

    for task_key, task_type in SHAP_TASKS:
        log.info("--- SHAP: %s ---", task_key)

        full_df = load_task(task_key)
        if full_df is None or len(full_df) < 20:
            log.warning("  SKIP %s: insufficient data", task_key)
            continue

        # Remove raw columns
        _pop_raw_cols(full_df)

        feat_cols = [c for c in full_df.columns if c not in ("PATNO", "target")]
        y_all = full_df["target"].values.astype(np.float32)
        patnos = full_df["PATNO"].values

        # Use first fold of first seed for SHAP (deterministic, representative)
        folds = create_cv_folds(patnos, y_all, task_type, n_folds=5, seed=0)
        fold = folds[0]

        train_df = full_df.iloc[fold["train_idx"]].reset_index(drop=True)
        test_df = full_df.iloc[fold["test_idx"]].reset_index(drop=True)

        # Normalize per fold
        from evaluation.frontier import _normalize_fold, _get_xy
        train_normed, test_normed, keep_cols = _normalize_fold(train_df, test_df, feat_cols)

        X_train_imp, X_train_nan, y_train, feat_names = _get_xy(train_normed)
        X_test_imp, X_test_nan, y_test, _ = _get_xy(test_normed)

        # Train XGBoost on NaN matrices
        model_cfg = MODELS_REGRESSION.get("xgboost", {})
        model = get_model("xgboost", task_type=task_type, config=model_cfg)
        model.fit(X_train_nan, y_train)

        try:
            inner_model = getattr(model, "model_", model)
            explainer = shap.TreeExplainer(inner_model)
            shap_values = explainer.shap_values(X_test_nan)

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(mean_abs_shap)[-N_TOP_FEATURES:][::-1]
            for rank, idx in enumerate(top_idx, 1):
                all_importance_rows.append({
                    "task_key": task_key,
                    "rank": rank,
                    "feature": feat_names[idx],
                    "mean_abs_shap": float(mean_abs_shap[idx]),
                })

            log.info("  Top 5: %s",
                     [(feat_names[i], f"{mean_abs_shap[i]:.4f}") for i in top_idx[:5]])

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                safe_key = task_key.replace("__", "_")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_nan, feature_names=feat_names,
                                  max_display=N_TOP_FEATURES, show=False)
                fig = plt.gcf()
                fig.tight_layout()
                fig.savefig(FIGURES_DIR / f"shap_summary_{safe_key}.pdf", dpi=300, bbox_inches="tight")
                fig.savefig(FIGURES_DIR / f"shap_summary_{safe_key}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                log.warning("  SHAP figure failed for %s: %s", task_key, e)

        except Exception as e:
            log.warning("  SHAP computation failed for %s: %s", task_key, e)
            continue

    if all_importance_rows:
        imp_df = pd.DataFrame(all_importance_rows)
        imp_df.to_csv(TABLES_DIR / "shap_importance.csv", index=False)
        log.info("Saved SHAP importance: %d rows", len(imp_df))

    log.info("SHAP analysis complete in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    run_shap_analysis()
