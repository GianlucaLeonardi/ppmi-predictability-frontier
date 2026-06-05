#!/usr/bin/env python3
"""
SHAP-driven feature reduction experiment for XGBoost: retrain on top-k SHAP
features and measure R^2 loss versus the full feature matrix.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    CV_SEEDS,
    FIGURES_DIR,
    N_OUTER_FOLDS,
    N_SEEDS,
    RESULTS_DIR,
    SEED,
    TABLES_DIR,
)
from data_preprocessing.build_dataset import create_cv_folds
from evaluation.frontier import _get_xy, _normalize_fold, _pop_raw_cols, load_task
from evaluation.metrics import regression_metrics
from utils.logging_utils import get_logger

log = get_logger("feature_reduction")

# Match run_shap.py exactly; consistency of feature ranking is load-bearing.
SHAP_TASKS = [
    "updrs3_total__baseline_only__V08",
    "moca_total__baseline_only__V08",
    "updrs3_total__rolling__V08",
    "moca_total__rolling__V08",
]

K_VALUES = [3, 5, 10, 15, 20]


def _load_shap_ranking(task_key: str) -> list:
    """Return the top feature names for `task_key`, ordered by |SHAP|."""
    path = TABLES_DIR / "shap_importance.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"shap_importance.csv not found at {path}; run `python scripts/run_shap.py` first."
        )
    df = pd.read_csv(path)
    sub = df[df["task_key"] == task_key].sort_values("rank")
    if sub.empty:
        return []
    return sub["feature"].tolist()


def _load_hp_for(task_key: str) -> dict:
    """Return the most-frequent XGBoost hyperparameter set across fold-seed fits."""
    path = RESULTS_DIR / "exports" / "hyperparameter_selections.csv"
    if not path.exists():
        return {}
    hp = pd.read_csv(path)
    sub = hp[(hp["task_key"] == task_key) & (hp["model"] == "xgboost")]
    if sub.empty:
        return {}
    from collections import Counter

    configs = []
    for s in sub["best_params"]:
        try:
            d = json.loads(s) if isinstance(s, str) else {}
            configs.append(json.dumps(d, sort_keys=True))
        except Exception:
            continue
    if not configs:
        return {}
    mode_cfg = Counter(configs).most_common(1)[0][0]
    return json.loads(mode_cfg)


def _fit_xgb_topk(X_train_nan, y_train, X_test_nan, hp):
    from xgboost import XGBRegressor

    params = {k: v for k, v in hp.items() if k != "random_state"}
    params.setdefault("tree_method", "hist")
    params.setdefault("random_state", SEED)
    params.setdefault("verbosity", 0)
    model = XGBRegressor(**params)
    model.fit(X_train_nan, y_train)
    return model.predict(X_test_nan)


def run_feature_reduction():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    t_start = time.time()

    for task_key in SHAP_TASKS:
        log.info("--- Feature reduction: %s ---", task_key)
        full_df = load_task(task_key)
        if full_df is None or len(full_df) < 20:
            log.warning("  SKIP %s: insufficient data", task_key)
            continue

        _pop_raw_cols(full_df)
        feat_cols = [c for c in full_df.columns if c not in ("PATNO", "target")]
        y_all = full_df["target"].values.astype(np.float32)
        patnos = full_df["PATNO"].values

        shap_order = _load_shap_ranking(task_key)
        if not shap_order:
            log.warning("  SKIP %s: no SHAP ranking found", task_key)
            continue

        hp = _load_hp_for(task_key)
        if not hp:
            log.warning("  %s: no tuned HP found; using XGBoost library defaults",
                        task_key)

        for seed in CV_SEEDS[:N_SEEDS]:
            folds = create_cv_folds(patnos, y_all, "regression",
                                    n_folds=N_OUTER_FOLDS, seed=int(seed))
            for fold_i, fold in enumerate(folds):
                train_df = full_df.iloc[fold["train_idx"]].reset_index(drop=True)
                test_df = full_df.iloc[fold["test_idx"]].reset_index(drop=True)

                train_normed, test_normed, keep_cols = _normalize_fold(
                    train_df, test_df, feat_cols
                )

                # Full-feature reference (all kept features)
                X_tr_imp, X_tr_nan, y_tr, feat_names = _get_xy(train_normed)
                X_te_imp, X_te_nan, y_te, _ = _get_xy(test_normed)
                try:
                    y_pred_full = _fit_xgb_topk(X_tr_nan, y_tr, X_te_nan, hp)
                    m_full = regression_metrics(y_te, y_pred_full)
                    r2_full = float(m_full.get("r2", float("nan")))
                    spear_full = float(m_full.get("spearman", float("nan")))
                except Exception as e:
                    log.warning("  %s seed=%d fold=%d full-fit failed: %s",
                                task_key, seed, fold_i, e)
                    continue

                for k in K_VALUES:
                    # Pick top-k features from the SHAP ranking that still
                    # survive per-fold missingness filtering (keep_cols).
                    top = [f for f in shap_order if f in keep_cols][:k]
                    if len(top) < 1:
                        continue
                    idxs = [feat_names.index(f) for f in top
                            if f in feat_names]
                    if not idxs:
                        continue
                    X_tr_k = X_tr_nan[:, idxs]
                    X_te_k = X_te_nan[:, idxs]
                    try:
                        y_pred_k = _fit_xgb_topk(X_tr_k, y_tr, X_te_k, hp)
                        m_k = regression_metrics(y_te, y_pred_k)
                    except Exception as e:
                        log.warning("  %s k=%d seed=%d fold=%d fit failed: %s",
                                    task_key, k, seed, fold_i, e)
                        continue
                    rows.append({
                        "task_key": task_key,
                        "seed": int(seed),
                        "fold": int(fold_i),
                        "k": int(k),
                        "n_features_used": int(len(idxs)),
                        "r2": float(m_k.get("r2", float("nan"))),
                        "spearman": float(m_k.get("spearman", float("nan"))),
                        "rmse": float(m_k.get("rmse", float("nan"))),
                        "r2_full": r2_full,
                        "spearman_full": spear_full,
                        "delta_r2": (float(m_k.get("r2", float("nan"))) - r2_full),
                    })

    if not rows:
        log.warning("No feature-reduction rows produced; nothing to write.")
        return

    out = pd.DataFrame(rows)
    out_path = TABLES_DIR / "shap_feature_reduction.csv"
    out.to_csv(out_path, index=False)
    log.info("Saved feature-reduction results (%d rows) -> %s", len(out), out_path)

    # --- Plot curves: R^2 vs k per task, with 95% percentile CI bands --------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        agg = (
            out.groupby(["task_key", "k"])["r2"]
            .agg(["mean",
                  lambda s: float(np.percentile(s, 2.5)),
                  lambda s: float(np.percentile(s, 97.5))])
            .reset_index()
        )
        agg.columns = ["task_key", "k", "mean", "ci_lo", "ci_hi"]
        full_ref = (
            out.groupby("task_key")["r2_full"].mean().to_dict()
        )

        fig, ax = plt.subplots(figsize=(7, 4.5))
        # Explicit per-task colours (local to this figure only).
        TASK_COLORS = {
            "moca_total__baseline_only__V08":   "#0072B2",  # blue
            "moca_total__rolling__V08":         "#009E73",  # green
            "updrs3_total__baseline_only__V08": "#D55E00",  # orange
            "updrs3_total__rolling__V08":       "#CC79A7",  # pink
        }
        _fallback = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999"]
        for i, (task_key, g) in enumerate(agg.groupby("task_key")):
            g = g.sort_values("k")
            color = TASK_COLORS.get(task_key, _fallback[i % len(_fallback)])
            ax.plot(g["k"], g["mean"], marker="o", color=color, label=task_key)
            ax.fill_between(g["k"], g["ci_lo"], g["ci_hi"], color=color, alpha=0.12)
            ref = full_ref.get(task_key, np.nan)
            if np.isfinite(ref):
                ax.axhline(ref, color=color, linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("k (top-|SHAP| features used)")
        ax.set_ylabel(r"R$^2$ (95% CI across 25 folds)")
        ax.set_title("Feature reduction: R² vs. top-k SHAP features (XGBoost)",
                     fontweight="bold")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        fig.tight_layout()
        out_fig = FIGURES_DIR / "feature_reduction_curves.png"
        fig.savefig(out_fig, dpi=200, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "feature_reduction_curves.pdf",
                    bbox_inches="tight")
        plt.close(fig)
        log.info("Saved figure -> %s", out_fig)
    except Exception as e:
        log.warning("Could not render feature_reduction_curves figure: %s", e)

    log.info("Feature reduction complete in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    run_feature_reduction()
