#!/usr/bin/env python3
"""
LEDD sensitivity analysis: compares prediction tasks with and without LEDD as a
longitudinal feature, using 5-fold CV across cross-modal regimes.

Usage:
    python scripts/sensitivity_ledd.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    COHORT_PRIMARY,
    MODELS_REGRESSION,
    MIN_NON_NAN_FRAC,
    MODALITY_FAMILIES,
    N_OUTER_FOLDS,
    SEED,
    TABLES_DIR,
    TARGETS,
)
from data_preprocessing.build_dataset import (
    ColumnNormalizer,
    construct_task,
    create_cv_folds,
    load_raw_longitudinal,
    sanity_check_no_leakage,
)
from evaluation.metrics import regression_metrics
from models.ml_models import get_model
from utils.io import get_cohort_patnos, load_ledd_longitudinal, load_static_modality
from utils.logging_utils import get_logger

log = get_logger("sensitivity_ledd")

# Representative tasks: cross-modal regimes where LEDD can appear
TASKS = [
    # (target_name, regime_name, horizon)
    ("updrs3_total",   "baseline_plus_12m", "V08"),
    ("updrs3_total",   "rolling",           "V08"),
    ("moca_total",     "baseline_plus_12m", "V08"),
    ("moca_total",     "rolling",           "V08"),
    ("updrs1_total",   "baseline_plus_12m", "V08"),
    ("updrs1_total",   "rolling",           "V08"),
    ("updrs3_total",   "rolling",           "V10"),
    ("moca_total",     "rolling",           "V10"),
]

MODEL_NAMES = ["xgboost", "locf"]


def _target_spec_by_name(name: str):
    for tspec in TARGETS:
        if tspec.name == name:
            return tspec
    raise ValueError(f"Unknown target: {name}")


def _get_xy(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in ("PATNO", "target")
                 and not c.startswith("__raw__")]
    X_nan = df[feat_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    X = np.nan_to_num(X_nan.copy(), nan=0.0)
    return X, X_nan, y, feat_cols


def _pop_raw_cols(df: pd.DataFrame) -> pd.DataFrame:
    raw_cols = sorted([c for c in df.columns if c.startswith("__raw__")])
    if not raw_cols:
        return pd.DataFrame(index=df.index)
    raw_df = df[raw_cols].copy()
    df.drop(columns=raw_cols, inplace=True)
    return raw_df


def _regime_visits(regime_name: str):
    from configs.config import REGIMES
    for r in REGIMES:
        if r.name == regime_name:
            return r.history_visits
    return None


def _build_full_task(longitudinal, static_all, tspec, regime_name, horizon,
                     cohort_patnos):
    """Build a full-cohort task dataset (no split filtering)."""
    rv = _regime_visits(regime_name)
    task_df = construct_task(
        longitudinal=longitudinal,
        static=static_all,
        target_col=tspec.column,
        target_modality=tspec.source_table,
        horizon_visit=horizon,
        regime_name=regime_name,
        regime_visits=rv,
        cohort_patnos=cohort_patnos,
    )
    return task_df


def _run_cv_for_task(full_df, tspec, horizon, seed=SEED):
    """Run 5-fold CV on a single task dataset. Returns list of per-fold results."""
    if full_df is None or len(full_df) < 20:
        return []

    raw_df = _pop_raw_cols(full_df)
    feat_cols = [c for c in full_df.columns if c not in ("PATNO", "target")]
    sanity_check_no_leakage(feat_cols, horizon)

    patnos = full_df["PATNO"].values
    y_all = full_df["target"].values.astype(np.float32)
    task_type = tspec.task_type

    folds = create_cv_folds(
        patnos, y_all, task_type,
        n_folds=N_OUTER_FOLDS, seed=seed,
    )

    fold_results = []
    for fold_idx, fold in enumerate(folds):
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]

        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        test_df = full_df.iloc[test_idx].reset_index(drop=True)
        raw_train = raw_df.iloc[train_idx].reset_index(drop=True) if not raw_df.empty else pd.DataFrame()
        raw_test = raw_df.iloc[test_idx].reset_index(drop=True) if not raw_df.empty else pd.DataFrame()

        # Normalize
        missing_fracs = train_df[feat_cols].isna().mean()
        keep_cols = missing_fracs[missing_fracs < (1 - MIN_NON_NAN_FRAC)].index.tolist()

        normalizer = ColumnNormalizer()
        normalizer.fit(train_df[keep_cols])

        train_normed = pd.concat([
            train_df[["PATNO", "target"]].reset_index(drop=True),
            normalizer.transform(train_df[keep_cols]).reset_index(drop=True),
        ], axis=1)
        test_normed = pd.concat([
            test_df[["PATNO", "target"]].reset_index(drop=True),
            normalizer.transform(test_df[keep_cols]).reset_index(drop=True),
        ], axis=1)

        X_train, X_train_nan, y_train, fn = _get_xy(train_normed)
        X_test, X_test_nan, y_test, _ = _get_xy(test_normed)
        n_test = len(y_test)
        if n_test < 5:
            continue

        n_ledd_feats = sum(1 for f in fn if f.startswith("ledd__"))

        for model_name in MODEL_NAMES:
            model_cfg = MODELS_REGRESSION.get(model_name, {})
            try:
                model = get_model(model_name, task_type="regression", config=model_cfg)
                if model_name == "locf":
                    model.fit(y_train)
                    y_pred = model.predict(raw_test)
                elif model_name == "xgboost":
                    model.fit(X_train_nan, y_train)
                    y_pred = model.predict(X_test_nan)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                m = regression_metrics(y_test, y_pred)
                fold_results.append({
                    "fold": fold_idx,
                    "model": model_name,
                    "r2": m["r2"],
                    "spearman": m["spearman"],
                    "mae": m["mae"],
                    "n_test": n_test,
                    "n_features": len(fn),
                    "n_ledd_features": n_ledd_feats,
                })
            except Exception as e:
                log.error("  FAILED fold=%d %s: %s", fold_idx, model_name, e)

    return fold_results


def main():
    t_start = time.time()
    log.info("=== LEDD sensitivity analysis (v2, CV-based) ===")

    # 1. Load cohort
    pd_patnos = get_cohort_patnos(COHORT_PRIMARY)

    # 2. Load longitudinal modalities (WITHOUT LEDD)
    log.info("Loading longitudinal modalities (without LEDD)...")
    long_modalities = ["updrs3", "updrs1", "moca", "vital_signs"]
    longitudinal_base = {}
    for mod in long_modalities:
        df = load_raw_longitudinal(mod)
        df = df[df["PATNO"].isin(pd_patnos)]
        longitudinal_base[mod] = df

    # 3. Load LEDD separately
    log.info("Loading LEDD...")
    ledd_df = load_ledd_longitudinal(pd_patnos)
    if ledd_df.empty:
        log.error("No LEDD data available; aborting sensitivity analysis")
        return

    # 4. Build longitudinal dict WITH LEDD
    longitudinal_with_ledd = dict(longitudinal_base)
    longitudinal_with_ledd["ledd"] = ledd_df

    # 5. Load static modalities
    static_frames = []
    for mf in MODALITY_FAMILIES:
        if mf.kind != "static":
            continue
        df = load_static_modality(mf.source_key)
        if df.empty:
            continue
        df = df[df["PATNO"].isin(pd_patnos)]
        rename = {c: f"{mf.name}__{c}" for c in df.columns if c != "PATNO"}
        df = df.rename(columns=rename)
        static_frames.append(df)
    static_all = None
    if static_frames:
        static_all = static_frames[0]
        for sf in static_frames[1:]:
            static_all = static_all.merge(sf, on="PATNO", how="outer")
        static_all = static_all[static_all["PATNO"].isin(pd_patnos)]

    # 6. Run side-by-side CV comparison
    all_rows = []
    for target_name, regime_name, horizon in TASKS:
        tspec = _target_spec_by_name(target_name)
        task_key = f"{target_name}__{regime_name}__{horizon}"
        log.info("Task: %s", task_key)

        for ledd_label, long_dict in [("without_ledd", longitudinal_base),
                                       ("with_ledd", longitudinal_with_ledd)]:
            # Build full task dataset from scratch (with or without LEDD)
            full_df = _build_full_task(
                long_dict, static_all, tspec, regime_name, horizon, pd_patnos
            )
            if full_df is None or len(full_df) < 20:
                log.warning("  SKIP %s (%s): insufficient data", task_key, ledd_label)
                continue

            # Run CV
            fold_results = _run_cv_for_task(full_df, tspec, horizon)
            for r in fold_results:
                r["task_key"] = task_key
                r["target"] = target_name
                r["regime"] = regime_name
                r["horizon"] = horizon
                r["ledd_condition"] = ledd_label
                all_rows.append(r)

    # 7. Save results
    results_df = pd.DataFrame(all_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(TABLES_DIR / "sensitivity_ledd.csv", index=False)
    log.info("Saved per-fold results: %d rows", len(results_df))

    # 8. Summary: compare with vs without LEDD (aggregated across folds)
    if not results_df.empty:
        summary_rows = []
        for (task_key, model), grp in results_df.groupby(["task_key", "model"]):
            without = grp[grp["ledd_condition"] == "without_ledd"]
            with_l = grp[grp["ledd_condition"] == "with_ledd"]
            if without.empty or with_l.empty:
                continue

            r2_without = without["r2"].mean()
            r2_with = with_l["r2"].mean()
            sp_without = without["spearman"].mean()
            sp_with = with_l["spearman"].mean()

            summary_rows.append({
                "task_key": task_key,
                "model": model,
                "r2_without": round(r2_without, 4),
                "r2_with": round(r2_with, 4),
                "delta_r2": round(r2_with - r2_without, 4),
                "spearman_without": round(sp_without, 4),
                "spearman_with": round(sp_with, 4),
                "delta_spearman": round(sp_with - sp_without, 4),
                "n_ledd_features": int(with_l["n_ledd_features"].iloc[0]),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(TABLES_DIR / "sensitivity_ledd_summary.csv", index=False)

        # Log summary
        log.info("\n=== LEDD sensitivity summary (CV-averaged) ===")
        log.info("%-45s %-10s %8s %8s %8s %8s %8s %8s",
                 "task", "model",
                 "r2_w/o", "r2_with", "Δr2",
                 "sp_w/o", "sp_with", "Δsp")
        log.info("-" * 115)
        for _, r in summary_df.iterrows():
            log.info("%-45s %-10s %8.4f %8.4f %+8.4f %8.4f %8.4f %+8.4f",
                     r["task_key"], r["model"],
                     r["r2_without"], r["r2_with"], r["delta_r2"],
                     r["spearman_without"], r["spearman_with"], r["delta_spearman"])
        mean_delta_r2 = summary_df["delta_r2"].mean()
        max_abs_delta_r2 = summary_df["delta_r2"].abs().max()
        log.info("Mean ΔR²: %+.4f, Max |ΔR²|: %.4f", mean_delta_r2, max_abs_delta_r2)

    elapsed = time.time() - t_start
    log.info("=== LEDD sensitivity complete in %.1fs ===", elapsed)


if __name__ == "__main__":
    main()
