"""
Quality-control checks run after the frontier benchmark.

Checks LOCF scale, sample sizes, non-monotonic horizons, output completeness,
R^2 population, and dimensionality.
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import MIN_N_WARNING, PROCESSED_DATA_DIR, RESULTS_DIR
from utils.logging_utils import get_logger

log = get_logger("qc_checks")


def check_locf_scale(results_df: pd.DataFrame) -> Dict:
    locf = results_df[(results_df["model"] == "locf") & (results_df["task_type"] == "regression")]
    if locf.empty:
        return {"status": "SKIP", "reason": "No LOCF results"}
    failures = []
    for _, row in locf.iterrows():
        r2 = row.get("r2", float("nan"))
        if np.isfinite(r2) and r2 < -2.0:
            failures.append({"task_key": row["task_key"], "r2": r2})
    return {
        "status": "FAIL" if failures else "PASS",
        "n_locf_tasks": len(locf),
        "mean_r2": float(locf["r2"].mean()) if "r2" in locf.columns else None,
        "failures": failures if failures else None,
    }


def check_sample_sizes(results_df: pd.DataFrame) -> Dict:
    n_col = "n_test_mean" if "n_test_mean" in results_df.columns else "n_test"
    if n_col not in results_df.columns:
        return {"status": "SKIP", "reason": "No sample size column"}
    small = results_df[results_df[n_col] < MIN_N_WARNING][
        ["task_key", "target_display", "horizon", "regime", n_col]
    ].drop_duplicates(subset="task_key")
    return {
        "status": "PASS" if small.empty else "WARN",
        "threshold": MIN_N_WARNING,
        "n_small_tasks": len(small),
    }


def check_nonmonotonic_horizon(results_df: pd.DataFrame) -> Dict:
    """Check for cases where R^2 increases at later horizons (unexpected)."""
    reg = results_df[(results_df["task_type"] == "regression") & (results_df["model"] == "xgboost")]
    flags = []
    for (target, regime), grp in reg.groupby(["target", "regime"]):
        g = grp.sort_values("horizon_months")
        r2s = g["r2"].values
        months = g["horizon_months"].values
        for i in range(1, len(r2s)):
            if np.isfinite(r2s[i]) and np.isfinite(r2s[i-1]) and r2s[i] > r2s[i-1] + 0.05:
                flags.append({
                    "target": target, "regime": regime,
                    "from_months": int(months[i-1]), "to_months": int(months[i]),
                    "r2_increase": float(r2s[i] - r2s[i-1]),
                })
    return {
        "status": "PASS" if not flags else "WARN",
        "n_nonmonotonic": len(flags),
        "flags": flags[:10] if flags else None,
    }


def check_output_completeness(results_df: pd.DataFrame) -> Dict:
    expected = [RESULTS_DIR / "frontier_results.csv"]
    missing = [str(f) for f in expected if not f.exists()]
    pred_dir = RESULTS_DIR / "predictions"
    n_preds = sum(1 for d in pred_dir.iterdir() if d.is_dir()) if pred_dir.exists() else 0
    return {
        "status": "PASS" if not missing else "FAIL",
        "missing_files": missing if missing else None,
        "n_task_predictions": n_preds,
    }


def check_r2_primary(results_df: pd.DataFrame) -> Dict:
    """Verify R^2 column exists and is populated for regression tasks."""
    reg = results_df[results_df["task_type"] == "regression"]
    if reg.empty:
        return {"status": "SKIP"}
    if "r2" not in reg.columns:
        return {"status": "FAIL", "reason": "r2 column missing"}
    n_nan = reg["r2"].isna().sum()
    return {
        "status": "PASS" if n_nan == 0 else "WARN",
        "n_r2_nan": int(n_nan),
        "mean_r2": float(reg["r2"].mean()),
    }


def check_dimensionality() -> Dict:
    """Report-only check flagging tasks where feature count p exceeds the training-set size n_train."""
    manifest_path = PROCESSED_DATA_DIR / "task_manifest.csv"
    if not manifest_path.exists():
        return {"status": "SKIP", "reason": "task_manifest.csv not found"}

    manifest = pd.read_csv(manifest_path)
    if not {"n_total", "n_features", "task_key"}.issubset(manifest.columns):
        return {"status": "SKIP", "reason": "manifest missing n_total/n_features"}

    # Read outer+inner fold counts from the config.
    try:
        from configs.config import N_OUTER_FOLDS as _K, N_INNER_FOLDS as _KI
    except Exception:
        _K = 5
        _KI = 3

    manifest = manifest.copy()
    # Outer/inner training-fold sample sizes: ceil((k-1)/k * n).
    manifest["n_train_cv"] = np.ceil((_K - 1) / _K * manifest["n_total"]).astype(int)
    manifest["n_train_inner"] = np.ceil(
        (_KI - 1) / _KI * manifest["n_train_cv"]).astype(int)
    manifest["p_over_n_total"] = manifest["n_features"] / manifest["n_total"].replace(0, np.nan)
    manifest["p_over_n_train"] = manifest["n_features"] / manifest["n_train_cv"].replace(0, np.nan)
    manifest["p_over_n_inner"] = manifest["n_features"] / manifest["n_train_inner"].replace(0, np.nan)

    # Flag on the training-set ratio; retain the total ratio as a diagnostic.
    high_dim_train = manifest[manifest["n_features"] > manifest["n_train_cv"]].copy()
    high_dim_total = manifest[manifest["n_features"] > manifest["n_total"]].copy()

    n_tasks = len(manifest)
    n_hd_train = len(high_dim_train)
    n_hd_total = len(high_dim_total)
    pct_hd_train = (100 * n_hd_train / n_tasks) if n_tasks else 0.0

    # Sidecar table for Supplementary.
    try:
        tables_dir = RESULTS_DIR / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        out_path = tables_dir / "supp_dimensionality_check.csv"
        cols = [c for c in ["task_key", "target", "regime", "horizon",
                            "n_total", "n_train_cv", "n_train_inner",
                            "n_features",
                            "p_over_n_train", "p_over_n_inner",
                            "p_over_n_total"]
                if c in manifest.columns]
        manifest[cols].sort_values("p_over_n_train", ascending=False).to_csv(
            out_path, index=False)
    except Exception as e:
        log.warning("Could not write supp_dimensionality_check.csv: %s", e)

    # Top-10 by the training-set ratio.
    top_flagged = []
    for _, r in high_dim_train.sort_values("p_over_n_train", ascending=False).head(10).iterrows():
        top_flagged.append({
            "task_key": str(r.get("task_key", "")),
            "n_total": int(r.get("n_total", 0)),
            "n_train_cv": int(r.get("n_train_cv", 0)),
            "p": int(r.get("n_features", 0)),
            "p_over_n_train": round(float(r.get("p_over_n_train", 0.0)), 3),
            "p_over_n_total": round(float(r.get("p_over_n_total", 0.0)), 3),
        })

    return {
        "status": "PASS",
        "k_outer_folds": int(_K),
        "k_inner_folds": int(_KI),
        "had_high_dim_train": bool(n_hd_train),
        "n_tasks_total": n_tasks,
        # Count on the outer training fold.
        "n_tasks_p_gt_n_train": n_hd_train,
        "pct_p_gt_n_train": round(pct_hd_train, 1),
        "max_p_over_n_train": (float(manifest["p_over_n_train"].max())
                                if "p_over_n_train" in manifest.columns else None),
        # Inner-CV ratio.
        "max_p_over_n_inner": (float(manifest["p_over_n_inner"].max())
                                if "p_over_n_inner" in manifest.columns else None),
        # Count on the full sample.
        "n_tasks_p_gt_n_total": n_hd_total,
        "max_p_over_n_total": (float(manifest["p_over_n_total"].max())
                                if "p_over_n_total" in manifest.columns else None),
        "models_exposed": (
            ["Ridge/ElasticNet (fit via regularised path, no rank issue)",
             "RF/XGBoost (fit regardless, implicit regularisation via "
             "subsampling; mild at p/n_train <= 2)",
             "LME: NOT exposed (2 fixed effects only; effective p=2 << n_long)"]
            if n_hd_train else []
        ),
        "note": (
            "Report-only. High-dimensional regime defined as p > n_train "
            "(Hastie et al. 2015; Buehlmann & van de Geer 2011), with "
            "n_train = ceil((k-1)/k * n_total) under k-fold CV. "
            "p-vs-n_test is a separate concern (test-set variance) and is "
            "covered by `check_sample_sizes()`."
        ),
        "top_flagged": top_flagged or None,
    }


def run_all_qc_checks() -> Dict:
    results_path = RESULTS_DIR / "frontier_results.csv"
    if not results_path.exists():
        log.error("frontier_results.csv not found")
        return {"error": "frontier_results.csv not found"}

    results_df = pd.read_csv(results_path)
    log.info("QC on %d results", len(results_df))

    report = {
        "locf_scale": check_locf_scale(results_df),
        "sample_sizes": check_sample_sizes(results_df),
        "nonmonotonic_horizon": check_nonmonotonic_horizon(results_df),
        "output_completeness": check_output_completeness(results_df),
        "r2_primary": check_r2_primary(results_df),
        "dimensionality": check_dimensionality(),
    }

    statuses = [v.get("status", "UNKNOWN") for v in report.values()]
    report["overall"] = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")

    out_path = RESULTS_DIR / "qc_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("QC report: %s (overall: %s)", out_path, report["overall"])

    return report


if __name__ == "__main__":
    report = run_all_qc_checks()
    if report.get("overall") == "FAIL":
        sys.exit(1)
