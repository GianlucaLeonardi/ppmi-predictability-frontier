"""
Calibration analysis for classification tasks.

Computes reliability diagrams (calibration curves) and calibration metrics
for the binary classification targets (motor worsening, cognitive impairment).
"""

import json

import numpy as np
import pandas as pd

from configs.config import (
    RESULTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
)
from utils.logging_utils import get_logger

log = get_logger(__name__)


def calibration_curve(y_true, y_prob, n_bins=10):
    """Compute calibration curve (fraction of positives vs mean predicted probability) plus ECE/MCE."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    fraction_pos = []
    mean_pred = []
    bin_counts = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo == bins[-2]:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        n_in_bin = mask.sum()
        bin_counts.append(int(n_in_bin))
        if n_in_bin == 0:
            bin_centers.append((lo + hi) / 2)
            fraction_pos.append(float("nan"))
            mean_pred.append(float("nan"))
        else:
            bin_centers.append((lo + hi) / 2)
            fraction_pos.append(float(y_true[mask].mean()))
            mean_pred.append(float(y_prob[mask].mean()))

    bin_centers = np.array(bin_centers)
    fraction_pos = np.array(fraction_pos)
    mean_pred = np.array(mean_pred)
    bin_counts = np.array(bin_counts)

    # ECE: expected calibration error (weighted by bin size)
    valid = np.isfinite(fraction_pos)
    n_total = bin_counts[valid].sum()
    ece = float(np.sum(bin_counts[valid] * np.abs(fraction_pos[valid] - mean_pred[valid])) / max(n_total, 1))

    # MCE: maximum calibration error
    mce = float(np.max(np.abs(fraction_pos[valid] - mean_pred[valid]))) if valid.any() else float("nan")

    return {
        "bin_edges": bins.tolist(),
        "bin_centers": bin_centers.tolist(),
        "fraction_positive": fraction_pos.tolist(),
        "mean_predicted": mean_pred.tolist(),
        "bin_counts": bin_counts.tolist(),
        "ece": ece,
        "mce": mce,
        "n_total": int(n_total),
    }


def run_calibration_analysis():
    """Aggregate out-of-fold predictions and compute calibration for each binary classification task."""
    pred_dir = RESULTS_DIR / "predictions"
    if not pred_dir.exists():
        log.warning("No predictions directory found; skipping calibration analysis")
        return

    # Load frontier results to identify classification tasks
    frontier_path = RESULTS_DIR / "frontier_results.csv"
    if not frontier_path.exists():
        log.warning("No frontier_results.csv found; skipping calibration analysis")
        return

    results_df = pd.read_csv(frontier_path)
    clf_tasks = results_df[results_df["task_type"] == "classification"]["task_key"].unique()

    all_calibration = []
    clf_models = ["logistic_regression", "random_forest_clf", "xgboost_clf"]

    for task_key in clf_tasks:
        task_pred_dir = pred_dir / task_key
        if not task_pred_dir.exists():
            continue

        for model_name in clf_models:
            # Aggregate predictions across all seed/fold subdirectories
            all_y_test = []
            all_y_prob = []
            for fold_dir in sorted(task_pred_dir.iterdir()):
                if not fold_dir.is_dir() or not fold_dir.name.startswith("seed"):
                    continue
                y_test_path = fold_dir / "y_test.npy"
                proba_path = fold_dir / f"{model_name}_proba.npy"
                if not y_test_path.exists() or not proba_path.exists():
                    continue
                y_test = np.load(y_test_path)
                y_prob = np.load(proba_path)
                if len(y_test) != len(y_prob):
                    continue
                # Only use 1D probability arrays (binary)
                if y_prob.ndim > 1:
                    continue
                all_y_test.append(y_test)
                all_y_prob.append(y_prob)

            if not all_y_test:
                continue

            y_test_agg = np.concatenate(all_y_test)
            y_prob_agg = np.concatenate(all_y_prob)

            valid = np.isfinite(y_test_agg) & np.isfinite(y_prob_agg)
            if valid.sum() < 20:
                continue

            cal = calibration_curve(y_test_agg[valid].astype(int), y_prob_agg[valid], n_bins=10)
            cal["task_key"] = task_key
            cal["model"] = model_name

            # Extract target info
            parts = task_key.split("__")
            cal["target"] = parts[0] if parts else task_key
            cal["regime"] = parts[1] if len(parts) > 1 else ""
            cal["horizon"] = parts[2] if len(parts) > 2 else ""

            all_calibration.append(cal)
            log.info("  Calibration %s / %s: ECE=%.4f, MCE=%.4f",
                     task_key, model_name, cal["ece"], cal["mce"])

    if not all_calibration:
        log.warning("No calibration results computed")
        return

    # Save calibration summary
    summary_rows = []
    for cal in all_calibration:
        summary_rows.append({
            "task_key": cal["task_key"],
            "target": cal["target"],
            "regime": cal["regime"],
            "horizon": cal["horizon"],
            "model": cal["model"],
            "ece": cal["ece"],
            "mce": cal["mce"],
            "n_total": cal["n_total"],
        })

    summary_df = pd.DataFrame(summary_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(TABLES_DIR / "calibration_summary.csv", index=False)
    log.info("Saved calibration summary to %s (%d rows)", TABLES_DIR / "calibration_summary.csv", len(summary_df))

    # Save detailed calibration data (for plotting)
    cal_path = RESULTS_DIR / "calibration_detail.json"
    with open(cal_path, "w") as f:
        json.dump(all_calibration, f, indent=2, default=str)
    log.info("Saved calibration detail to %s", cal_path)

    # Generate calibration figures
    try:
        from utils.plotting import plot_calibration_curves
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_calibration_curves(all_calibration, FIGURES_DIR)
        log.info("Calibration figures saved to %s", FIGURES_DIR)
    except Exception as e:
        log.warning("Calibration figure generation failed: %s", e)

    return summary_df


if __name__ == "__main__":
    run_calibration_analysis()
