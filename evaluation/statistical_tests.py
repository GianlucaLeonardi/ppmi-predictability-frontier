"""Fold-aware paired statistical tests (permutation + bootstrap CI) with BH-FDR
correction for model comparisons, regime lift, and ablations."""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    ALPHA_STATISTICAL,
    N_PERMUTATIONS,
    RESULTS_DIR,
    SEED,
)
from utils.logging_utils import get_logger

log = get_logger("statistical_tests")


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    n = len(p_values)
    if n == 0:
        return np.array([])
    valid = np.isfinite(p_values)
    adjusted = np.full(n, float("nan"))
    valid_p = p_values[valid]
    m = len(valid_p)
    if m == 0:
        return adjusted

    sorted_idx = np.argsort(valid_p)
    sorted_p = valid_p[sorted_idx]
    bh = np.zeros(m)
    bh[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        bh[i] = min(bh[i + 1], sorted_p[i] * m / (i + 1))
    bh = np.clip(bh, 0, 1)
    result = np.zeros(m)
    result[sorted_idx] = bh
    adjusted[valid] = result
    return adjusted


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d for paired samples."""
    diff = a - b
    if len(diff) < 2 or np.std(diff) < 1e-10:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def _fold_aware_paired_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired permutation test with bootstrap CI on fold-level scores."""
    valid = np.isfinite(scores_a) & np.isfinite(scores_b)
    a, b = scores_a[valid], scores_b[valid]
    n = len(a)
    if n < 3:
        return {"observed_diff": float("nan"), "p_value": float("nan"),
                "cohens_d": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n_folds": n}

    observed_diff = float(np.mean(a - b))
    d = _cohens_d(a, b)

    # Bootstrap CI for the mean difference
    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(2000):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs.append(np.mean(a[idx] - b[idx]))
    boot_diffs = np.array(boot_diffs)
    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))

    # Permutation test: swap model assignments within each fold
    count = 0
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        perm_diff = np.mean(np.where(swap, b - a, a - b))
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = float((count + 1) / (n_perm + 1))

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "cohens_d": d,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_folds": n,
    }


def run_model_comparison_tests(
    fold_results_path: Optional[Path] = None,
    model_a: str = "xgboost",
    model_b: str = "locf",
) -> pd.DataFrame:
    """Compare two models across tasks using fold-level scores."""
    if fold_results_path is None:
        fold_results_path = RESULTS_DIR / "frontier_results_per_fold.csv"
    if not fold_results_path.exists():
        log.warning("Per-fold results not found: %s", fold_results_path)
        return pd.DataFrame()

    df = pd.read_csv(fold_results_path)
    reg = df[df["task_type"] == "regression"]

    rows = []
    for task_key, grp in reg.groupby("task_key"):
        a_scores = grp[grp["model"] == model_a]
        b_scores = grp[grp["model"] == model_b]

        if a_scores.empty or b_scores.empty:
            continue

        # Match by (seed, fold)
        merged = a_scores.merge(
            b_scores, on=["seed", "fold"], suffixes=("_a", "_b"),
            how="inner",
        )
        if len(merged) < 3:
            continue

        task_info = a_scores.iloc[0]

        # Test on R^2 (primary)
        r2_test = _fold_aware_paired_test(
            merged["r2_a"].values, merged["r2_b"].values,
            n_perm=N_PERMUTATIONS, seed=SEED,
        )

        # Test on Spearman (secondary)
        sp_test = _fold_aware_paired_test(
            merged["spearman_a"].values, merged["spearman_b"].values,
            n_perm=N_PERMUTATIONS, seed=SEED,
        )

        rows.append({
            "task_key": task_key,
            "target": task_info["target"],
            "target_display": task_info["target_display"],
            "target_domain": task_info["target_domain"],
            "horizon": task_info["horizon"],
            "horizon_months": task_info["horizon_months"],
            "regime": task_info["regime"],
            "model_a": model_a, "model_b": model_b,
            # R^2 test results
            "r2_diff": r2_test["observed_diff"],
            "r2_p_value": r2_test["p_value"],
            "r2_cohens_d": r2_test["cohens_d"],
            "r2_ci_lo": r2_test["ci_lo"],
            "r2_ci_hi": r2_test["ci_hi"],
            # Spearman test results
            "spearman_diff": sp_test["observed_diff"],
            "spearman_p_value": sp_test["p_value"],
            "spearman_cohens_d": sp_test["cohens_d"],
            "spearman_ci_lo": sp_test["ci_lo"],
            "spearman_ci_hi": sp_test["ci_hi"],
            "n_folds": r2_test["n_folds"],
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["r2_p_adjusted"] = benjamini_hochberg(result["r2_p_value"].values, ALPHA_STATISTICAL)
        result["r2_significant"] = result["r2_p_adjusted"] < ALPHA_STATISTICAL
        result["spearman_p_adjusted"] = benjamini_hochberg(result["spearman_p_value"].values, ALPHA_STATISTICAL)
        result["spearman_significant"] = result["spearman_p_adjusted"] < ALPHA_STATISTICAL
    return result


def run_regime_lift_tests(
    fold_results_path: Optional[Path] = None,
    model: str = "xgboost",
    regime_a: str = "rolling",
    regime_b: str = "baseline_only",
) -> pd.DataFrame:
    """Compare two regimes using fold-level scores for a given model."""
    if fold_results_path is None:
        fold_results_path = RESULTS_DIR / "frontier_results_per_fold.csv"
    if not fold_results_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(fold_results_path)
    reg = df[(df["task_type"] == "regression") & (df["model"] == model)]

    rows = []
    for (target, horizon), grp in reg.groupby(["target", "horizon"]):
        a_scores = grp[grp["regime"] == regime_a]
        b_scores = grp[grp["regime"] == regime_b]
        if a_scores.empty or b_scores.empty:
            continue

        merged = a_scores.merge(
            b_scores, on=["seed", "fold"], suffixes=("_a", "_b"),
            how="inner",
        )
        if len(merged) < 3:
            continue

        task_info = a_scores.iloc[0]

        r2_test = _fold_aware_paired_test(
            merged["r2_a"].values, merged["r2_b"].values,
            n_perm=N_PERMUTATIONS, seed=SEED,
        )

        rows.append({
            "target": target,
            "target_display": task_info["target_display"],
            "target_domain": task_info["target_domain"],
            "horizon": horizon,
            "horizon_months": task_info["horizon_months"],
            "regime_a": regime_a, "regime_b": regime_b, "model": model,
            "r2_diff": r2_test["observed_diff"],
            "r2_p_value": r2_test["p_value"],
            "r2_cohens_d": r2_test["cohens_d"],
            "r2_ci_lo": r2_test["ci_lo"],
            "r2_ci_hi": r2_test["ci_hi"],
            "n_folds": r2_test["n_folds"],
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["r2_p_adjusted"] = benjamini_hochberg(result["r2_p_value"].values, ALPHA_STATISTICAL)
        result["r2_significant"] = result["r2_p_adjusted"] < ALPHA_STATISTICAL
    return result


def run_ablation_perfold_tests(
    perfold_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Paired test per (task_key, dropped_modality) on per-fold ablation rows, with BH-FDR."""
    if perfold_path is None:
        perfold_path = RESULTS_DIR / "tables" / "ablation_perfold_xgboost.csv"
    if not perfold_path.exists():
        log.warning("Per-fold ablation CSV not found: %s", perfold_path)
        return pd.DataFrame()

    df = pd.read_csv(perfold_path)
    if df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["task_key", "dropped_modality"]
    for (task_key, modality), grp in df.groupby(group_cols):
        full = grp["full_r2"].values
        abl = grp["ablated_r2"].values
        valid = np.isfinite(full) & np.isfinite(abl)
        if valid.sum() < 3:
            continue
        # Paired test: H0 is full == ablated. Positive delta = full > ablated
        # = removing this modality HURT performance.
        test = _fold_aware_paired_test(
            full[valid], abl[valid],
            n_perm=N_PERMUTATIONS, seed=SEED,
        )
        meta = grp.iloc[0]
        mean_full = float(np.mean(full[valid]))
        mean_abl = float(np.mean(abl[valid]))
        fold_change = (
            100.0 * (mean_abl - mean_full) / mean_full
            if abs(mean_full) > 1e-6 else float("nan")
        )
        rows.append({
            "task_key": task_key,
            "target": meta["target"],
            "target_display": meta["target_display"],
            "target_domain": meta["target_domain"],
            "horizon": meta["horizon"],
            "horizon_months": meta["horizon_months"],
            "regime": meta["regime"],
            "dropped_modality": modality,
            "dropped_modality_display": meta["dropped_modality_display"],
            "n_dropped_features": int(meta["n_dropped_features"]),
            "n_folds": int(valid.sum()),
            "mean_full_r2": mean_full,
            "mean_ablated_r2": mean_abl,
            "mean_delta_r2": test["observed_diff"],
            "fold_change_pct": fold_change,
            "r2_p_value": test["p_value"],
            "r2_cohens_d": test["cohens_d"],
            "r2_ci_lo": test["ci_lo"],
            "r2_ci_hi": test["ci_hi"],
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["r2_p_adjusted"] = benjamini_hochberg(
            out["r2_p_value"].values, ALPHA_STATISTICAL)
        out["r2_significant"] = out["r2_p_adjusted"] < ALPHA_STATISTICAL
        tables_dir = RESULTS_DIR / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(tables_dir / "ablation_perfold_tests.csv", index=False)
        log.info("Per-fold ablation tests: %d (task, modality) pairs, %d significant after BH-FDR",
                 len(out), int(out["r2_significant"].sum()))
    return out


def run_all_statistical_tests() -> Dict[str, pd.DataFrame]:
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Model comparisons (R^2-first, with Spearman secondary).
    comparisons = [
        ("xgboost", "locf"), ("xgboost", "elastic_net"), ("xgboost", "ridge"),
        ("xgboost", "population_mean"), ("xgboost", "random_forest"),
        ("xgboost", "lme"), ("lme", "locf"),
        ("elastic_net", "locf"), ("random_forest", "locf"),
        ("elastic_net", "ridge"), ("elastic_net", "xgboost"),
        ("elastic_net", "random_forest"),
    ]
    all_model_tests = []
    for a, b in comparisons:
        log.info("Testing %s vs %s (R^2 + Spearman)", a, b)
        all_model_tests.append(run_model_comparison_tests(model_a=a, model_b=b))

    model_tests = pd.concat(all_model_tests, ignore_index=True) if all_model_tests else pd.DataFrame()
    if not model_tests.empty:
        # Re-apply BH correction across all comparisons
        model_tests["r2_p_adjusted"] = benjamini_hochberg(
            model_tests["r2_p_value"].values, ALPHA_STATISTICAL)
        model_tests["r2_significant"] = model_tests["r2_p_adjusted"] < ALPHA_STATISTICAL
        model_tests["spearman_p_adjusted"] = benjamini_hochberg(
            model_tests["spearman_p_value"].values, ALPHA_STATISTICAL)
        model_tests["spearman_significant"] = model_tests["spearman_p_adjusted"] < ALPHA_STATISTICAL
        model_tests.to_csv(tables_dir / "model_comparison_tests.csv", index=False)
        log.info("Model comparisons: %d total, %d significant (R^2), %d significant (Spearman)",
                 len(model_tests),
                 model_tests["r2_significant"].sum(),
                 model_tests["spearman_significant"].sum())
    results["model_comparison"] = model_tests

    # Regime lift
    regime_pairs = [
        ("rolling", "baseline_only"), ("baseline_plus_12m", "baseline_only"),
        ("rolling", "baseline_plus_12m"),
    ]
    all_regime_tests = []
    for a, b in regime_pairs:
        log.info("Testing regime %s vs %s (R^2)", a, b)
        all_regime_tests.append(run_regime_lift_tests(regime_a=a, regime_b=b))

    regime_tests = pd.concat(all_regime_tests, ignore_index=True) if all_regime_tests else pd.DataFrame()
    if not regime_tests.empty:
        regime_tests["r2_p_adjusted"] = benjamini_hochberg(
            regime_tests["r2_p_value"].values, ALPHA_STATISTICAL)
        regime_tests["r2_significant"] = regime_tests["r2_p_adjusted"] < ALPHA_STATISTICAL
        regime_tests.to_csv(tables_dir / "regime_lift_tests.csv", index=False)
    results["regime_lift"] = regime_tests

    # Per-fold ablation tests (Fig 6 significance annotations).
    # Silently no-ops if results/tables/ablation_perfold_xgboost.csv is absent.
    log.info("Running per-fold ablation paired tests (if data available)")
    ablation_tests = run_ablation_perfold_tests()
    if not ablation_tests.empty:
        results["ablation_perfold"] = ablation_tests

    # Summary table
    summary_rows = []
    for name, df in results.items():
        if df.empty:
            continue
        sig_col = "r2_significant" if "r2_significant" in df.columns else "significant"
        p_col = "r2_p_value" if "r2_p_value" in df.columns else "p_value"
        summary_rows.append({
            "test_type": name,
            "n_tests": len(df),
            "n_significant_r2": int(df[sig_col].sum()) if sig_col in df.columns else 0,
            "fdr_threshold": ALPHA_STATISTICAL,
            "median_p": float(df[p_col].median()) if p_col in df.columns else float("nan"),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables_dir / "statistical_tests_summary.csv", index=False)

    return results


if __name__ == "__main__":
    run_all_statistical_tests()
