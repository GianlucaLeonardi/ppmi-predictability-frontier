"""
NaN-safe scoring functions for regression, classification, and ranking tasks.
R^2 is the primary regression metric. Includes paired permutation tests.
"""

import numpy as np
from scipy import stats as sp_stats
from sklearn import metrics as sk_metrics
from typing import Dict, Callable, Optional


# =============================================================================
# Regression metrics
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all regression metrics. R^2 is the primary metric."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    if n < 5:
        return {k: float("nan") for k in ["r2", "spearman", "mae", "rmse", "pearson", "n"]}

    r2 = float(sk_metrics.r2_score(yt, yp))
    mae = float(sk_metrics.mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(sk_metrics.mean_squared_error(yt, yp)))

    if np.std(yt) < 1e-10 or np.std(yp) < 1e-10:
        spearman = pearson = 0.0
    else:
        spearman = float(sp_stats.spearmanr(yt, yp).correlation)
        pearson = float(sp_stats.pearsonr(yt, yp)[0])

    return {"r2": r2, "spearman": spearman, "mae": mae, "rmse": rmse, "pearson": pearson, "n": n}


# =============================================================================
# Classification metrics (expanded with MCC and balanced accuracy)
# =============================================================================

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    n_classes: int = 2,
) -> Dict[str, float]:
    """Compute classification metrics; probabilistic metrics require y_prob."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask].astype(int), y_pred[mask].astype(int)
    n = len(yt)

    all_metric_keys = [
        "auroc", "auprc", "f1", "mcc", "balanced_accuracy",
        "sensitivity", "specificity", "ppv", "npv",
        "accuracy", "brier", "prevalence", "n",
    ]

    if n < 10 or len(np.unique(yt)) < 2:
        return {k: float("nan") for k in all_metric_keys}

    result = {
        "accuracy": float(sk_metrics.accuracy_score(yt, yp)),
        "balanced_accuracy": float(sk_metrics.balanced_accuracy_score(yt, yp)),
        "mcc": float(sk_metrics.matthews_corrcoef(yt, yp)),
        "n": n,
    }

    if n_classes == 2:
        # Binary classification: full metric set
        tp = ((yp == 1) & (yt == 1)).sum()
        tn = ((yp == 0) & (yt == 0)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()

        result["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        result["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        result["ppv"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        result["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        result["f1"] = float(sk_metrics.f1_score(yt, yp, zero_division=0))
        result["prevalence"] = float(np.mean(yt))

        if y_prob is not None:
            yprob = y_prob[mask]
            try:
                result["auroc"] = float(sk_metrics.roc_auc_score(yt, yprob))
            except ValueError:
                result["auroc"] = float("nan")
            try:
                result["auprc"] = float(sk_metrics.average_precision_score(yt, yprob))
            except ValueError:
                result["auprc"] = float("nan")
            result["brier"] = float(sk_metrics.brier_score_loss(yt, yprob))
        else:
            result["auroc"] = result["auprc"] = result["brier"] = float("nan")
    else:
        # Multiclass: macro-averaged F1, no binary-specific metrics
        result["f1"] = float(sk_metrics.f1_score(yt, yp, average="macro", zero_division=0))
        result["sensitivity"] = float("nan")
        result["specificity"] = float("nan")
        result["ppv"] = float("nan")
        result["npv"] = float("nan")
        result["prevalence"] = float("nan")

        if y_prob is not None and y_prob.ndim == 2:
            yprob = y_prob[mask]
            try:
                result["auroc"] = float(sk_metrics.roc_auc_score(
                    yt, yprob, multi_class="ovr", average="macro"))
            except ValueError:
                result["auroc"] = float("nan")
            result["auprc"] = float("nan")  # not well-defined for multiclass
            result["brier"] = float("nan")
        else:
            result["auroc"] = result["auprc"] = result["brier"] = float("nan")

    return result


# =============================================================================
# Ranking metrics
# =============================================================================

def ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Metrics for ranking tasks: Spearman is primary, Kendall tau secondary."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    if n < 5:
        return {k: float("nan") for k in ["spearman", "kendall_tau", "n"]}

    if np.std(yt) < 1e-10 or np.std(yp) < 1e-10:
        return {"spearman": 0.0, "kendall_tau": 0.0, "n": n}

    spearman = float(sp_stats.spearmanr(yt, yp).correlation)
    kendall = float(sp_stats.kendalltau(yt, yp).correlation)

    return {"spearman": spearman, "kendall_tau": kendall, "n": n}


# =============================================================================
# R^2/Spearman divergence detection
# =============================================================================

def detect_r2_spearman_divergence(
    results_df,
    r2_thresh: float = 0.02,
    spearman_thresh: float = 0.05,
) -> list:
    """Find (task, model_a, model_b) triples where R^2 is similar but Spearman differs."""
    divergences = []
    reg = results_df[results_df["task_type"] == "regression"]

    for task_key, grp in reg.groupby("task_key"):
        models = grp[["model", "r2", "spearman"]].dropna()
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                row_a = models.iloc[i]
                row_b = models.iloc[j]
                r2_diff = abs(row_a["r2"] - row_b["r2"])
                sp_diff = abs(row_a["spearman"] - row_b["spearman"])
                if r2_diff < r2_thresh and sp_diff > spearman_thresh:
                    divergences.append({
                        "task_key": task_key,
                        "model_a": row_a["model"],
                        "model_b": row_b["model"],
                        "r2_a": row_a["r2"],
                        "r2_b": row_b["r2"],
                        "r2_diff": r2_diff,
                        "spearman_a": row_a["spearman"],
                        "spearman_b": row_b["spearman"],
                        "spearman_diff": sp_diff,
                    })

    return divergences


# =============================================================================
# Bootstrap CIs
# =============================================================================

def bootstrap_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    n_boot: int = 2000, seed: int = 42, ci: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)

    empty = {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "se": float("nan")}
    metrics_keys = ["r2", "spearman", "mae", "rmse", "pearson"]
    if n < 10:
        return {k: dict(empty) for k in metrics_keys}

    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2
    boot = {k: [] for k in metrics_keys}

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        bt, bp = yt[idx], yp[idx]

        if np.std(bt) < 1e-10 or np.std(bp) < 1e-10:
            boot["spearman"].append(0.0)
            boot["pearson"].append(0.0)
        else:
            boot["spearman"].append(float(sp_stats.spearmanr(bt, bp).correlation))
            boot["pearson"].append(float(sp_stats.pearsonr(bt, bp)[0]))

        boot["r2"].append(float(sk_metrics.r2_score(bt, bp)))
        boot["mae"].append(float(sk_metrics.mean_absolute_error(bt, bp)))
        boot["rmse"].append(float(np.sqrt(sk_metrics.mean_squared_error(bt, bp))))

    pt = regression_metrics(y_true, y_pred)

    def _summarize(point_val, boot_vals):
        arr = np.array(boot_vals)
        return {
            "point": point_val,
            "ci_lo": float(np.percentile(arr, 100 * alpha)),
            "ci_hi": float(np.percentile(arr, 100 * (1 - alpha))),
            "se": float(np.std(arr)),
        }

    return {k: _summarize(pt[k], boot[k]) for k in metrics_keys}


def bootstrap_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray,
    n_boot: int = 2000, seed: int = 42, ci: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    yt, yp = y_true[mask], y_prob[mask]
    n = len(yt)

    empty = {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "se": float("nan")}
    if n < 10 or len(np.unique(yt)) < 2:
        return {"auroc": dict(empty), "auprc": dict(empty)}

    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2
    boot_auroc, boot_auprc = [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        bt, bp = yt[idx], yp[idx]
        if len(np.unique(bt)) < 2:
            continue
        try:
            boot_auroc.append(float(sk_metrics.roc_auc_score(bt, bp)))
            boot_auprc.append(float(sk_metrics.average_precision_score(bt, bp)))
        except ValueError:
            continue

    def _summarize(boot_vals):
        if not boot_vals:
            return dict(empty)
        arr = np.array(boot_vals)
        return {
            "point": float(np.mean(arr)),
            "ci_lo": float(np.percentile(arr, 100 * alpha)),
            "ci_hi": float(np.percentile(arr, 100 * (1 - alpha))),
            "se": float(np.std(arr)),
        }

    return {"auroc": _summarize(boot_auroc), "auprc": _summarize(boot_auprc)}


# =============================================================================
# Paired permutation test
# =============================================================================

def paired_permutation_test(
    y_true, y_pred_a, y_pred_b,
    metric_fn: Optional[Callable] = None,
    n_perm: int = 5000, seed: int = 42,
) -> Dict[str, float]:
    """Paired permutation test comparing two models' predictions; default metric is R^2."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred_a) & np.isfinite(y_pred_b)
    yt, ya, yb = y_true[mask], y_pred_a[mask], y_pred_b[mask]
    n = len(yt)
    if n < 10:
        return {"observed_diff": float("nan"), "p_value": float("nan"), "n": n}

    if metric_fn is None:
        def metric_fn(y, yp):
            return float(sk_metrics.r2_score(y, yp))

    observed_diff = metric_fn(yt, ya) - metric_fn(yt, yb)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        perm_a = np.where(swap, yb, ya)
        perm_b = np.where(swap, ya, yb)
        if abs(metric_fn(yt, perm_a) - metric_fn(yt, perm_b)) >= abs(observed_diff):
            count += 1

    return {"observed_diff": float(observed_diff), "p_value": float((count + 1) / (n_perm + 1)), "n": n}


def target_variance_context(y_true: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true)
    yt = y_true[mask]
    if len(yt) < 2:
        return {}
    return {
        "target_mean": float(np.mean(yt)),
        "target_std": float(np.std(yt)),
        "target_median": float(np.median(yt)),
        "target_iqr": float(np.percentile(yt, 75) - np.percentile(yt, 25)),
        "target_range": float(np.ptp(yt)),
    }
