"""Publication-grade plotting utilities for the PPMI Predictability-Frontier Benchmark."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs.config import (
    FIGURE_DPI, FIGURE_FORMAT, PALETTE, REGIME_MARKERS, REGIME_COLORS,
    RESULTS_DIR, VISIT_SCHEDULE,
)

# -- Global style -------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": FIGURE_DPI, "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

# Model display names for consistent labelling
_MODEL_DISPLAY = {
    "population_mean": "Pop. Mean",
    "locf": "LOCF",
    "lme": "LME",
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "majority_class": "Majority Class",
    "logistic_regression": "Logistic Reg.",
    "random_forest_clf": "Random Forest",
    "xgboost_clf": "XGBoost",
}

# Consistent model colours
_MODEL_COLORS = {
    # Okabe-Ito colour-blind-safe; RF=blue and XGB=vermillion kept across reg+clf
    "population_mean": "#999999",
    "locf": "#E69F00",
    "lme": "#56B4E9",
    "ridge": "#009E73",
    "elastic_net": "#CC79A7",
    "random_forest": "#0072B2",
    "xgboost": "#D55E00",
    "majority_class": "#999999",
    "logistic_regression": "#56B4E9",
    "random_forest_clf": "#0072B2",
    "xgboost_clf": "#D55E00",
}

# Human-readable metric labels (dynamic, not hardcoded)
_METRIC_LABELS = {
    "r2": "R\u00b2",
    "spearman": "Spearman \u03c1",
    "mae": "MAE",
    "rmse": "RMSE",
    "pearson": "Pearson r",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "mcc": "MCC",
    "balanced_accuracy": "Balanced Accuracy",
}


def _metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _model_label(model: str) -> str:
    return _MODEL_DISPLAY.get(model, model.replace("_", " ").title())


def _save(fig, name: str, subdir: str = "figures") -> Path:
    out_dir = RESULTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, bbox_inches="tight")
    # Also save a PNG copy alongside the vector output
    if FIGURE_FORMAT != "png":
        fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def _metric_suffix(metric: str, primary: str = "r2") -> str:
    """Return '' for the primary metric (no suffix in filename), '_<metric>' otherwise."""
    return "" if metric == primary else f"_{metric}"


def _fmt2(x, decimals: int = 2) -> str:
    """Format a number to fixed decimals, suppressing the sign on rounded zeros; NaN/None render as empty string."""
    if x is None:
        return ""
    if isinstance(x, float) and x != x:  # NaN
        return ""
    fmt = f"{{:.{decimals}f}}"
    s = fmt.format(x)
    if float(s) == 0.0:
        return fmt.format(0.0)
    return s


# =============================================================================
# Figure 1: Predictability-frontier heatmap
# =============================================================================

def plot_frontier_heatmap_paired(results_df, model="xgboost",
                                 regime="baseline_only", save=True):
    """Paired frontier heatmap: R² (left) and RMSE (right) side-by-side, rows ordered by mean R²."""
    sub = results_df[(results_df["model"] == model) & (results_df["regime"] == regime)].copy()

    pivot_r2 = sub.pivot_table(index="target_display", columns="horizon_months",
                               values="r2", aggfunc="mean")
    order = sub.groupby("target_display")["r2"].mean().sort_values(ascending=False).index
    pivot_r2 = pivot_r2.reindex(order)

    if "rmse" not in sub.columns:
        return None
    pivot_rmse = sub.pivot_table(index="target_display", columns="horizon_months",
                                 values="rmse", aggfunc="mean").reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(15, max(3.8, len(order) * 0.6)),
                             gridspec_kw={"wspace": 0.25})

    def _annot(pivot, ci_lo_col, ci_hi_col):
        a = pivot.copy().astype(str)
        for target in pivot.index:
            for horizon in pivot.columns:
                row = sub[(sub["target_display"] == target) & (sub["horizon_months"] == horizon)]
                if len(row) == 0:
                    a.loc[target, horizon] = ""
                    continue
                v = row[pivot.name if hasattr(pivot, "name") else ""].values
                a.loc[target, horizon] = _fmt2(pivot.loc[target, horizon])
        return a

    # Left: R²
    ax_l = axes[0]
    vmin_l = min(0, pivot_r2.min().min())
    vmax_l = max(0.5, pivot_r2.max().max())
    sns.heatmap(pivot_r2, annot=pivot_r2.applymap(_fmt2), fmt="", cmap="viridis",
                vmin=vmin_l, vmax=vmax_l, linewidths=0.5,
                cbar_kws={"label": "R²", "shrink": 0.7}, ax=ax_l,
                annot_kws={"fontsize": 9})
    ax_l.set_title("(a) R² (variance explained, higher = better)",
                   fontweight="bold", fontsize=11)
    ax_l.set_xlabel("Forecast Horizon (months)", fontsize=10)
    ax_l.set_ylabel("")

    # Right: RMSE — reversed colormap (lower = better = green) so the
    # visual reading is consistent with the R² panel: greener = better.
    ax_r = axes[1]
    vmin_r = 0
    vmax_r = max(pivot_rmse.max().max(), 1e-6)
    sns.heatmap(pivot_rmse, annot=pivot_rmse.applymap(_fmt2), fmt="",
                cmap="viridis_r",
                vmin=vmin_r, vmax=vmax_r, linewidths=0.5,
                cbar_kws={"label": "RMSE (target's native units)", "shrink": 0.7},
                ax=ax_r, annot_kws={"fontsize": 9})
    ax_r.set_title("(b) RMSE (absolute error, lower = better)",
                   fontweight="bold", fontsize=11)
    ax_r.set_xlabel("Forecast Horizon (months)", fontsize=10)
    ax_r.set_ylabel("")

    regime_label = regime.replace("_", " ").title()
    fig.suptitle(f"Predictability Frontier — {regime_label} ({_model_label(model)})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save:
        _save(fig, f"frontier_heatmap_{regime}_{model}_paired")
    return fig


def plot_frontier_heatmap(results_df, metric="r2", model="xgboost",
                          regime="baseline_only", save=True):
    sub = results_df[(results_df["model"] == model) & (results_df["regime"] == regime)].copy()
    pivot = sub.pivot_table(index="target_display", columns="horizon_months",
                            values=metric, aggfunc="mean")
    order = sub.groupby("target_display")[metric].mean().sort_values(ascending=False)
    pivot = pivot.reindex(order.index)

    annot = pivot.copy().astype(str)
    ci_lo, ci_hi = f"{metric}_ci_lo", f"{metric}_ci_hi"
    for target in pivot.index:
        for horizon in pivot.columns:
            row = sub[(sub["target_display"] == target) & (sub["horizon_months"] == horizon)]
            if len(row) == 0:
                annot.loc[target, horizon] = ""
                continue
            val = row[metric].values[0]
            if ci_lo in sub.columns and np.isfinite(row[ci_lo].values[0]):
                annot.loc[target, horizon] = (
                    f"{_fmt2(val)}\n[{_fmt2(row[ci_lo].values[0])}, "
                    f"{_fmt2(row[ci_hi].values[0])}]"
                )
            else:
                annot.loc[target, horizon] = _fmt2(val)

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(pivot) * 0.6)))
    # Error metrics (RMSE, MAE) are "lower = better"; flip the colormap so
    # red still marks worse cells and green still marks better cells.
    error_metric = metric in {"rmse", "mae"}
    if error_metric:
        vmin = 0
        vmax = max(pivot.max().max(), 1e-6)
        cmap = "viridis_r"
    else:
        vmin = min(0, pivot.min().min()) if metric == "r2" else 0
        vmax = max(0.5, pivot.max().max())
        cmap = "viridis"
    sns.heatmap(pivot, annot=annot, fmt="", cmap=cmap, vmin=vmin, vmax=vmax,
                linewidths=0.5, cbar_kws={"label": _metric_label(metric), "shrink": 0.8},
                ax=ax, annot_kws={"fontsize": 8})
    regime_label = regime.replace("_", " ").title()
    ax.set_title(f"Predictability Frontier \u2014 {regime_label} ({_model_label(model)})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (months)", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    if save:
        _save(fig, f"frontier_heatmap_{regime}_{model}{_metric_suffix(metric)}")
    return fig


# =============================================================================
# Figure 2: Predictability-frontier scatter
# =============================================================================

def plot_frontier_scatter(results_df, metric="r2", model="xgboost",
                          regime="baseline_only", save=True):
    sub = results_df[(results_df["model"] == model) & (results_df["regime"] == regime)].copy()
    ci_lo, ci_hi = f"{metric}_ci_lo", f"{metric}_ci_hi"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for domain, grp in sub.groupby("target_domain"):
        color = PALETTE.get(domain, "#999")
        n_col = "n_test" if "n_test" in grp.columns else "n_test_mean"
        ax.scatter(grp["horizon_months"], grp[metric], c=color, label=domain.title(),
                   alpha=0.7, s=grp[n_col].clip(30, 200) * 0.6,
                   edgecolors="white", linewidths=0.5, zorder=3)
        if ci_lo in sub.columns:
            for _, row in grp.iterrows():
                lo = row.get(ci_lo, np.nan)
                hi = row.get(ci_hi, np.nan)
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.vlines(row["horizon_months"], lo, hi,
                              color=color, alpha=0.25, linewidth=0.8, zorder=2)
        for _, tgrp in grp.groupby("target"):
            t = tgrp.sort_values("horizon_months")
            ax.plot(t["horizon_months"], t[metric], c=color, alpha=0.3,
                    linewidth=0.8, zorder=1)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=0)
    ax.set_xlabel("Forecast Horizon (months)")
    ax.set_ylabel(_metric_label(metric))
    regime_label = regime.replace("_", " ").title()
    ax.set_title(f"Predictability Frontier \u2014 {regime_label}", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    if save:
        _save(fig, f"frontier_scatter_{regime}_{model}{_metric_suffix(metric)}")
    return fig


# =============================================================================
# Figures 3-4: Model comparison grouped bar chart
# =============================================================================

def plot_model_comparison(results_df, regime="baseline_only", metric="r2",
                          horizon="V08", save=True, sig_df=None):
    sub = results_df[(results_df["regime"] == regime) & (results_df["horizon"] == horizon)].copy()
    # Exclude population_mean — it's always zero and wastes space
    sub = sub[sub["model"] != "population_mean"]
    ci_lo, ci_hi = f"{metric}_ci_lo", f"{metric}_ci_hi"

    targets = sub.groupby("target_display")[metric].max().sort_values(ascending=False).index
    models = [m for m in ["locf", "lme", "ridge", "elastic_net", "random_forest", "xgboost"]
              if m in sub["model"].values]
    x = np.arange(len(targets))
    width = 0.75 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals, errs_lo, errs_hi = [], [], []
        for t in targets:
            row = sub[(sub["target_display"] == t) & (sub["model"] == model)]
            val = row[metric].values[0] if len(row) > 0 else 0
            vals.append(val)
            if ci_lo in sub.columns and len(row) > 0:
                lo = row[ci_lo].values[0] if np.isfinite(row[ci_lo].values[0]) else val
                hi_v = row[ci_hi].values[0] if np.isfinite(row[ci_hi].values[0]) else val
                errs_lo.append(max(0, val - lo))
                errs_hi.append(max(0, hi_v - val))
            else:
                errs_lo.append(0)
                errs_hi.append(0)

        color = _MODEL_COLORS.get(model, f"C{i}")
        ax.bar(x + i * width, vals, width, label=_model_label(model),
               color=color, edgecolor="white", linewidth=0.5)
        if any(e > 0 for e in errs_hi):
            ax.errorbar(x + i * width, vals, yerr=[errs_lo, errs_hi],
                        fmt="none", color="#333", capsize=2, linewidth=0.7)

    ax.yaxis.grid(True, which="major", color="#dddddd", linewidth=0.6,
                  linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#666", linewidth=0.7, linestyle="--", zorder=2)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(targets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(_metric_label(metric))
    months = VISIT_SCHEDULE.get(horizon, "?")
    regime_label = regime.replace("_", " ").title()
    ax.set_title(f"Model Comparison \u2014 {regime_label}, {months}-month Horizon",
                 fontweight="bold", fontsize=12)
    ax.legend(frameon=False, ncol=len(models), fontsize=9, loc="best")

    # No on-figure significance overlay; error bars convey per-bar uncertainty.
    fig.tight_layout()
    if save:
        _save(fig, f"model_comparison_{regime}_{horizon}{_metric_suffix(metric)}")
    return fig


# =============================================================================
# Figure 5: Regime comparison line plot
# =============================================================================

def plot_regime_comparison(results_df, target, metric="r2",
                           model="xgboost", save=True, regime_lift_df=None):
    """Regime comparison line plot per target: mean metric per (regime, horizon) with 95% CV-distribution bands."""
    sub = results_df[(results_df["target"] == target) & (results_df["model"] == model)].copy()
    ci_lo, ci_hi = f"{metric}_ci_lo", f"{metric}_ci_hi"
    regime_order = ["baseline_only", "baseline_multimodal", "baseline_plus_12m", "rolling"]

    # Optional significance lookup: BH-FDR p_adjusted of (regime_a vs baseline_only)
    # for this (target, horizon, model). Keys: (regime, horizon_months) -> p_adj.
    sig_lookup = {}
    if (
        metric == "r2"
        and regime_lift_df is not None
        and not regime_lift_df.empty
        and {"target", "horizon_months", "model", "regime_a", "regime_b",
             "r2_p_adjusted"}.issubset(regime_lift_df.columns)
    ):
        rl = regime_lift_df[
            (regime_lift_df["target"] == target)
            & (regime_lift_df["model"] == model)
            & (regime_lift_df["regime_b"] == "baseline_only")
        ]
        for row in rl.itertuples(index=False):
            sig_lookup[(row.regime_a, int(row.horizon_months))] = float(row.r2_p_adjusted)

    fig, ax = plt.subplots(figsize=(5.8, 4.0))

    for regime in regime_order:
        grp = sub[sub["regime"] == regime].sort_values("horizon_months")
        if grp.empty:
            continue
        color = REGIME_COLORS.get(regime, "#333")
        marker = REGIME_MARKERS.get(regime, "o")
        ax.plot(grp["horizon_months"], grp[metric],
                marker=marker,
                color=color,
                label=regime.replace("_", " ").title(),
                linewidth=1.6, markersize=6.5, zorder=3)
        if ci_lo in grp.columns:
            lo, hi = grp[ci_lo].values, grp[ci_hi].values
            if np.any(np.isfinite(lo)):
                ax.fill_between(grp["horizon_months"].values, lo, hi, alpha=0.12,
                                color=color, zorder=1)

    ax.yaxis.grid(True, which="major", color="#dddddd", linewidth=0.6,
                  linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#666", linewidth=0.6, linestyle="--", zorder=2)
    ax.set_xlabel("Forecast Horizon (months)")
    ax.set_ylabel(_metric_label(metric))
    display = sub["target_display"].iloc[0] if len(sub) > 0 else target
    ax.set_title(f"{display} \u2014 Information Regime Comparison",
                 fontweight="bold", fontsize=12)
    # Legend below the axes so it never collides with the significance asterisks.
    ax.legend(frameon=False, fontsize=9,
              loc="upper center",
              bbox_to_anchor=(0.5, -0.18),
              ncol=4, borderaxespad=0)

    # Per-regime BH-FDR-adjusted asterisks: one horizontal row per regime above the data envelope.
    if metric == "r2" and sig_lookup:
        top_val = -np.inf
        for regime in regime_order:
            grp = sub[sub["regime"] == regime]
            if grp.empty:
                continue
            v = grp[metric].values
            if np.any(np.isfinite(v)):
                top_val = max(top_val, float(np.nanmax(v)))
            if ci_hi in grp.columns:
                up = grp[ci_hi].values
                if np.any(np.isfinite(up)):
                    top_val = max(top_val, float(np.nanmax(up)))
        # Rolling on top, BL+12m below it, BM lowest.
        row_y = {"baseline_multimodal": top_val + 0.04,
                 "baseline_plus_12m":   top_val + 0.09,
                 "rolling":             top_val + 0.14}
        any_star = False
        for regime in ["baseline_multimodal", "baseline_plus_12m", "rolling"]:
            grp = sub[sub["regime"] == regime].sort_values("horizon_months")
            if grp.empty:
                continue
            color = REGIME_COLORS.get(regime, "#333")
            y = row_y[regime]
            for hm in grp["horizon_months"].values:
                p = sig_lookup.get((regime, int(hm)))
                stars = _stars_from_p(p) if p is not None else ""
                if not stars:
                    continue
                ax.text(hm, y, stars,
                        ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color=color,
                        zorder=10)
                any_star = True
        if any_star:
            cur_lo, cur_hi = ax.get_ylim()
            ax.set_ylim(cur_lo, max(cur_hi, top_val + 0.20))

    # Reserve bottom margin for the below-axes legend so tight_layout
    # does not clip it.
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    if save:
        _save(fig, f"regime_comparison_{target}_{model}{_metric_suffix(metric)}")
    return fig


def plot_regime_comparison_paired(results_df, target, model="xgboost",
                                  save=True, regime_lift_df=None):
    """Paired regime comparison: R\u00b2 (left) and RMSE (right) for one target across the four information regimes."""
    sub = results_df[(results_df["target"] == target) & (results_df["model"] == model)].copy()
    if sub.empty or "rmse" not in sub.columns:
        return None
    regime_order = ["baseline_only", "baseline_multimodal", "baseline_plus_12m", "rolling"]
    display = sub["target_display"].iloc[0]

    # Significance lookup (for footer only \u2014 no per-point overlay).
    sig_lookup = {}
    if (regime_lift_df is not None and not regime_lift_df.empty
            and {"target", "horizon_months", "model", "regime_a", "regime_b",
                 "r2_p_adjusted"}.issubset(regime_lift_df.columns)):
        rl = regime_lift_df[
            (regime_lift_df["target"] == target)
            & (regime_lift_df["model"] == model)
            & (regime_lift_df["regime_b"] == "baseline_only")
        ]
        for row in rl.itertuples(index=False):
            sig_lookup[(row.regime_a, int(row.horizon_months))] = float(row.r2_p_adjusted)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax_r2, ax_rmse = axes[0], axes[1]

    def _panel(ax, metric, ylabel, panel_title):
        ci_lo, ci_hi = f"{metric}_ci_lo", f"{metric}_ci_hi"
        for regime in regime_order:
            grp = sub[sub["regime"] == regime].sort_values("horizon_months")
            if grp.empty:
                continue
            color = REGIME_COLORS.get(regime, "#333")
            marker = REGIME_MARKERS.get(regime, "o")
            ax.plot(grp["horizon_months"], grp[metric],
                    marker=marker, color=color,
                    label=regime.replace("_", " ").title(),
                    linewidth=1.6, markersize=6.5, zorder=3)
            if ci_lo in grp.columns:
                lo, hi = grp[ci_lo].values, grp[ci_hi].values
                if np.any(np.isfinite(lo)):
                    ax.fill_between(grp["horizon_months"].values, lo, hi,
                                    alpha=0.12, color=color, zorder=1)
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(0, color="#666", linewidth=0.6, linestyle="--", zorder=2)
        ax.set_xlabel("Forecast Horizon (months)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(panel_title, fontweight="bold", fontsize=11)

    _panel(ax_r2, "r2", "R\u00b2",
           "(a) R\u00b2 (variance explained, higher = better)")
    _panel(ax_rmse, "rmse", "RMSE (target units)",
           "(b) RMSE (absolute error, lower = better)")

    # Per-regime BH-FDR-adjusted asterisks on the R\u00b2 panel only, one row per regime above the R\u00b2 envelope.
    if sig_lookup:
        top_val = -np.inf
        for regime in regime_order:
            grp = sub[sub["regime"] == regime]
            if grp.empty:
                continue
            v = grp["r2"].values
            if np.any(np.isfinite(v)):
                top_val = max(top_val, float(np.nanmax(v)))
            if "r2_ci_hi" in grp.columns:
                up = grp["r2_ci_hi"].values
                if np.any(np.isfinite(up)):
                    top_val = max(top_val, float(np.nanmax(up)))
        row_y = {"baseline_multimodal": top_val + 0.04,
                 "baseline_plus_12m":   top_val + 0.09,
                 "rolling":             top_val + 0.14}
        any_star = False
        for regime in ["baseline_multimodal", "baseline_plus_12m", "rolling"]:
            grp = sub[sub["regime"] == regime].sort_values("horizon_months")
            if grp.empty:
                continue
            color = REGIME_COLORS.get(regime, "#333")
            y = row_y[regime]
            for hm in grp["horizon_months"].values:
                p = sig_lookup.get((regime, int(hm)))
                stars = _stars_from_p(p) if p is not None else ""
                if not stars:
                    continue
                ax_r2.text(hm, y, stars,
                           ha="center", va="bottom",
                           fontsize=11, fontweight="bold", color=color,
                           zorder=10)
                any_star = True
        if any_star:
            cur_lo, cur_hi = ax_r2.get_ylim()
            ax_r2.set_ylim(cur_lo, max(cur_hi, top_val + 0.20))

    handles, labels_ = ax_r2.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f"{display} \u2014 Information Regime Comparison",
                 fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    if save:
        _save(fig, f"regime_comparison_{target}_{model}_paired")
    return fig


# =============================================================================
# Figure 6: Modality ablation heatmap
# =============================================================================

def plot_modality_ablation(ablation_df, metric="r2", save=True,
                           significance_df=None, bold_rule="bh_fdr",
                           cohen_f2_thresh=0.02, filename_suffix=""):
    """Modality ablation heatmap; cells show raw mean delta-metric, bold mask follows bold_rule (BH-FDR, optionally with Cohen f²)."""
    pivot = ablation_df.pivot_table(index="target_display", columns="dropped_modality",
                                    values=f"delta_{metric}", aggfunc="mean")
    name_map = {}
    if "dropped_modality_display" in ablation_df.columns:
        name_map = ablation_df.drop_duplicates("dropped_modality").set_index(
            "dropped_modality")["dropped_modality_display"].to_dict()
    pivot_display = pivot.rename(columns=name_map)

    sig_mask = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
    if (
        metric == "r2"
        and significance_df is not None
        and not significance_df.empty
        and {"target_display", "dropped_modality", "r2_p_adjusted"}.issubset(significance_df.columns)
    ):
        sig_pivot = significance_df.pivot_table(
            index="target_display", columns="dropped_modality",
            values="r2_p_adjusted", aggfunc="min",
        ).reindex(index=pivot.index, columns=pivot.columns)

        f2_pivot = None
        if bold_rule == "bh_fdr_and_cohen_f2":
            sig_df = significance_df.copy()
            denom = (1.0 - sig_df["mean_full_r2"]).clip(lower=1e-6)
            sig_df["__cohen_f2"] = (sig_df["mean_full_r2"] - sig_df["mean_ablated_r2"]) / denom
            f2_pivot = sig_df.pivot_table(
                index="target_display", columns="dropped_modality",
                values="__cohen_f2", aggfunc="mean",
            ).reindex(index=pivot.index, columns=pivot.columns)

        for r in pivot.index:
            for c in pivot.columns:
                p = sig_pivot.loc[r, c] if (r in sig_pivot.index and c in sig_pivot.columns) else float("nan")
                if not (pd.notnull(p) and p < 0.05):
                    continue
                if bold_rule == "bh_fdr":
                    sig_mask.loc[r, c] = True
                elif bold_rule == "bh_fdr_and_cohen_f2" and f2_pivot is not None:
                    f2 = f2_pivot.loc[r, c] if (r in f2_pivot.index and c in f2_pivot.columns) else float("nan")
                    if pd.notnull(f2) and abs(f2) > cohen_f2_thresh:
                        sig_mask.loc[r, c] = True

    fig, ax = plt.subplots(figsize=(10, max(3.5, len(pivot_display) * 0.55)))
    sns.heatmap(pivot_display, annot=False, cmap="RdBu_r", center=0,
                linewidths=0.5,
                cbar_kws={"label": f"\u0394 {_metric_label(metric)} (drop in performance)",
                          "shrink": 0.8},
                ax=ax)
    n_rows, n_cols = pivot.shape
    for i, r in enumerate(pivot.index):
        for j, c in enumerate(pivot.columns):
            v = pivot.loc[r, c]
            if pd.isnull(v):
                continue
            is_sig = bool(sig_mask.loc[r, c])
            ax.text(j + 0.5, i + 0.5, _fmt2(v, 3),
                    ha="center", va="center",
                    fontsize=9 if is_sig else 8,
                    fontweight="bold" if is_sig else "normal",
                    color="#000")
    title = "Modality Ablation \u2014 Leave-One-Out Contribution"
    if bold_rule == "bh_fdr_and_cohen_f2":
        title += f"  (bold: BH-FDR sig AND |Cohen f\u00b2| > {cohen_f2_thresh})"
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("Removed Modality", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    if save:
        _save(fig, f"modality_ablation_heatmap{_metric_suffix(metric)}{filename_suffix}")
    return fig


# =============================================================================
# Figure 7: Classification performance (AUPRC grouped bar chart) — REWRITTEN
# =============================================================================

def _stars_from_p(p):
    """Return a star string given a p-value (BH-corrected or raw)."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_classification_pr_curves(results_df, save=True, per_fold_df=None,
                                  exclude_targets=()):
    """Three-panel grouped bar chart (AUROC, AUPRC, MCC by target and model) with 95% CI error bars and BH-FDR significance asterisks."""
    clf = results_df[results_df["task_type"] == "classification"].copy()
    if exclude_targets:
        clf = clf[~clf["target"].isin(exclude_targets)]
    if clf.empty or "auprc" not in clf.columns:
        return None

    # Pull per-fold data for CI + significance computation. Fall back to the
    # legacy point-estimate behaviour if the per-fold CSV is unavailable.
    if per_fold_df is None:
        pf_path = RESULTS_DIR / "frontier_results_per_fold.csv"
        per_fold_df = pd.read_csv(pf_path) if pf_path.exists() else None

    pf_clf = None
    if per_fold_df is not None:
        pf_clf = per_fold_df[per_fold_df["task_type"] == "classification"].copy()
        if exclude_targets:
            pf_clf = pf_clf[~pf_clf["target"].isin(exclude_targets)]

    # Point-estimate aggregation (means across regime x horizon, for bar heights)
    agg_cols = {
        "auprc": ("auprc", "mean"),
        "auroc": ("auroc", "mean"),
        "prevalence": ("prevalence", "mean"),
    }
    if "mcc" in clf.columns:
        agg_cols["mcc"] = ("mcc", "mean")
    agg = clf.groupby(["target_display", "model"]).agg(**agg_cols).reset_index()

    targets = agg["target_display"].unique()
    # Active model bars (Majority Class is rendered separately as a
    # deterministic horizontal tick, NOT a bar with a misleading CI).
    model_order = ["logistic_regression", "random_forest_clf", "xgboost_clf"]
    models = [m for m in model_order if m in agg["model"].values]
    # Majority Class point estimates (per target) for the tick overlay.
    mc_vals = {}
    if "majority_class" in agg["model"].values:
        for t in targets:
            r = agg[(agg["target_display"] == t) & (agg["model"] == "majority_class")]
            if not r.empty:
                mc_vals[t] = {col: float(r[col].values[0]) for col in
                              ("auroc", "auprc", "mcc")
                              if col in r.columns and np.isfinite(r[col].values[0])}

    x = np.arange(len(targets))
    width = 0.22
    n_panels = 3 if "mcc" in agg.columns else 2
    # A single shared legend is added below the panels.
    fig, axes = plt.subplots(1, n_panels, figsize=(7.5 * n_panels, 5.5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    panel_specs = [
        {"metric": "auroc", "ylabel": "AUROC", "title": "(a) AUROC",
         "baseline_type": "chance"},
        {"metric": "auprc", "ylabel": "AUPRC", "title": "(b) AUPRC",
         "baseline_type": "prevalence"},
    ]
    if "mcc" in agg.columns:
        panel_specs.append(
            {"metric": "mcc", "ylabel": "MCC", "title": "(c) MCC",
             "baseline_type": "zero"})

    # Precompute per-(target_display, model, metric) fold-level arrays so CIs +
    # significance tests can be drawn without repeatedly filtering per panel.
    fold_arrays = {}
    if pf_clf is not None:
        for (td, m), g in pf_clf.groupby(["target_display", "model"]):
            fold_arrays[(td, m)] = {
                "auroc": g["auroc"].dropna().values if "auroc" in g.columns else np.array([]),
                "auprc": g["auprc"].dropna().values if "auprc" in g.columns else np.array([]),
                "mcc":   g["mcc"].dropna().values   if "mcc"   in g.columns else np.array([]),
            }

    from scipy.stats import wilcoxon
    from evaluation.statistical_tests import benjamini_hochberg

    # ── Pass 1: collect raw Wilcoxon p AND per-bar CI bounds for ALL 18 tests
    # (n_models × n_targets × n_metrics), so we can apply BH-FDR globally. ──
    raw_records = []  # list of (panel_idx, target_idx, model_idx, p_raw, lower_ci, baseline)
    for panel_idx, spec in enumerate(panel_specs):
        metric = spec["metric"]
        for ti, t in enumerate(targets):
            for mi, model in enumerate(models):
                fold_vals = fold_arrays.get((t, model), {}).get(metric, np.array([]))
                fold_vals = fold_vals[np.isfinite(fold_vals)] if fold_vals.size else fold_vals
                if spec["baseline_type"] == "chance":
                    baseline = 0.5
                elif spec["baseline_type"] == "zero":
                    baseline = 0.0
                else:
                    prow = agg[agg["target_display"] == t]["prevalence"]
                    baseline = float(prow.values[0]) if len(prow) else 0.0
                if fold_vals.size >= 3:
                    lo_ci = float(np.percentile(fold_vals, 2.5))
                    try:
                        diffs = fold_vals - baseline
                        if np.std(diffs) < 1e-10:
                            p = 1.0 if np.mean(diffs) <= 0 else 0.0
                        else:
                            res = wilcoxon(diffs, alternative="greater",
                                           zero_method="wilcox")
                            p = float(res.pvalue)
                    except Exception:
                        p = float("nan")
                else:
                    lo_ci = float("nan")
                    p = float("nan")
                raw_records.append({
                    "panel_idx": panel_idx, "target_idx": ti, "model_idx": mi,
                    "p_raw": p, "lower_ci": lo_ci, "baseline": baseline,
                })

    # BH-FDR across all 18 tests.
    p_arr = np.array([r["p_raw"] for r in raw_records], dtype=float)
    valid = np.isfinite(p_arr)
    p_adj_arr = np.full_like(p_arr, np.nan)
    if valid.any():
        p_adj_arr[valid] = benjamini_hochberg(p_arr[valid], 0.05)
    for rec, padj in zip(raw_records, p_adj_arr):
        rec["p_adj"] = padj
    # Index by (panel_idx, target_idx, model_idx) for the second pass.
    sig_index = {(r["panel_idx"], r["target_idx"], r["model_idx"]): r
                 for r in raw_records}

    # ── Pass 2: render the bars; place asterisks only where p_adj < α AND
    # the lower 95% CI bound exceeds the panel's no-skill baseline. ──
    panel_ymax = {}
    panel_handles = {}
    panel_records = {p["metric"]: [] for p in panel_specs}

    for panel_idx, (ax, spec) in enumerate(zip(axes, panel_specs)):
        metric = spec["metric"]
        ymax_panel = 0.0
        for i, model in enumerate(models):
            vals, errs_lo, errs_hi = [], [], []
            for ti, t in enumerate(targets):
                row = agg[(agg["target_display"] == t) & (agg["model"] == model)]
                v = row[metric].values[0] if len(row) > 0 else 0
                v = v if np.isfinite(v) else 0
                vals.append(v)
                fold_vals = fold_arrays.get((t, model), {}).get(metric, np.array([]))
                fold_vals = fold_vals[np.isfinite(fold_vals)] if fold_vals.size else fold_vals
                if fold_vals.size >= 3:
                    lo = float(np.percentile(fold_vals, 2.5))
                    hi = float(np.percentile(fold_vals, 97.5))
                    mean_v = float(np.mean(fold_vals))
                    errs_lo.append(max(0, mean_v - lo))
                    errs_hi.append(max(0, hi - mean_v))
                    ymax_panel = max(ymax_panel, hi)
                else:
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
            color = _MODEL_COLORS.get(model, f"C{i}")
            bar_x = x + i * width - width * (len(models) - 1) / 2
            bars = ax.bar(bar_x, vals, width,
                          label=_model_label(model), color=color,
                          edgecolor="white", linewidth=0.5)
            panel_handles[model] = bars[0]
            if any(e > 0 for e in errs_hi):
                ax.errorbar(bar_x, vals, yerr=[errs_lo, errs_hi],
                            fmt="none", color="#222", capsize=5, linewidth=1.5,
                            zorder=4)
            # Significance: BH-FDR p_adj AND lower 95% CI > baseline.
            for ti, xb in enumerate(bar_x):
                rec = sig_index.get((panel_idx, ti, i))
                if rec is None:
                    panel_records[metric].append((xb, float("nan")))
                    continue
                p_adj = rec["p_adj"]
                lo_ci = rec["lower_ci"]
                baseline = rec["baseline"]
                ci_clears = (np.isfinite(lo_ci) and np.isfinite(baseline)
                             and lo_ci > baseline)
                p_ok = np.isfinite(p_adj) and p_adj < 0.05
                effective_p = p_adj if (p_ok and ci_clears) else float("nan")
                panel_records[metric].append((xb, effective_p))
        panel_ymax[spec["metric"]] = ymax_panel

        # No-skill baseline lines.
        if spec["baseline_type"] == "chance":
            ax.axhline(0.5, color="#666", linestyle="--", linewidth=1.2, zorder=5)
        elif spec["baseline_type"] == "prevalence":
            # Per-target dashed prevalence segment with a label at the right edge.
            for j, t in enumerate(targets):
                prev = agg[agg["target_display"] == t]["prevalence"].values[0]
                ax.hlines(prev, j - 0.45, j + 0.45, color="#666", linestyle="--",
                          linewidth=1.2, zorder=5)
                ax.text(j + 0.46, prev, _fmt2(prev),
                        fontsize=12.5, va="center", ha="left", color="#444")
        elif spec["baseline_type"] == "zero":
            ax.axhline(0, color="#666", linestyle="--", linewidth=1.2, zorder=5)

        # Light gray gridlines on y-axis only.
        ax.yaxis.grid(True, which="major", color="#dddddd", linewidth=0.6,
                      linestyle="-", zorder=0)
        ax.set_axisbelow(True)

        ax.set_xticks(x)
        ax.set_xticklabels(targets, fontsize=15)
        ax.set_ylabel(spec["ylabel"], fontsize=15)
        # Title left-anchored to clear the centred asterisks row.
        ax.set_title(spec["title"], fontweight="bold", fontsize=12,
                     loc="left", pad=8)
        ax.tick_params(axis="y", labelsize=9)
        ymax = panel_ymax.get(metric, 0.0)
        # Uniform y-limits [0.0, 1.0] so the asterisks row aligns across panels.
        ax.set_ylim(0.0, 1.0)

        # Leveled asterisks above each bar at a fixed absolute y across panels.
        star_y = 0.96
        for xb, p in panel_records[metric]:
            stars = _stars_from_p(p) if np.isfinite(p) else ""
            if stars:
                ax.text(xb, star_y, stars, ha="center", va="center",
                        fontsize=14, color="#222", zorder=7)

        # Majority Class — render as a horizontal tick (deterministic; no bar, no CI).
        for j, t in enumerate(targets):
            mc = mc_vals.get(t, {})
            mc_v = mc.get(metric)
            if mc_v is None or not np.isfinite(mc_v):
                continue
            ax.hlines(mc_v, j - 0.45, j + 0.45,
                      colors="#777777", linewidth=2.2, zorder=6)
        # Add a single proxy entry to the (figure-level) legend so readers
        # can see the Majority Class tick is referenced.

    # Shared legend at the top so it never overlaps any panel.
    legend_handles = [panel_handles[m] for m in models if m in panel_handles]
    legend_labels  = [_model_label(m)   for m in models if m in panel_handles]
    if mc_vals:
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D([0], [0], color="#777777",
                                     linewidth=2.2, linestyle="-"))
        legend_labels.append("Majority Class (no-skill ref.)")
    fig.legend(legend_handles, legend_labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=len(legend_handles),
               frameon=False, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        _save(fig, "classification_auprc")
    return fig


# =============================================================================
# Figure 8: Target availability (missingness) heatmap
# =============================================================================

def plot_missingness_heatmap(availability_pivot, save=True):
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(availability_pivot) * 0.45)))
    sns.heatmap(availability_pivot, annot=True, fmt=".0f", cmap="YlOrBr_r",
                linewidths=0.5, cbar_kws={"label": "N patients", "shrink": 0.8},
                ax=ax, annot_kws={"fontsize": 9})
    ax.set_title("Target Availability by Visit", fontweight="bold")
    ax.set_xlabel("Visit (months)", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    if save:
        _save(fig, "missingness_heatmap")
    return fig


# =============================================================================
# Figure 9: Test-set sample sizes
# =============================================================================

def plot_sample_sizes(sample_df, save=True):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    targets = sample_df["target_display"].unique()
    horizons = sorted(sample_df["horizon_months"].unique())
    x = np.arange(len(horizons))
    width = 0.8 / len(targets)

    cb_cycle = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
                "#F0E442", "#999999", "#332288", "#117733", "#882255", "#44AA99"]
    for i, t in enumerate(targets):
        sub = sample_df[sample_df["target_display"] == t].groupby("horizon_months")["n_test"].max()
        vals = [int(sub.loc[h]) if h in sub.index else 0 for h in horizons]
        ax.bar(x + i * width, vals, width, label=t, color=cb_cycle[i % len(cb_cycle)])

    ax.set_xticks(x + width * (len(targets) - 1) / 2)
    ax.set_xticklabels([f"{h}m" for h in horizons], fontsize=10)
    ax.set_xlabel("Forecast Horizon", fontsize=10)
    ax.set_ylabel("N (test set)", fontsize=10)
    ax.set_title("Test-Set Sample Sizes by Target and Horizon", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right",
              bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    if save:
        _save(fig, "sample_sizes")
    return fig


# =============================================================================
# Figure 10: Cohort attrition bar chart
# =============================================================================

def plot_survivorship_attrition(attrition_df=None, save=True):
    if attrition_df is None:
        path = RESULTS_DIR / "tables" / "survivorship_summary.csv"
        if not path.exists():
            return None
        attrition_df = pd.read_csv(path)

    # Exclude V12 — not a modelled endpoint
    attrition_df = attrition_df[attrition_df["visit"] != "V12"].copy()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(attrition_df)), attrition_df["n_patients"],
                  color="#457B9D", edgecolor="white", linewidth=0.5)
    for i, (_, row) in enumerate(attrition_df.iterrows()):
        ax.text(i, row["n_patients"] + 25, f"{row['pct_of_baseline']:.0f}%",
                ha="center", fontsize=9, fontweight="bold")
        ax.text(i, row["n_patients"] - 80,
                f"n={int(row['n_patients'])}", ha="center", fontsize=7.5,
                color="white", fontweight="bold")

    ax.set_xticks(range(len(attrition_df)))
    labels = [f"{row['visit']}\n({int(row['months'])}m)" for _, row in attrition_df.iterrows()]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Visit (months from baseline)", fontsize=10)
    ax.set_ylabel("N Patients with Data", fontsize=10)
    ax.set_title("Cohort Attrition over Follow-up", fontweight="bold")
    ax.set_ylim(0, attrition_df["n_patients"].max() * 1.12)
    fig.tight_layout()
    if save:
        _save(fig, "survivorship_attrition")
    return fig


# =============================================================================
# Figure 11: Survivorship bias effect sizes
# =============================================================================

def plot_survivorship_effect_sizes(bias_df=None, save=True):
    if bias_df is None:
        path = RESULTS_DIR / "tables" / "survivorship_bias.csv"
        if not path.exists():
            return None
        bias_df = pd.read_csv(path)

    # Nicer feature labels
    feature_labels = {
        "NP3TOT": "UPDRS-III Total",
        "TREMOR_SUBSCORE": "UPDRS-III Tremor",
        "BRADY_SUBSCORE": "UPDRS-III Bradykinesia",
        "PIGD_SUBSCORE": "UPDRS-III PIGD",
        "MCATOT": "MoCA Total",
        "DELAYED_RECALL_SUM": "MoCA Delayed Recall",
        "NP1RTOT": "UPDRS-I Total",
        "ORTHO_SYS_DROP": "Orthostatic SBP Drop",
        "ORTHO_DIA_DROP": "Orthostatic DBP Drop",
    }
    bias_df = bias_df.copy()
    bias_df["feature_display"] = bias_df["feature"].map(
        lambda f: feature_labels.get(f, f))

    pivot = bias_df.pivot_table(index="feature_display", columns="visit",
                                values="cohens_d", aggfunc="mean")
    visit_order = ["V04", "V06", "V08", "V10"]
    pivot = pivot.reindex(columns=[v for v in visit_order if v in pivot.columns])

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=pivot.applymap(_fmt2), fmt="",
                cmap="RdBu_r", center=0,
                linewidths=0.5, cbar_kws={"label": "Cohen's d (survivors \u2212 dropouts)",
                                           "shrink": 0.8},
                ax=ax, annot_kws={"fontsize": 9})
    ax.set_title("Survivorship Bias \u2014 Baseline Differences by Follow-up Duration",
                 fontweight="bold")
    ax.set_xlabel("Last Observed Visit", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    if save:
        _save(fig, "survivorship_effect_sizes")
    return fig


# =============================================================================
# Figure 12: Horizon diagnosis (dual-axis metric + sample size)
# =============================================================================

def plot_horizon_diagnosis_paired(results_df, target, model="xgboost",
                                  regime="baseline_only", save=True):
    """Two-panel paired horizon diagnosis: R² (left) and RMSE (right) vs. horizon, each with CI bands and a test-N overlay."""
    sub = results_df[(results_df["model"] == model)
                     & (results_df["regime"] == regime)
                     & (results_df["target"] == target)].sort_values("horizon_months")
    if sub.empty:
        return None
    if "rmse" not in sub.columns:
        return None
    domain = sub["target_domain"].iloc[0]
    color = PALETTE.get(domain, "#333")
    display = sub["target_display"].iloc[0]
    months = sub["horizon_months"].values
    n_test = (sub["n_test"].values if "n_test" in sub.columns
              else sub["n_test_mean"].values)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    def _panel(ax, metric, ylabel, panel_title):
        vals = sub[metric].values
        lo = sub[f"{metric}_ci_lo"].values if f"{metric}_ci_lo" in sub.columns else None
        hi = sub[f"{metric}_ci_hi"].values if f"{metric}_ci_hi" in sub.columns else None
        ax.plot(months, vals, "o-", color=color, linewidth=1.6, markersize=6, zorder=3)
        if lo is not None and np.any(np.isfinite(lo)):
            valid = np.isfinite(lo) & np.isfinite(hi)
            ax.fill_between(months[valid], lo[valid], hi[valid], alpha=0.15,
                            color=color, zorder=1)
        ax.set_xlabel("Forecast Horizon (months)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.set_title(panel_title, fontweight="bold", fontsize=11)
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        # Test-N overlay (right axis)
        ax2 = ax.twinx()
        ax2.bar(months, n_test, width=3.0, alpha=0.18, color="gray", zorder=0)
        ax2.set_ylabel("N (test)", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelsize=8, colors="gray")

    _panel(axes[0], "r2", "R²", "(a) R² (variance explained)")
    _panel(axes[1], "rmse", "RMSE (target units)", "(b) RMSE (absolute error)")

    fig.suptitle(f"{display} — Horizon Diagnosis ({_model_label(model)}, "
                 f"{regime.replace('_', ' ').title()})",
                 fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    if save:
        _save(fig, f"horizon_diagnosis_{target}_{regime}_paired")
    return fig


def plot_model_comparison_paired(results_df, regime="baseline_only",
                                 horizon="V08", save=True):
    """Two-panel paired model comparison at one (regime, horizon): R² bars (left) and RMSE bars (right)."""
    sub = results_df[(results_df["regime"] == regime)
                     & (results_df["horizon"] == horizon)].copy()
    sub = sub[sub["model"] != "population_mean"]
    if sub.empty or "rmse" not in sub.columns:
        return None
    targets = sub.groupby("target_display")["r2"].max().sort_values(ascending=False).index
    models = [m for m in ["locf", "lme", "ridge", "elastic_net",
                          "random_forest", "xgboost"] if m in sub["model"].values]
    x = np.arange(len(targets))
    width = 0.75 / max(len(models), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    def _panel(ax, metric, ylabel, panel_title, neg_ok=False):
        ci_lo, ci_hi = f"{metric}_ci_lo", f"{metric}_ci_hi"
        for i, model in enumerate(models):
            vals, errs_lo, errs_hi = [], [], []
            for t in targets:
                row = sub[(sub["target_display"] == t) & (sub["model"] == model)]
                v = float(row[metric].values[0]) if len(row) else 0.0
                vals.append(v)
                if ci_lo in sub.columns and len(row):
                    lo = row[ci_lo].values[0] if np.isfinite(row[ci_lo].values[0]) else v
                    hi = row[ci_hi].values[0] if np.isfinite(row[ci_hi].values[0]) else v
                    errs_lo.append(max(0, v - lo))
                    errs_hi.append(max(0, hi - v))
                else:
                    errs_lo.append(0); errs_hi.append(0)
            color = _MODEL_COLORS.get(model, f"C{i}")
            bx = x + i * width - width * (len(models) - 1) / 2
            ax.bar(bx, vals, width, label=_model_label(model), color=color,
                   edgecolor="white", linewidth=0.5)
            if any(e > 0 for e in errs_hi):
                ax.errorbar(bx, vals, yerr=[errs_lo, errs_hi],
                            fmt="none", color="#333", capsize=2.5, linewidth=0.8)
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        if neg_ok:
            ax.axhline(0, color="#666", linewidth=0.7, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(panel_title, fontweight="bold", fontsize=11)

    _panel(axes[0], "r2", "R²", "(a) R² (variance explained, higher = better)", neg_ok=True)
    _panel(axes[1], "rmse", "RMSE (target units)", "(b) RMSE (absolute error, lower = better)", neg_ok=False)
    # Shared legend below both panels (avoid overlap with bars).
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=len(models),
               frameon=False, fontsize=10)

    months = VISIT_SCHEDULE.get(horizon, "?")
    fig.suptitle(f"Model Comparison — {regime.replace('_', ' ').title()}, "
                 f"{months}-month Horizon", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    if save:
        _save(fig, f"model_comparison_{regime}_{horizon}_paired")
    return fig


def plot_horizon_diagnosis(results_df, target=None, model="xgboost",
                           regime="baseline_only", metric="r2", save=True):
    """Dual-axis plot: metric vs horizon (with CI) plus sample-size overlay to diagnose survivorship-driven non-monotonicity."""
    sub = results_df[
        (results_df["model"] == model) & (results_df["regime"] == regime)
    ].copy()

    if target is not None:
        sub = sub[sub["target"] == target]
        targets = [target]
    else:
        targets = sub["target"].unique()

    n_targets = len(targets)
    if n_targets == 0:
        return None

    fig, axes = plt.subplots(n_targets, 1, figsize=(6, 2.5 * n_targets),
                             sharex=True)
    if n_targets == 1:
        axes = [axes]

    ci_lo_col = f"{metric}_ci_lo"
    ci_hi_col = f"{metric}_ci_hi"

    for ax, tgt in zip(axes, targets):
        tgt_data = sub[sub["target"] == tgt].sort_values("horizon_months")
        if tgt_data.empty:
            continue

        display = tgt_data["target_display"].iloc[0]
        months = tgt_data["horizon_months"].values
        metric_vals = tgt_data[metric].values if metric in tgt_data.columns else np.zeros(len(months))
        n_test = tgt_data["n_test"].values if "n_test" in tgt_data.columns else tgt_data["n_test_mean"].values
        domain = tgt_data["target_domain"].iloc[0]

        color = PALETTE.get(domain, "#333")
        ax.plot(months, metric_vals, "o-", color=color, linewidth=1.5, markersize=5)

        if ci_lo_col in tgt_data.columns:
            lo = tgt_data[ci_lo_col].values
            hi = tgt_data[ci_hi_col].values
            valid = np.isfinite(lo) & np.isfinite(hi)
            if valid.any():
                ax.fill_between(months[valid], lo[valid], hi[valid],
                                alpha=0.15, color=color)

        ax.set_ylabel(_metric_label(metric), fontsize=9)
        ax.set_title(display, fontsize=10, fontweight="bold")

        # Sample size on secondary axis
        ax2 = ax.twinx()
        ax2.bar(months, n_test, width=2, alpha=0.15, color="gray")
        ax2.set_ylabel("N (test)", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelsize=8, colors="gray")

    axes[-1].set_xlabel("Forecast Horizon (months)", fontsize=10)
    fig.suptitle(f"Horizon Diagnosis: {_metric_label(metric)} vs. Sample Size",
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    suffix = f"_{target}" if target else "_all"
    if save:
        _save(fig, f"horizon_diagnosis{suffix}_{regime}{_metric_suffix(metric)}")
    return fig


# =============================================================================
# Figure 13: Prediction scatter (observed vs predicted)
# =============================================================================

# =============================================================================
# Calibration curves for classification tasks
# =============================================================================

def plot_calibration_curves(calibration_data, output_dir):
    """Plot reliability diagrams per classification target, one panel per horizon (models averaged across regimes)."""
    # Group by target
    by_target = {}
    for cal in calibration_data:
        target = cal["target"]
        by_target.setdefault(target, []).append(cal)

    model_colors = {
        "logistic_regression": "#56B4E9",
        "random_forest_clf":   "#0072B2",
        "xgboost_clf":         "#D55E00",
    }
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest_clf":   "Random Forest",
        "xgboost_clf":         "XGBoost",
    }

    # Standard bin edges for aggregation (10 equal bins from 0 to 1)
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    horizon_labels = {"V04": "12 mo", "V06": "24 mo", "V08": "36 mo",
                      "V10": "48 mo", "V12": "60 mo"}
    horizon_order = ["V04", "V06", "V08", "V10", "V12"]

    for target, cals in by_target.items():
        # Identify available horizons for this target
        horizons_present = sorted(
            {c["horizon"] for c in cals if c.get("horizon")},
            key=lambda h: horizon_order.index(h) if h in horizon_order else 99,
        )
        if not horizons_present:
            continue

        n_panels = len(horizons_present)
        fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.8),
                                 squeeze=False)
        axes = axes.flatten()

        for pidx, horizon in enumerate(horizons_present):
            ax = axes[pidx]
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)

            horizon_cals = [c for c in cals if c.get("horizon") == horizon]

            # Group by model, average across regimes within this horizon
            models_in_horizon = {}
            for cal in horizon_cals:
                model = cal["model"]
                models_in_horizon.setdefault(model, []).append(cal)

            for model, model_cals in models_in_horizon.items():
                # Average the fraction_positive across regimes at matched bins
                all_fp = []
                all_mp = []
                for cal in model_cals:
                    mp = np.array(cal["mean_predicted"])
                    fp = np.array(cal["fraction_positive"])
                    valid = np.isfinite(mp) & np.isfinite(fp)
                    if valid.any():
                        all_mp.append(mp[valid])
                        all_fp.append(fp[valid])

                if not all_fp:
                    continue

                # Bin-align: digitize each curve's points into the standard
                # bins, then average
                binned_fp = np.full(10, np.nan)
                binned_counts = np.zeros(10)
                for mp_arr, fp_arr in zip(all_mp, all_fp):
                    indices = np.digitize(mp_arr, bin_edges) - 1
                    indices = np.clip(indices, 0, 9)
                    for i, idx in enumerate(indices):
                        if np.isfinite(fp_arr[i]):
                            if np.isnan(binned_fp[idx]):
                                binned_fp[idx] = 0.0
                            binned_fp[idx] += fp_arr[i]
                            binned_counts[idx] += 1

                has_data = binned_counts > 0
                if not has_data.any():
                    continue
                binned_fp[has_data] /= binned_counts[has_data]

                mean_ece = np.mean([c["ece"] for c in model_cals])
                color = model_colors.get(model, "#666666")
                label = f'{model_labels.get(model, model)} (ECE={mean_ece:.2f})'
                ax.plot(bin_centers[has_data], binned_fp[has_data], "o-",
                        color=color, label=label, markersize=4, linewidth=1.2)

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            ax.set_title(horizon_labels.get(horizon, horizon), fontsize=10,
                         fontweight="bold")
            ax.grid(True, alpha=0.15)
            if pidx == 0:
                ax.set_ylabel("Fraction of positives", fontsize=9)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("Predicted probability", fontsize=9)
            ax.tick_params(labelsize=8)

        # Add legend to last panel
        axes[-1].legend(loc="lower right", fontsize=7, frameon=False)

        target_display = target.replace("_", " ").title()
        fig.suptitle(f"Calibration: {target_display}", fontweight="bold",
                     fontsize=12, y=1.02)
        fig.tight_layout()

        fig.savefig(output_dir / f"calibration_{target}.pdf", dpi=300,
                    bbox_inches="tight")
        fig.savefig(output_dir / f"calibration_{target}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# Figure: Model ranking bar chart (R²-centric summary across all tasks)
# =============================================================================

def plot_model_ranking(ranking_df, metric: str = "r2", save: bool = True):
    """Horizontal bar chart of per-model mean metric (with std) across all regression tasks."""
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in ranking_df.columns:
        raise KeyError(f"{mean_col} not in ranking_df (cols: {list(ranking_df.columns)})")

    d = ranking_df[[c for c in ["model", mean_col, std_col] if c in ranking_df.columns]].copy()
    # Drop the population-mean row: it sits at zero R² by construction.
    d = d[~d["model"].isin({"population_mean", "pop_mean", "majority_class"})]
    d = d.sort_values(mean_col, ascending=True)
    d["label"] = d["model"].map(_model_label)
    d["color"] = d["model"].map(_MODEL_COLORS).fillna("#666666")

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    y = np.arange(len(d))
    xerr = d[std_col].values if std_col in d.columns else None
    ax.barh(
        y, d[mean_col].values,
        xerr=xerr,
        color=d["color"].values, edgecolor="white",
        error_kw={"ecolor": "#333333", "elinewidth": 0.9, "capsize": 3},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"].values)
    ax.set_xlabel(f"Mean {_metric_label(metric)} across all regression tasks")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    for yi, v in zip(y, d[mean_col].values):
        ax.text(v + 0.005, yi, _fmt2(v), va="center", fontsize=8, color="#222")
    means_arr = d[mean_col].values
    if xerr is not None and len(xerr) == len(means_arr):
        bar_lo = float(np.nanmin(means_arr - xerr))
        bar_hi = float(np.nanmax(means_arr + xerr))
    else:
        bar_lo = float(np.nanmin(means_arr))
        bar_hi = float(np.nanmax(means_arr))
    ax.set_xlim(min(0.0, bar_lo) - 0.03,
                max(0.35, bar_hi) + 0.06)
    ax.set_title(
        f"Model ranking by mean {_metric_label(metric)} (135 regression tasks)",
        fontweight="bold", fontsize=11,
    )
    fig.tight_layout()
    if save:
        _save(fig, f"model_ranking{_metric_suffix(metric)}")
    return fig


# =============================================================================
# Figure: R²/Spearman divergence scatter
# =============================================================================

def plot_r2_spearman_divergence(frontier_df, save: bool = True,
                                anchor: str = "xgboost", baseline: str = "locf"):
    """Scatter of the per-task anchor−baseline (default XGBoost−LOCF) gap on (Spearman, R²) axes."""
    req = {"task_key", "model", "r2", "spearman"}
    if not req.issubset(frontier_df.columns):
        missing = req - set(frontier_df.columns)
        raise KeyError(f"frontier_df missing columns: {missing}")

    piv_r2 = frontier_df.pivot_table(index="task_key", columns="model", values="r2",
                                     aggfunc="mean")
    piv_sp = frontier_df.pivot_table(index="task_key", columns="model", values="spearman",
                                     aggfunc="mean")
    if anchor not in piv_r2.columns or baseline not in piv_r2.columns:
        return None

    dr = (piv_r2[anchor] - piv_r2[baseline]).dropna()
    ds = (piv_sp[anchor] - piv_sp[baseline]).dropna()
    common = dr.index.intersection(ds.index)
    dr = dr.loc[common]
    ds = ds.loc[common]

    # Domain lookup for colour
    domain_map = (
        frontier_df.drop_duplicates("task_key")
        .set_index("task_key")["target_domain"].to_dict()
        if "target_domain" in frontier_df.columns else {}
    )
    dom_colors = {
        "motor": "#D55E00",
        "cognitive": "#0072B2",
        "nonmotor": "#009E73",
        "autonomic": "#CC79A7",
    }
    colors = [dom_colors.get(domain_map.get(k, ""), "#666666") for k in dr.index]

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    lim = max(abs(dr).max(), abs(ds).max(), 0.1) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="#aaaaaa", linewidth=0.8,
            linestyle=":", alpha=0.8, label="1:1")

    ax.scatter(ds.values, dr.values, c=colors, s=26, alpha=0.78,
               edgecolors="white", linewidth=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"\u0394 Spearman \u03c1 ({_model_label(anchor)} \u2212 {_model_label(baseline)})")
    ax.set_ylabel(f"\u0394 R\u00b2 ({_model_label(anchor)} \u2212 {_model_label(baseline)})")
    ax.set_title(
        f"Per-task R\u00b2 vs. Spearman gap: {_model_label(anchor)} vs. {_model_label(baseline)}",
        fontweight="bold", fontsize=11,
    )

    # Legend for domains actually present
    present = sorted({domain_map.get(k, "other") for k in dr.index})
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=dom_colors.get(d, "#666666"),
                          markeredgecolor="white", markersize=6, label=d.capitalize())
               for d in present if d in dom_colors]
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    if save:
        _save(fig, f"divergence_scatter_{anchor}_vs_{baseline}")
    return fig
