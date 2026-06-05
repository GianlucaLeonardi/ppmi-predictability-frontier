#!/usr/bin/env python3
"""
Post-hoc hyperparameter analysis: top-5 configurations per model.

NOT part of the automatic pipeline. Run manually after a completed benchmark:

    cd /path/to/ppmi-predictability-frontier
    python scripts/analyze_hyperparameters.py

Reads: results/frontier_results_per_fold.csv
Writes: results/reports/hyperparameter_analysis.md
        results/tables/hp_top5_per_model.csv

For each tunable model, shows the top-5 selected HP configs, their mean
performance, per-regime breakdown, and grid-edge guidance.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import RESULTS_DIR

# Keys to strip from the HP dict (these are fixed, not tuned)
STRIP_KEYS = {
    "random_state", "verbosity", "max_iter", "n_jobs", "class_weight",
    "scale_pos_weight", "tree_method", "device", "subsample",
    "colsample_bytree", "max_features",
}

TUNABLE_MODELS = [
    "ridge", "elastic_net", "random_forest", "xgboost",
    "logistic_regression", "random_forest_clf", "xgboost_clf",
]

MODEL_DISPLAY = {
    "ridge": "Ridge", "elastic_net": "Elastic Net",
    "random_forest": "Random Forest (reg)", "xgboost": "XGBoost (reg)",
    "logistic_regression": "Logistic Regression",
    "random_forest_clf": "Random Forest (clf)", "xgboost_clf": "XGBoost (clf)",
}


def clean_params(raw: str) -> dict:
    """Parse JSON params string, strip fixed keys, return tuned-only dict."""
    try:
        d = json.loads(raw) if isinstance(raw, str) else {}
        return {k: v for k, v in sorted(d.items()) if k not in STRIP_KEYS}
    except Exception:
        return {}


def main():
    pf_path = RESULTS_DIR / "frontier_results_per_fold.csv"
    if not pf_path.exists():
        print(f"ERROR: {pf_path} not found. Run the pipeline first.")
        sys.exit(1)

    pf = pd.read_csv(pf_path)
    pf = pf[pf["model"].isin(TUNABLE_MODELS)].copy()
    pf = pf[pf["best_params"].notna() & (pf["best_params"].astype(str) != "")]

    if pf.empty:
        print("No tunable-model rows with best_params found.")
        sys.exit(1)

    pf["hp_clean"] = pf["best_params"].apply(lambda s: json.dumps(clean_params(s), sort_keys=True))
    pf["hp_dict"] = pf["best_params"].apply(clean_params)

    # Primary metric for ranking: R² for regression/ranking, MCC for classification
    def _metric(row):
        if row.get("task_type") == "classification":
            return row.get("mcc", float("nan"))
        return row.get("r2", float("nan"))

    pf["primary_metric"] = pf.apply(_metric, axis=1)

    # ── Per-model top-5 analysis ─────────────────────────────────────
    all_rows = []
    report_lines = ["# Hyperparameter Analysis — Top 5 Configs per Model\n"]

    for model in TUNABLE_MODELS:
        sub = pf[pf["model"] == model]
        if sub.empty:
            continue
        mname = MODEL_DISPLAY.get(model, model)
        report_lines.append(f"\n## {mname}\n")
        report_lines.append(f"Total fold-level fits: {len(sub)}")

        counts = sub["hp_key" if "hp_key" in sub.columns else "hp_clean"].value_counts()
        total = len(sub)

        # Get unique HP keys (up to top-5)
        top_n = min(5, len(counts))
        report_lines.append(f"Distinct configs observed: {len(counts)}")
        report_lines.append(f"Top {top_n} shown:\n")

        header = f"| Rank | Config | Freq | Freq % | Mean R²/MCC | Std |"
        sep = "|------|--------|------|--------|-------------|-----|"
        report_lines.append(header)
        report_lines.append(sep)

        for rank, (hp_key, count) in enumerate(counts.head(top_n).items(), 1):
            pct = 100 * count / total
            mask = sub["hp_clean"] == hp_key
            mean_perf = sub.loc[mask, "primary_metric"].mean()
            std_perf = sub.loc[mask, "primary_metric"].std()
            # Pretty-print the config
            try:
                hp_dict = json.loads(hp_key)
                hp_short = ", ".join(f"{k}={v}" for k, v in hp_dict.items())
            except Exception:
                hp_short = hp_key

            report_lines.append(
                f"| {rank} | {hp_short} | {count} | {pct:.1f}% | "
                f"{mean_perf:.3f} | {std_perf:.3f} |"
            )
            all_rows.append({
                "model": model,
                "model_display": mname,
                "rank": rank,
                "config": hp_short,
                "config_json": hp_key,
                "count": count,
                "freq_pct": round(pct, 1),
                "mean_primary_metric": round(mean_perf, 4),
                "std_primary_metric": round(std_perf, 4),
            })

        # Grid boundary check: is the best config at the edge of the grid?
        report_lines.append("")
        best_hp = clean_params(counts.index[0]) if isinstance(counts.index[0], str) else json.loads(counts.index[0])
        from configs.config import MODELS_REGRESSION, MODELS_CLASSIFICATION
        model_cfg = {**MODELS_REGRESSION, **MODELS_CLASSIFICATION}.get(model, {})
        grid = model_cfg.get("param_grid", {})
        edge_warnings = []
        for param, val in best_hp.items():
            if param in grid:
                grid_vals = sorted(grid[param])
                if val == grid_vals[0]:
                    edge_warnings.append(f"  ⚠ {param}={val} is at the LOW edge of grid {grid_vals}")
                elif val == grid_vals[-1]:
                    edge_warnings.append(f"  ⚠ {param}={val} is at the HIGH edge of grid {grid_vals}")
        if edge_warnings:
            report_lines.append("**Grid-edge warnings** (consider expanding):")
            report_lines.extend(edge_warnings)
        else:
            report_lines.append("No grid-edge warnings — top config sits in the interior of all grids.")

        # Per-regime breakdown of the top-1 config frequency
        report_lines.append("\nTop-1 config frequency by regime:")
        top1_key = counts.index[0]
        for regime in sub["regime"].unique():
            rsub = sub[sub["regime"] == regime]
            rcount = (rsub["hp_clean"] == top1_key).sum()
            rpct = 100 * rcount / len(rsub)
            report_lines.append(f"  {regime}: {rcount}/{len(rsub)} ({rpct:.0f}%)")

        report_lines.append("")

    # ── Save outputs ─────────────────────────────────────────────────
    reports_dir = RESULTS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "hyperparameter_analysis.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report: {report_path}")

    csv_path = tables_dir / "hp_top5_per_model.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"CSV:    {csv_path}")

    # ── Console summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    for model in TUNABLE_MODELS:
        sub = pf[pf["model"] == model]
        if sub.empty:
            continue
        counts = sub["hp_clean"].value_counts()
        top1_pct = 100 * counts.iloc[0] / len(sub)
        n_distinct = len(counts)
        mname = MODEL_DISPLAY.get(model, model)
        stability = "STABLE" if top1_pct > 50 else "MODERATE" if top1_pct > 25 else "UNSTABLE"
        print(f"  {mname:<25} top-1 freq={top1_pct:5.1f}%  distinct={n_distinct:3d}  [{stability}]")
    print(f"\nFull report: {report_path}")
    print(f"CSV table:   {csv_path}")


if __name__ == "__main__":
    main()
