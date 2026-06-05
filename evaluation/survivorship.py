"""
Survivorship bias analysis.

Characterizes attrition and compares baseline characteristics of patients
who survive to later visits vs. those who drop out.
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import COHORT_PRIMARY, RESULTS_DIR, VISIT_ORDER, VISIT_SCHEDULE
from utils.logging_utils import get_logger

log = get_logger("survivorship")


def characterize_survivorship(
    longitudinal: Dict[str, pd.DataFrame] = None,
    cohort_patnos: np.ndarray = None,
) -> pd.DataFrame:
    if longitudinal is None:
        from utils.io import get_cohort_patnos
        from data_preprocessing.build_dataset import load_raw_longitudinal
        cohort_patnos = get_cohort_patnos(COHORT_PRIMARY)
        longitudinal = {}
        for mod in ["updrs3", "moca", "updrs1", "vital_signs"]:
            df = load_raw_longitudinal(mod)
            df = df[df["PATNO"].isin(cohort_patnos)]
            longitudinal[mod] = df

    # Baseline characteristics for the survivorship-bias check are exactly the nine
    # regression prediction endpoints (Fig. 1/2), so the survivorship figure aligns
    # one-to-one with the endpoints whose late-horizon R^2 could be biased by attrition.
    key_features = {
        "updrs3": ["NP3TOT", "TREMOR_SUBSCORE", "BRADY_SUBSCORE", "PIGD_SUBSCORE"],
        "moca": ["MCATOT", "DELAYED_RECALL_SUM"],
        "updrs1": ["NP1RTOT"],
        "vital_signs": ["ORTHO_SYS_DROP", "ORTHO_DIA_DROP"],
    }

    bl_all = None
    for mod, feats in key_features.items():
        df = longitudinal.get(mod)
        if df is None or df.empty:
            continue
        bl_df = df[df["VISIT"] == "BL"][["PATNO"] + [f for f in feats if f in df.columns]]
        bl_all = bl_df if bl_all is None else bl_all.merge(bl_df, on="PATNO", how="outer")

    if bl_all is None or bl_all.empty:
        return pd.DataFrame()

    ref_df = longitudinal.get("updrs3")
    if ref_df is None:
        return pd.DataFrame()

    rows = []
    feat_cols = [c for c in bl_all.columns if c != "PATNO"]
    for visit in ["V04", "V06", "V08", "V10", "V12"]:
        months = VISIT_SCHEDULE[visit]
        pats_at_visit = set(ref_df.loc[ref_df["VISIT"] == visit, "PATNO"].unique())
        survivors = bl_all[bl_all["PATNO"].isin(pats_at_visit)]
        dropouts = bl_all[~bl_all["PATNO"].isin(pats_at_visit)]

        for feat in feat_cols:
            sv = survivors[feat].dropna().values
            dv = dropouts[feat].dropna().values
            if len(sv) < 10 or len(dv) < 10:
                continue
            pooled_std = np.sqrt(((len(sv)-1)*np.var(sv,ddof=1)+(len(dv)-1)*np.var(dv,ddof=1))/(len(sv)+len(dv)-2))
            cohens_d = (np.mean(sv) - np.mean(dv)) / pooled_std if pooled_std > 1e-10 else 0.0
            _, p_val = sp_stats.ttest_ind(sv, dv, equal_var=False)
            rows.append({
                "visit": visit, "months": months, "feature": feat,
                "n_survivors": len(sv), "n_dropouts": len(dv),
                "pct_retained": len(sv)/(len(sv)+len(dv))*100,
                "survivors_mean": float(np.mean(sv)), "dropouts_mean": float(np.mean(dv)),
                "cohens_d": float(cohens_d), "p_value": float(p_val),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        # Benjamini-Hochberg FDR correction across all survivorship tests
        p_vals = result["p_value"].values
        n = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        sorted_p = p_vals[sorted_idx]
        bh = np.zeros(n)
        bh[-1] = sorted_p[-1]
        for i in range(n - 2, -1, -1):
            bh[i] = min(bh[i + 1], sorted_p[i] * n / (i + 1))
        bh = np.clip(bh, 0, 1)
        p_adj = np.zeros(n)
        p_adj[sorted_idx] = bh
        result["p_adjusted"] = p_adj
        result["significant_fdr05"] = result["p_adjusted"] < 0.05

        out_dir = RESULTS_DIR / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_dir / "survivorship_bias.csv", index=False)

        all_pats = bl_all["PATNO"].nunique()
        attrition = []
        for visit in VISIT_ORDER:
            n = all_pats if visit == "BL" else ref_df.loc[ref_df["VISIT"] == visit, "PATNO"].nunique()
            attrition.append({
                "visit": visit, "months": VISIT_SCHEDULE[visit],
                "n_patients": n, "pct_of_baseline": n / all_pats * 100 if all_pats > 0 else 0,
            })
        attrition_df = pd.DataFrame(attrition)
        attrition_df.to_csv(out_dir / "survivorship_summary.csv", index=False)
        log.info("Survivorship analysis: %d rows, attrition saved", len(result))

    return result


if __name__ == "__main__":
    characterize_survivorship()
