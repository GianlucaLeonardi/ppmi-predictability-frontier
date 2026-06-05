"""
Exploratory distributional analysis for supplementary material (distributions,
trajectories, group differences, UMAP, target correlations).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    COHORT_PRIMARY,
    EXPLORATORY_DIR,
    VISIT_ORDER,
    VISIT_SCHEDULE,
)
from utils.logging_utils import get_logger

log = get_logger("exploratory")


def _ensure_dirs():
    EXPLORATORY_DIR.mkdir(parents=True, exist_ok=True)


def univariate_distributions(
    longitudinal: Dict[str, pd.DataFrame],
    targets: List[Dict],
    cohort_patnos: np.ndarray,
) -> pd.DataFrame:
    """Compute univariate summary stats for each target at each visit."""
    _ensure_dirs()
    rows = []
    for tspec in targets:
        mod_df = longitudinal.get(tspec["source_table"])
        if mod_df is None or mod_df.empty:
            continue
        mod_df = mod_df[mod_df["PATNO"].isin(cohort_patnos)]

        for visit in VISIT_ORDER:
            vdata = mod_df[mod_df["VISIT"] == visit]
            col = tspec["column"]
            if col not in vdata.columns:
                continue
            vals = vdata[col].dropna().values
            if len(vals) < 10:
                continue

            rows.append({
                "target": tspec["name"],
                "target_display": tspec["display"],
                "visit": visit,
                "months": VISIT_SCHEDULE[visit],
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
                "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                "skew": float(pd.Series(vals).skew()),
                "kurtosis": float(pd.Series(vals).kurtosis()),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(EXPLORATORY_DIR / "univariate_distributions.csv", index=False)
        log.info("Saved univariate distributions: %d rows", len(result))
    return result


def horizon_trajectories(
    longitudinal: Dict[str, pd.DataFrame],
    targets: List[Dict],
    cohort_patnos: np.ndarray,
) -> pd.DataFrame:
    """Compute per-patient trajectories (PATNO x visits) for key targets."""
    _ensure_dirs()
    all_trajectories = []

    for tspec in targets:
        mod_df = longitudinal.get(tspec["source_table"])
        if mod_df is None or mod_df.empty:
            continue
        mod_df = mod_df[mod_df["PATNO"].isin(cohort_patnos)]
        col = tspec["column"]
        if col not in mod_df.columns:
            continue

        pivot = mod_df.pivot_table(index="PATNO", columns="VISIT", values=col)
        pivot.columns = [f"{tspec['name']}__{v}" for v in pivot.columns]
        all_trajectories.append(pivot)

    if all_trajectories:
        result = pd.concat(all_trajectories, axis=1)
        result.to_csv(EXPLORATORY_DIR / "horizon_trajectories.csv")
        log.info("Saved horizon trajectories: %d patients, %d columns",
                 len(result), len(result.columns))
        return result
    return pd.DataFrame()


def target_correlation_matrix(
    longitudinal: Dict[str, pd.DataFrame],
    targets: List[Dict],
    cohort_patnos: np.ndarray,
    visit: str = "BL",
) -> pd.DataFrame:
    """Compute Spearman correlation matrix among targets at a given visit."""
    _ensure_dirs()
    target_vals = {}

    for tspec in targets:
        mod_df = longitudinal.get(tspec["source_table"])
        if mod_df is None or mod_df.empty:
            continue
        vdata = mod_df[(mod_df["VISIT"] == visit) & (mod_df["PATNO"].isin(cohort_patnos))]
        col = tspec["column"]
        if col in vdata.columns:
            target_vals[tspec["display"]] = vdata.set_index("PATNO")[col]

    if len(target_vals) < 2:
        return pd.DataFrame()

    combined = pd.DataFrame(target_vals)
    corr = combined.corr(method="spearman")
    corr.to_csv(EXPLORATORY_DIR / f"target_correlation_{visit}.csv")
    log.info("Saved target correlation matrix at %s: %d x %d", visit, *corr.shape)
    return corr


def umap_feature_space(
    task_key: str = "updrs3_total__baseline_multimodal__V08",
) -> Optional[pd.DataFrame]:
    """Run UMAP on the feature space for a representative task."""
    _ensure_dirs()
    try:
        import umap
    except ImportError:
        log.warning("umap-learn not installed; skipping UMAP analysis")
        return None

    from evaluation.frontier import load_task, _pop_raw_cols

    full_df = load_task(task_key)
    if full_df is None or len(full_df) < 50:
        log.warning("Insufficient data for UMAP: %s", task_key)
        return None

    _pop_raw_cols(full_df)
    feat_cols = [c for c in full_df.columns if c not in ("PATNO", "target")]
    X = full_df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = full_df["target"].values

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X)

    result = pd.DataFrame({
        "PATNO": full_df["PATNO"].values,
        "UMAP1": embedding[:, 0],
        "UMAP2": embedding[:, 1],
        "target": y,
    })

    safe_name = task_key.replace("__", "_")
    result.to_csv(EXPLORATORY_DIR / f"umap_{safe_name}.csv", index=False)
    log.info("Saved UMAP embedding for %s: %d points", task_key, len(result))
    return result


def run_exploratory_analysis():
    """Run all exploratory analyses."""
    _ensure_dirs()
    log.info("Running exploratory analyses...")

    from utils.io import get_cohort_patnos
    from data_preprocessing.build_dataset import load_raw_longitudinal
    from configs.config import TARGETS_REGRESSION

    cohort_patnos = get_cohort_patnos(COHORT_PRIMARY)
    longitudinal = {}
    for mod in ["updrs3", "updrs1", "moca", "vital_signs"]:
        df = load_raw_longitudinal(mod)
        df = df[df["PATNO"].isin(cohort_patnos)]
        longitudinal[mod] = df

    # Convert targets to dict format for this module
    reg_targets = [{"name": t.name, "display": t.display, "source_table": t.source_table,
                     "column": t.column} for t in TARGETS_REGRESSION]

    # 1. Univariate distributions
    log.info("[1/4] Univariate distributions...")
    univariate_distributions(longitudinal, reg_targets, cohort_patnos)

    # 2. Horizon trajectories
    log.info("[2/4] Horizon trajectories...")
    horizon_trajectories(longitudinal, reg_targets, cohort_patnos)

    # 3. Target correlation matrix
    log.info("[3/4] Target correlation matrix...")
    target_correlation_matrix(longitudinal, reg_targets, cohort_patnos, visit="BL")

    # 4. UMAP (optional, depends on umap-learn)
    log.info("[4/4] UMAP feature space...")
    for task in ["updrs3_total__baseline_multimodal__V08", "moca_total__baseline_multimodal__V08"]:
        umap_feature_space(task)

    log.info("Exploratory analyses complete. Outputs in %s", EXPLORATORY_DIR)


if __name__ == "__main__":
    run_exploratory_analysis()
