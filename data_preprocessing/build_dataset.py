"""
Build analysis-ready, un-normalized task datasets for the PPMI Predictability-Frontier Benchmark.

Usage:
    python -m data_preprocessing.build_dataset [--force]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    COHORT_PRIMARY,
    FORECAST_HORIZONS,
    MODALITY_FAMILIES,
    MOTOR_WORSEN_THRESHOLD,
    N_OUTER_FOLDS,
    N_SEEDS,
    N_STRATIFICATION_BINS,
    PROCESSED_DATA_DIR,
    REGIMES,
    RESULTS_DIR,
    TARGETS_CLASSIFICATION,
    TARGETS_RANKING,
    TARGETS_REGRESSION,
    VISIT_ORDER,
    VISIT_SCHEDULE,
    WINSOR_QUANTILES,
    Z_CLIP,
)
from utils.io import add_derived_columns, get_cohort_patnos, load_ledd_longitudinal, load_static_modality, read_raw
from utils.logging_utils import add_file_handler, get_logger

log = get_logger("build_dataset")


# =============================================================================
# 1. Load raw longitudinal tables
# =============================================================================

RAW_TABLE_MAP = {
    "updrs3": "MDS-UPDRS_Part_III",
    "updrs1": "MDS-UPDRS_Part_I",
    "moca": "Montreal_Cognitive_Assessment__MoCA",
    "vital_signs": "Vital_Signs",
    "blood_chemistry": "Blood_Chemistry___Hematology",
    "ledd": None,  # LEDD uses a dedicated loader (load_ledd_longitudinal)
}


def load_raw_longitudinal(modality: str) -> pd.DataFrame:
    """Load raw longitudinal data, canonicalize visits, add derived columns."""
    if modality not in RAW_TABLE_MAP:
        raise ValueError(f"Unknown longitudinal modality: {modality}")
    if RAW_TABLE_MAP[modality] is None:
        return pd.DataFrame()

    df = read_raw(RAW_TABLE_MAP[modality])
    df.columns = [c.upper() for c in df.columns]

    if "EVENT_ID" not in df.columns:
        return pd.DataFrame()

    # Canonicalize EVENT_ID -> VISIT
    visit_map = {"SCREENING": "SC", "BASELINE": "BL", "V00": "BL"}
    df["VISIT"] = df["EVENT_ID"].replace(visit_map)
    df = df[df["VISIT"].isin(VISIT_ORDER + ["SC"])]

    # SC -> BL for patients without a BL row
    has_bl = set(df.loc[df["VISIT"] == "BL", "PATNO"].unique())
    sc_mask = (df["VISIT"] == "SC") & (~df["PATNO"].isin(has_bl))
    df.loc[sc_mask, "VISIT"] = "BL"
    df = df[df["VISIT"].isin(VISIT_ORDER)]

    df["MONTHS"] = df["VISIT"].map(VISIT_SCHEDULE)

    # UPDRS-III: keep OFF-state assessments only.
    if modality == "updrs3" and "PDSTATE" in df.columns:
        n_before = len(df)
        df = df[df["PDSTATE"].isin(["OFF"]) | df["PDSTATE"].isna()].copy()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            log.info("UPDRS-III: dropped %d ON-state rows, kept %d OFF/untreated",
                     n_dropped, len(df))

    # Drop metadata columns
    drop_cols = ["REC_ID", "EVENT_ID", "PAG_NAME", "INFODT", "ORIG_ENTRY",
                 "LAST_UPDATE", "NUPSOURC", "PDSTATE", "PDTRTMNT",
                 "HRPOSTMED", "HRDBSON", "HRDBSOFF", "PDMEDYN", "DBSYN",
                 "ONOFFORDER", "OFFEXAM", "OFFNORSN", "DBSOFFYN", "DBSOFFTM",
                 "ONEXAM", "ONNORSN", "HIFUYN", "DBSONYN", "DBSONTM",
                 "PDMEDDT", "PDMEDTM", "EXAMDT", "EXAMTM"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Parse numeric
    for c in df.columns:
        if c not in ("PATNO", "VISIT", "MONTHS"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace PPMI sentinel 101 ("Unable to Rate") with NaN BEFORE derived scores.
    sentinel_cols = [c for c in df.columns
                     if c.startswith(("NP1", "NP3")) or c == "NHY"]
    for c in sentinel_cols:
        n_sentinel = (df[c] == 101).sum()
        if n_sentinel > 0:
            log.info("Replacing %d sentinel-101 values in %s with NaN", n_sentinel, c)
            df.loc[df[c] == 101, c] = np.nan

    # Deduplicate: keep median per PATNO x VISIT
    meta_cols = ["PATNO", "VISIT", "MONTHS"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df.groupby(["PATNO", "VISIT"], as_index=False)[feat_cols + ["MONTHS"]].median()

    # Add derived columns
    df = add_derived_columns(df, modality)

    return df


# =============================================================================
# 2. Normalization (used per-fold in CV, not at preprocessing time)
# =============================================================================

class ColumnNormalizer:
    """Winsorize + impute + z-score for continuous; mode-impute for binary/ordinal.
    Fit on train only, transform any split."""

    def __init__(self, winsor_q=WINSOR_QUANTILES, z_clip=Z_CLIP):
        self.winsor_q = winsor_q
        self.z_clip = z_clip
        self.stats_ = {}

    def fit(self, df: pd.DataFrame, exclude: List[str] = None) -> "ColumnNormalizer":
        exclude = set(exclude or [])
        for col in df.columns:
            if col in exclude:
                continue
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            uniq = vals.unique()
            is_binary = len(uniq) <= 2 and set(uniq).issubset({0, 1, 0.0, 1.0})
            is_ordinal = (len(uniq) <= 12
                         and all(float(v).is_integer() for v in uniq)
                         and not is_binary)

            if is_binary or is_ordinal:
                self.stats_[col] = {
                    "kind": "binary" if is_binary else "ordinal",
                    "mode": float(vals.mode().iloc[0]),
                }
            else:
                lo = vals.quantile(self.winsor_q[0])
                hi = vals.quantile(self.winsor_q[1])
                clipped = vals.clip(lo, hi)
                self.stats_[col] = {
                    "kind": "continuous",
                    "lo": float(lo), "hi": float(hi),
                    "median": float(clipped.median()),
                    "mean": float(clipped.mean()),
                    "std": float(clipped.std()),
                }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, st in self.stats_.items():
            if col not in df.columns:
                continue
            if st["kind"] in ("binary", "ordinal"):
                df[col] = df[col].fillna(st["mode"])
            else:
                df[col] = df[col].clip(st["lo"], st["hi"])
                df[col] = df[col].fillna(st["median"])
                if st["std"] > 1e-10:
                    df[col] = (df[col] - st["mean"]) / st["std"]
                    df[col] = df[col].clip(-self.z_clip, self.z_clip)
                else:
                    df[col] = 0.0
        return df


# =============================================================================
# 3. CV fold creation (used at evaluation time, not preprocessing)
# =============================================================================

def create_cv_folds(
    patnos: np.ndarray,
    y_values: np.ndarray,
    task_type: str,
    n_folds: int = N_OUTER_FOLDS,
    seed: int = 0,
) -> List[Dict[str, np.ndarray]]:
    """Create stratified K-fold patient-level splits, returning train/test index dicts per fold."""
    rng = np.random.default_rng(seed)
    n = len(patnos)

    # Create stratification labels
    if task_type == "classification":
        strata = y_values.astype(int)
    else:
        # Bin continuous targets into quantiles for stratification
        finite_mask = np.isfinite(y_values)
        strata = np.zeros(n, dtype=int)
        if finite_mask.sum() > N_STRATIFICATION_BINS:
            quantiles = np.linspace(0, 100, N_STRATIFICATION_BINS + 1)
            bin_edges = np.percentile(y_values[finite_mask], quantiles)
            # np.digitize can produce bins 0..N_STRATIFICATION_BINS; clip to valid range
            strata[finite_mask] = np.clip(
                np.digitize(y_values[finite_mask], bin_edges[1:-1]),
                0, N_STRATIFICATION_BINS - 1,
            )

    # Stratified split: within each stratum, shuffle and deal into folds
    fold_indices = [[] for _ in range(n_folds)]
    for stratum_val in np.unique(strata):
        stratum_idx = np.where(strata == stratum_val)[0]
        rng.shuffle(stratum_idx)
        for i, idx in enumerate(stratum_idx):
            fold_indices[i % n_folds].append(idx)

    # Build train/test for each fold
    folds = []
    for fold_i in range(n_folds):
        test_idx = np.array(sorted(fold_indices[fold_i]))
        train_idx = np.array(sorted(
            [idx for j in range(n_folds) if j != fold_i for idx in fold_indices[j]]
        ))
        # Verify no overlap
        assert len(set(test_idx) & set(train_idx)) == 0, \
            f"Patient overlap in fold {fold_i}"
        folds.append({"train_idx": train_idx, "test_idx": test_idx})

    return folds


# =============================================================================
# 4. Classification target construction
# =============================================================================

def _compute_motor_worsening(
    longitudinal: Dict[str, pd.DataFrame],
    horizon_visit: str,
) -> pd.DataFrame:
    """Binary: NP3TOT at horizon >= NP3TOT at BL + threshold."""
    updrs3 = longitudinal.get("updrs3")
    if updrs3 is None or updrs3.empty:
        return pd.DataFrame()

    bl = updrs3[updrs3["VISIT"] == "BL"][["PATNO", "NP3TOT"]].dropna()
    bl = bl.rename(columns={"NP3TOT": "NP3TOT_BL"})

    hz = updrs3[updrs3["VISIT"] == horizon_visit][["PATNO", "NP3TOT"]].dropna()
    hz = hz.rename(columns={"NP3TOT": "NP3TOT_HZ"})

    merged = bl.merge(hz, on="PATNO", how="inner")
    merged["MOTOR_WORSEN"] = (
        (merged["NP3TOT_HZ"] - merged["NP3TOT_BL"]) >= MOTOR_WORSEN_THRESHOLD
    ).astype(float)

    return merged[["PATNO", "MOTOR_WORSEN"]]


def _compute_motor_rank(
    longitudinal: Dict[str, pd.DataFrame],
    horizon_visit: str,
) -> pd.DataFrame:
    """Ranking target: patients ranked by NP3TOT change from BL to horizon (higher rank = more worsening)."""
    updrs3 = longitudinal.get("updrs3")
    if updrs3 is None or updrs3.empty:
        return pd.DataFrame()

    bl = updrs3[updrs3["VISIT"] == "BL"][["PATNO", "NP3TOT"]].dropna()
    bl = bl.rename(columns={"NP3TOT": "NP3TOT_BL"})

    hz = updrs3[updrs3["VISIT"] == horizon_visit][["PATNO", "NP3TOT"]].dropna()
    hz = hz.rename(columns={"NP3TOT": "NP3TOT_HZ"})

    merged = bl.merge(hz, on="PATNO", how="inner")
    merged["DELTA_NP3TOT"] = merged["NP3TOT_HZ"] - merged["NP3TOT_BL"]
    # Rank: higher delta = higher rank number (worse progression)
    merged["MOTOR_RANK"] = merged["DELTA_NP3TOT"].rank(method="average")

    return merged[["PATNO", "MOTOR_RANK"]]


# =============================================================================
# 5. Task construction
# =============================================================================

def construct_task(
    longitudinal: Dict[str, pd.DataFrame],
    static: pd.DataFrame,
    target_col: str,
    target_modality: str,
    horizon_visit: str,
    regime_name: str,
    regime_visits: Optional[List[str]],
    cohort_patnos: np.ndarray = None,
    classification_extra: Dict[str, pd.DataFrame] = None,
    motor_rank_cache: Dict[str, pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Build one feature-target matrix for all cohort patients (no split filtering)."""
    if cohort_patnos is None:
        raise ValueError("cohort_patnos is required")
    # 1. Target values at horizon
    if target_col == "MOTOR_WORSEN" and classification_extra:
        target_at_horizon = classification_extra.get(horizon_visit)
        if target_at_horizon is None or target_at_horizon.empty:
            return None
        target_at_horizon = target_at_horizon.rename(columns={"MOTOR_WORSEN": "target"})
    elif target_col == "MOTOR_RANK" and motor_rank_cache:
        target_at_horizon = motor_rank_cache.get(horizon_visit)
        if target_at_horizon is None or target_at_horizon.empty:
            return None
        target_at_horizon = target_at_horizon.rename(columns={"MOTOR_RANK": "target"})
    else:
        target_df = longitudinal.get(target_modality)
        if target_df is None or target_df.empty:
            return None
        target_at_horizon = target_df[target_df["VISIT"] == horizon_visit][
            ["PATNO", target_col]
        ].dropna(subset=[target_col])
        target_at_horizon = target_at_horizon.rename(columns={target_col: "target"})

    if len(target_at_horizon) == 0:
        return None

    # 2. Restrict to cohort patients
    target_at_horizon = target_at_horizon[target_at_horizon["PATNO"].isin(cohort_patnos)]
    if len(target_at_horizon) == 0:
        return None

    # 3. Determine feature visits (strictly before horizon -- no leakage)
    horizon_months = VISIT_SCHEDULE[horizon_visit]
    if regime_name == "rolling":
        history_visits = [v for v in VISIT_ORDER if VISIT_SCHEDULE[v] < horizon_months]
    else:
        history_visits = [v for v in (regime_visits or []) if VISIT_SCHEDULE[v] < horizon_months]

    if not history_visits:
        return None

    # 4. Decide which modalities to include based on regime
    use_cross_modal = regime_name in ("baseline_multimodal", "baseline_plus_12m", "rolling")
    use_static = regime_name in ("baseline_multimodal", "baseline_plus_12m", "rolling")

    # 5. Build feature matrix
    feat_frames = []

    # 5a. Longitudinal features: flatten into wide format
    for mod_name, mod_df in longitudinal.items():
        if mod_df.empty:
            continue
        # For baseline_only, restrict to same modality
        if not use_cross_modal and mod_name != target_modality:
            continue
        for visit in history_visits:
            visit_data = mod_df[mod_df["VISIT"] == visit].copy()
            if visit_data.empty:
                continue
            drop = ["VISIT", "MONTHS"]
            feat_data = visit_data.drop(columns=[c for c in drop if c in visit_data.columns])
            rename = {c: f"{mod_name}__{visit}__{c}" for c in feat_data.columns if c != "PATNO"}
            feat_data = feat_data.rename(columns=rename)
            feat_frames.append(feat_data)

    # 5b. Static features (only for multimodal+ regimes)
    if use_static and static is not None and not static.empty:
        feat_frames.append(static)

    if not feat_frames:
        return None

    # Merge all feature blocks
    features = feat_frames[0]
    for ff in feat_frames[1:]:
        features = features.merge(ff, on="PATNO", how="outer")

    # 6. Join features with target
    task_df = target_at_horizon.merge(features, on="PATNO", how="inner")

    return task_df if len(task_df) > 0 else None


# =============================================================================
# 6. Sanity checks
# =============================================================================

def sanity_check_no_leakage(feat_names: List[str], horizon: str) -> None:
    """Assert no feature comes from the horizon visit or later."""
    horizon_months = VISIT_SCHEDULE[horizon]
    for fname in feat_names:
        parts = fname.split("__")
        if len(parts) >= 2:
            visit_part = parts[1]
            if visit_part in VISIT_SCHEDULE:
                assert VISIT_SCHEDULE[visit_part] < horizon_months, (
                    f"LEAKAGE: feature '{fname}' from {visit_part} "
                    f"({VISIT_SCHEDULE[visit_part]}m) but horizon is "
                    f"{horizon} ({horizon_months}m)"
                )


def sanity_check_no_duplicates(df: pd.DataFrame) -> None:
    """Assert no duplicate PATNO rows."""
    dupes = df["PATNO"].duplicated().sum()
    assert dupes == 0, f"Found {dupes} duplicate PATNO rows"


# =============================================================================
# 7. Availability characterization
# =============================================================================

def characterize_availability(
    longitudinal: Dict[str, pd.DataFrame],
    targets: list,
    cohort_patnos: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for tspec in targets:
        mod_df = longitudinal.get(tspec.source_table)
        if mod_df is None or mod_df.empty:
            continue
        mod_df = mod_df[mod_df["PATNO"].isin(cohort_patnos)]
        for visit in VISIT_ORDER:
            visit_data = mod_df[mod_df["VISIT"] == visit]
            n_avail = visit_data[tspec.column].notna().sum() if tspec.column in visit_data.columns else 0
            rows.append({
                "target": tspec.name,
                "target_display": tspec.display,
                "domain": tspec.domain,
                "visit": visit,
                "months": VISIT_SCHEDULE[visit],
                "n_available": n_avail,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 8. Main pipeline
# =============================================================================

def run_pipeline(force: bool = False):
    out_dir = PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    add_file_handler(log, out_dir / "preprocessing.log")

    # -- Step 1: Cohort filtering --
    log.info("Step 1: Identifying cohort patients")
    pd_patnos = get_cohort_patnos(COHORT_PRIMARY)
    log.info("  PD patients: %d", len(pd_patnos))

    # -- Step 2: Load longitudinal data --
    log.info("Step 2: Loading longitudinal modalities from raw CSVs")
    long_modalities = ["updrs3", "updrs1", "moca", "vital_signs"]
    longitudinal = {}
    for mod in long_modalities:
        log.info("  Loading %s ...", mod)
        df = load_raw_longitudinal(mod)
        df = df[df["PATNO"].isin(pd_patnos)]
        log.info("    Rows: %d, Patients: %d, Visits: %s",
                 len(df), df["PATNO"].nunique(), sorted(df["VISIT"].unique()))

        # Validate: no duplicate PATNO x VISIT
        dupes = df.groupby(["PATNO", "VISIT"]).size()
        n_dupe = (dupes > 1).sum()
        if n_dupe > 0:
            log.warning("    %d duplicate PATNO x VISIT groups in %s", n_dupe, mod)
        longitudinal[mod] = df

    # Load LEDD separately for sensitivity analysis
    log.info("  Loading LEDD (medication) for sensitivity analysis ...")
    ledd_df = load_ledd_longitudinal(pd_patnos)
    if not ledd_df.empty:
        log.info("    LEDD: %d rows, %d patients, Visits: %s",
                 len(ledd_df), ledd_df["PATNO"].nunique(),
                 sorted(ledd_df["VISIT"].unique()))
    else:
        log.warning("    LEDD: no data loaded")

    # -- Step 3: Load static data --
    log.info("Step 3: Loading static modalities")
    static_frames = []
    for mf in MODALITY_FAMILIES:
        if mf.kind != "static":
            continue
        df = load_static_modality(mf.source_key)
        if df.empty:
            log.warning("  %s: not found, skipping", mf.name)
            continue
        df = df[df["PATNO"].isin(pd_patnos)]
        rename = {c: f"{mf.name}__{c}" for c in df.columns if c != "PATNO"}
        df = df.rename(columns=rename)
        static_frames.append(df)
        log.info("  %s: %d patients, %d features", mf.name, len(df), len(df.columns) - 1)

    static_all = None
    if static_frames:
        static_all = static_frames[0]
        for sf in static_frames[1:]:
            static_all = static_all.merge(sf, on="PATNO", how="outer")
        static_all = static_all[static_all["PATNO"].isin(pd_patnos)]
        log.info("  Merged static: %d patients, %d features",
                 len(static_all), len(static_all.columns) - 1)

    # -- Step 4: Characterize availability --
    log.info("Step 4: Characterizing target availability")
    availability = characterize_availability(longitudinal, TARGETS_REGRESSION, pd_patnos)
    availability.to_csv(out_dir / "target_availability.csv", index=False)

    # -- Step 5: Pre-compute classification targets --
    log.info("Step 5: Computing classification and ranking targets")
    motor_worsen_cache = {}
    motor_rank_cache = {}

    for horizon in FORECAST_HORIZONS:
        # Motor worsening
        mw = _compute_motor_worsening(longitudinal, horizon)
        if not mw.empty:
            motor_worsen_cache[horizon] = mw
            prev = mw["MOTOR_WORSEN"].mean()
            log.info("  Motor worsening at %s: n=%d, prevalence=%.1f%%",
                     horizon, len(mw), prev * 100)

        # Motor change rank
        mr = _compute_motor_rank(longitudinal, horizon)
        if not mr.empty:
            motor_rank_cache[horizon] = mr
            log.info("  Motor rank at %s: n=%d", horizon, len(mr))

    # -- Step 6: Build task datasets --
    log.info("Step 6: Building task datasets (full cohort, no split)")
    tasks_dir = out_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    task_manifest = []

    all_targets = TARGETS_REGRESSION + TARGETS_CLASSIFICATION + TARGETS_RANKING

    for tspec in all_targets:
        for regime in REGIMES:
            for horizon in FORECAST_HORIZONS:
                # Skip if horizon is not ahead of the regime's last history visit
                horizon_months = VISIT_SCHEDULE[horizon]
                if regime.history_visits is not None:
                    last_hist = max(VISIT_SCHEDULE[v] for v in regime.history_visits)
                    if horizon_months <= last_hist:
                        continue

                task_key = f"{tspec.name}__{regime.name}__{horizon}"

                # Build for all patients (no split)
                task_df = construct_task(
                    longitudinal=longitudinal,
                    static=static_all,
                    target_col=tspec.column,
                    target_modality=tspec.source_table,
                    horizon_visit=horizon,
                    regime_name=regime.name,
                    regime_visits=regime.history_visits,
                    cohort_patnos=pd_patnos,
                    classification_extra=motor_worsen_cache if tspec.column == "MOTOR_WORSEN" else None,
                    motor_rank_cache=motor_rank_cache if tspec.column == "MOTOR_RANK" else None,
                )

                if task_df is None or len(task_df) == 0:
                    log.debug("  SKIP %s: no data", task_key)
                    continue

                n_total = len(task_df)
                if n_total < tspec.min_patients * 2:
                    # Need at least 2x min_patients for meaningful CV
                    log.debug("  SKIP %s: only %d total patients (need %d for CV)",
                             task_key, n_total, tspec.min_patients * 2)
                    continue

                feat_cols = [c for c in task_df.columns if c not in ("PATNO", "target")]

                # Leakage check
                sanity_check_no_leakage(feat_cols, horizon)

                # Save raw target columns for LOCF/LME
                # Identify raw target-variable columns at each history visit
                if regime.name == "rolling":
                    history_visits = [v for v in VISIT_ORDER
                                      if VISIT_SCHEDULE[v] < horizon_months]
                else:
                    history_visits = [v for v in (regime.history_visits or [])
                                      if VISIT_SCHEDULE[v] < horizon_months]
                raw_target_cols = {}
                for v in history_visits:
                    candidate = f"{tspec.source_table}__{v}__{tspec.column}"
                    if candidate in feat_cols:
                        raw_target_cols[v] = candidate

                # Add raw target columns with __raw__ prefix
                for v, col in raw_target_cols.items():
                    if col in task_df.columns:
                        task_df[f"__raw__{v}__{tspec.column}"] = task_df[col].values.copy()

                # Final sanity: no duplicate PATNOs
                sanity_check_no_duplicates(task_df)

                # Save the full dataset (not split, not normalized)
                task_out = tasks_dir / task_key
                task_out.mkdir(exist_ok=True)
                task_df.to_csv(task_out / "full.csv.gz", index=False, compression="gzip")

                # Prevalence for classification tasks
                prevalence = None
                if tspec.task_type == "classification":
                    prevalence = float(task_df["target"].mean())

                task_manifest.append({
                    "task_key": task_key,
                    "target": tspec.name,
                    "target_display": tspec.display,
                    "target_domain": tspec.domain,
                    "target_column": tspec.column,
                    "task_type": tspec.task_type,
                    "primary_metric": tspec.primary_metric,
                    "horizon": horizon,
                    "horizon_months": VISIT_SCHEDULE[horizon],
                    "regime": regime.name,
                    "regime_display": regime.display,
                    "n_total": n_total,
                    "n_features": len(feat_cols),
                    "prevalence": json.dumps(prevalence) if isinstance(prevalence, dict) else prevalence,
                })

                log.info("  BUILT %s: n=%d, feat=%d", task_key, n_total, len(feat_cols))

    # -- Step 7: Save manifest --
    manifest_df = pd.DataFrame(task_manifest)
    manifest_df.to_csv(out_dir / "task_manifest.csv", index=False)
    log.info("Saved task manifest: %d tasks", len(manifest_df))

    # -- Step 8: Cohort summary --
    cohort_summary = {
        "n_pd_patients": int(len(pd_patnos)),
        "cv_design": f"{N_OUTER_FOLDS}-fold CV x {N_SEEDS} seeds",
        "n_tasks_built": len(task_manifest),
        "n_regression_tasks": sum(1 for t in task_manifest if t["task_type"] == "regression"),
        "n_classification_tasks": sum(1 for t in task_manifest if t["task_type"] == "classification"),
        "n_ranking_tasks": sum(1 for t in task_manifest if t["task_type"] == "ranking"),
        "targets_regression": [t.name for t in TARGETS_REGRESSION],
        "targets_classification": [t.name for t in TARGETS_CLASSIFICATION],
        "targets_ranking": [t.name for t in TARGETS_RANKING],
        "regimes": [r.name for r in REGIMES],
        "horizons": FORECAST_HORIZONS,
        "visit_schedule": VISIT_SCHEDULE,
    }
    with open(out_dir / "cohort_summary.json", "w") as f:
        json.dump(cohort_summary, f, indent=2)

    log.info("Pipeline complete. %d tasks built. Outputs in %s", len(manifest_df), out_dir)
    return manifest_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build benchmark datasets")
    parser.add_argument("--force", action="store_true", help="Overwrite existing data")
    args = parser.parse_args()
    run_pipeline(force=args.force)
