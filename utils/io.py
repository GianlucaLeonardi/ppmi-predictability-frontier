"""
Data loading and derived-column utilities.

Loads raw PPMI CSVs and already-processed data, adds derived sub-scores.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from configs.config import (
    RAW_DATA_DIR,
    EXISTING_PROCESSED,
    UPDRS3_TREMOR_ITEMS,
    UPDRS3_BRADY_ITEMS,
    UPDRS3_PIGD_ITEMS,
    MOCA_IMPAIRMENT_CUTOFF,
    VISIT_ORDER,
    VISIT_MONTH_MAP,
)
from utils.logging_utils import get_logger

log = get_logger(__name__)


# -- Raw CSV readers ---------------------------------------------------------

def read_raw(name: str) -> pd.DataFrame:
    """Load a raw PPMI CSV file by table name."""
    path = RAW_DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw table not found: {path}")
    return pd.read_csv(path, low_memory=False)


def read_csv_gz(path: Path) -> pd.DataFrame:
    """Read CSV or CSV.GZ file."""
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


# -- Cohort ------------------------------------------------------------------

def get_cohort_patnos(cohort_labels: List[str]) -> np.ndarray:
    """Get patient IDs for the specified cohort labels."""
    pstatus = read_raw("Participant_Status")
    pstatus.columns = [c.upper() for c in pstatus.columns]
    mask = pstatus["COHORT_DEFINITION"].isin(cohort_labels)
    patnos = pstatus.loc[mask, "PATNO"].unique()
    log.info("Cohort %s: %d patients", cohort_labels, len(patnos))
    return patnos


# -- Processed data loaders --------------------------------------------------

def load_static_modality(modality: str) -> pd.DataFrame:
    """Load a pre-processed static modality from the upstream pipeline."""
    path = EXISTING_PROCESSED / "static" / modality / "data.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Static modality '{modality}' not found at {path}. "
            "Run `python data_preprocessing/01_data_cleaning_and_pivot.py` first to "
            "build the processed feature tables (see README, 'Reproduction')."
        )
    df = read_csv_gz(path)
    df.columns = [c.upper() for c in df.columns]
    return df



# -- Derived sub-scores ------------------------------------------------------

def add_derived_columns(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Add clinically meaningful derived columns to a longitudinal modality."""
    df = df.copy()

    if modality == "updrs3":
        avail = [c for c in UPDRS3_TREMOR_ITEMS if c in df.columns]
        if avail:
            df["TREMOR_SUBSCORE"] = df[avail].sum(axis=1, min_count=len(avail))

        avail = [c for c in UPDRS3_BRADY_ITEMS if c in df.columns]
        if avail:
            df["BRADY_SUBSCORE"] = df[avail].sum(axis=1, min_count=len(avail))

        avail = [c for c in UPDRS3_PIGD_ITEMS if c in df.columns]
        if avail:
            df["PIGD_SUBSCORE"] = df[avail].sum(axis=1, min_count=len(avail))

    elif modality == "moca":
        recall_items = sorted([c for c in df.columns if c.startswith("MCAREC")])
        if recall_items:
            df["DELAYED_RECALL_SUM"] = df[recall_items].sum(axis=1, min_count=len(recall_items))

        if "MCATOT" in df.columns:
            df["MCI_FLAG"] = (df["MCATOT"] < MOCA_IMPAIRMENT_CUTOFF).astype(float)
            df.loc[df["MCATOT"].isna(), "MCI_FLAG"] = np.nan

    elif modality == "vital_signs":
        # -- Temperature unit correction ------------------------------------
        # PPMI records TEMPC in Celsius.  A handful of values are clearly
        # in Fahrenheit (95-99 range → 35-37 °C after conversion).
        # Values 50-90 are ambiguous/corrupt; >100 are data-entry typos.
        if "TEMPC" in df.columns:
            t = df["TEMPC"]
            fahrenheit = t > 90  # 95-99 °F → normal body temp in °C
            df.loc[fahrenheit, "TEMPC"] = (t[fahrenheit] - 32) * 5.0 / 9.0
            corrupt = (t > 42) & (t <= 90) | (t > 100)
            n_fix = fahrenheit.sum() + corrupt.sum()
            df.loc[corrupt, "TEMPC"] = np.nan
            if n_fix > 0:
                log.info("TEMPC: converted %d Fahrenheit, set %d corrupt to NaN",
                         fahrenheit.sum(), corrupt.sum())

        # -- Height unit correction -----------------------------------------
        # PPMI records HTCM in centimetres.  Values 50-100 are likely inches.
        # Values ≤50 are physiologically impossible for adults.
        if "HTCM" in df.columns:
            h = df["HTCM"]
            inches = (h > 50) & (h < 100)
            impossible = h <= 50
            df.loc[inches, "HTCM"] = h[inches] * 2.54
            df.loc[impossible, "HTCM"] = np.nan
            n_fix = inches.sum() + impossible.sum()
            if n_fix > 0:
                log.info("HTCM: converted %d inches→cm, set %d impossible to NaN",
                         inches.sum(), impossible.sum())

        # -- All-zero BP/HR rows (not assessed) -----------------------------
        bp_cols = [c for c in ["SYSSUP", "SYSSTND", "DIASUP", "DIASTND",
                               "HRSUP", "HRSTND"] if c in df.columns]
        if bp_cols:
            all_zero = (df[bp_cols] == 0).all(axis=1)
            df.loc[all_zero, bp_cols] = np.nan

        # Partial-zero standing vitals: if all three standing measures are 0
        # but supine values exist, the standing assessment was not performed.
        standing_cols = [c for c in ["SYSSTND", "DIASTND", "HRSTND"]
                         if c in df.columns]
        if len(standing_cols) == 3:
            standing_zero = (df[standing_cols] == 0).all(axis=1)
            supine_cols = [c for c in ["SYSSUP", "DIASUP", "HRSUP"]
                           if c in df.columns]
            supine_valid = df[supine_cols].notna().any(axis=1) & \
                           (df[supine_cols] != 0).any(axis=1)
            partial_zero = standing_zero & supine_valid
            if partial_zero.any():
                df.loc[partial_zero, standing_cols] = np.nan
                log.info("Vital signs: set %d partial-zero standing rows to NaN",
                         partial_zero.sum())

        # -- Orthostatic drop -----------------------------------------------
        if "SYSSTND" in df.columns and "SYSSUP" in df.columns:
            df["ORTHO_SYS_DROP"] = df["SYSSUP"] - df["SYSSTND"]
        if "DIASTND" in df.columns and "DIASUP" in df.columns:
            df["ORTHO_DIA_DROP"] = df["DIASUP"] - df["DIASTND"]

    return df


# -- LEDD (Levodopa Equivalent Daily Dose) -----------------------------------
# The PPMI file LEDD_Concomitant_Medication_Log.csv stores one row per
# medication per patient per time period, with MM/YYYY start/stop dates
# and per-medication LEDD values.  Some rows carry formula entries
# ("LD x 0.33") for COMT inhibitors whose contribution is a fraction of
# concurrent levodopa dose.
#
# Algorithm (5 steps):
#   1. Build a visit-date lookup (PATNO, EVENT_ID) -> MM/YYYY from
#      the UPDRS-III INFODT column (most complete longitudinal file).
#   2. For each (patient, visit_month): find medications active at that
#      month — STARTDT <= visit_month AND (STOPDT >= visit_month OR NaN).
#   3. Two-pass LEDD summation per patient-visit:
#      (a) Sum all numeric LEDD values -> LD_total.
#      (b) Parse formula entries ("LD x F") and add LD_total * F.
#   4. Output one row per (PATNO, VISIT) with LEDD_TOTAL (mg/day).
#   5. Patients absent from the medication log get LEDD_TOTAL = 0.

def _parse_mmyyyy(s) -> Optional[int]:
    """Convert 'MM/YYYY' to an integer YYYYMM for comparison."""
    if not isinstance(s, str) or len(s) != 7:
        return None
    try:
        parts = s.split("/")
        month = int(parts[0])
        year = int(parts[1])
        if not (1 <= month <= 12):
            return None
        return year * 100 + month
    except (ValueError, IndexError):
        return None


def load_ledd_longitudinal(cohort_patnos: np.ndarray) -> pd.DataFrame:
    """Compute visit-aligned total LEDD per patient (PATNO, VISIT, MONTHS, LEDD_TOTAL)."""
    # Step 1: Build visit-date lookup from UPDRS-III INFODT
    u3_path = RAW_DATA_DIR / "MDS-UPDRS_Part_III.csv"
    if not u3_path.exists():
        log.warning("MDS-UPDRS_Part_III.csv not found; cannot compute LEDD")
        return pd.DataFrame(columns=["PATNO", "VISIT", "MONTHS", "LEDD_TOTAL"])
    u3 = pd.read_csv(u3_path, low_memory=False, usecols=["PATNO", "EVENT_ID", "INFODT"])
    u3.columns = [c.upper() for c in u3.columns]

    # Canonicalize visits
    visit_map = {"SCREENING": "SC", "BASELINE": "BL", "V00": "BL"}
    u3["VISIT"] = u3["EVENT_ID"].replace(visit_map)
    u3 = u3[u3["VISIT"].isin(VISIT_ORDER)]
    u3 = u3[u3["PATNO"].isin(cohort_patnos)]

    # Parse visit dates to YYYYMM
    u3["VISIT_YYYYMM"] = u3["INFODT"].apply(_parse_mmyyyy)
    u3 = u3.dropna(subset=["VISIT_YYYYMM"])
    u3["VISIT_YYYYMM"] = u3["VISIT_YYYYMM"].astype(int)

    # Keep one date per (PATNO, VISIT) — use median if multiple
    visit_dates = u3.groupby(["PATNO", "VISIT"])["VISIT_YYYYMM"].median().astype(int).reset_index()

    # Step 2: Load medication log
    ledd_path = RAW_DATA_DIR / "LEDD_Concomitant_Medication_Log.csv"
    if not ledd_path.exists():
        log.warning("LEDD_Concomitant_Medication_Log.csv not found")
        return pd.DataFrame(columns=["PATNO", "VISIT", "MONTHS", "LEDD_TOTAL"])
    med = pd.read_csv(ledd_path, low_memory=False)
    med.columns = [c.upper() for c in med.columns]
    med = med[med["PATNO"].isin(cohort_patnos)]

    # Parse dates
    med["START_YYYYMM"] = med["STARTDT"].apply(_parse_mmyyyy)
    med["STOP_YYYYMM"] = med["STOPDT"].apply(_parse_mmyyyy)

    # Parse LEDD: numeric values and formula entries
    med["LEDD_NUM"] = pd.to_numeric(med["LEDD"], errors="coerce")
    med["LEDD_FORMULA"] = np.nan
    formula_mask = med["LEDD_NUM"].isna() & med["LEDD"].notna()
    for idx in med.index[formula_mask]:
        val = str(med.at[idx, "LEDD"])
        # Parse "LD x 0.33" -> 0.33
        if "LD" in val.upper() and "X" in val.upper():
            try:
                factor = float(val.upper().replace("LD", "").replace("X", "").strip())
                med.at[idx, "LEDD_FORMULA"] = factor
            except ValueError:
                pass

    med = med.dropna(subset=["START_YYYYMM"])
    med["START_YYYYMM"] = med["START_YYYYMM"].astype(int)

    # Step 3-4: For each (patient, visit), compute total LEDD
    # Vectorised approach: cross-join visit_dates with med on PATNO, filter active
    merged = visit_dates.merge(med[["PATNO", "START_YYYYMM", "STOP_YYYYMM",
                                     "LEDD_NUM", "LEDD_FORMULA"]],
                                on="PATNO", how="inner")
    # Filter active: started before/at visit, not stopped before visit
    active = merged[
        (merged["START_YYYYMM"] <= merged["VISIT_YYYYMM"])
        & (merged["STOP_YYYYMM"].isna() | (merged["STOP_YYYYMM"] >= merged["VISIT_YYYYMM"]))
    ].copy()

    # Pass 1: sum numeric LEDD per (PATNO, VISIT)
    numeric_sum = active.groupby(["PATNO", "VISIT"])["LEDD_NUM"].sum().reset_index()
    numeric_sum = numeric_sum.rename(columns={"LEDD_NUM": "LD_NUMERIC"})

    # Pass 2: for formula entries, multiply factor by concurrent numeric LEDD total
    formula_active = active[active["LEDD_FORMULA"].notna()].copy()
    if not formula_active.empty:
        formula_active = formula_active.merge(numeric_sum, on=["PATNO", "VISIT"], how="left")
        formula_active["FORMULA_CONTRIB"] = formula_active["LEDD_FORMULA"] * formula_active["LD_NUMERIC"].fillna(0)
        formula_sum = formula_active.groupby(["PATNO", "VISIT"])["FORMULA_CONTRIB"].sum().reset_index()
    else:
        formula_sum = pd.DataFrame(columns=["PATNO", "VISIT", "FORMULA_CONTRIB"])

    # Combine
    result = visit_dates[["PATNO", "VISIT"]].drop_duplicates()
    result = result.merge(numeric_sum, on=["PATNO", "VISIT"], how="left")
    result = result.merge(formula_sum, on=["PATNO", "VISIT"], how="left")
    result["LD_NUMERIC"] = result["LD_NUMERIC"].fillna(0.0)
    result["FORMULA_CONTRIB"] = result["FORMULA_CONTRIB"].fillna(0.0)
    result["LEDD_TOTAL"] = result["LD_NUMERIC"] + result["FORMULA_CONTRIB"]
    result["MONTHS"] = result["VISIT"].map(
        {v: VISIT_MONTH_MAP[v] for v in VISIT_ORDER if v in VISIT_MONTH_MAP}
    )
    result = result[["PATNO", "VISIT", "MONTHS", "LEDD_TOTAL"]]

    n_patients = result["PATNO"].nunique()
    n_nonzero = (result["LEDD_TOTAL"] > 0).sum()
    log.info("LEDD: computed for %d patients, %d visit-level records (%d non-zero)",
             n_patients, len(result), n_nonzero)

    return result
