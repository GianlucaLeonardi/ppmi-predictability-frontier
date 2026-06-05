#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPMI longitudinal_coherence — 01_data_cleaning_and_pivot.py

What this script does:
1) Writes metadata dumps (Code List / Data Dictionary) if present.

2) Loads STATIC tables:
   - demographics, participant_status, genetic_consensus, genetic_prs
   - encodes demographics + participant_status semantically
   - genetic_consensus -> numeric indicators + APOE allele counts
   - genetic_prs -> PATNO + numeric columns only

3) Loads LONG tables (true longitudinal):
   - age_at_visit, moca, updrs1, updrs3, neurological_exam, vital_signs
   - filters to allowed events
   - drops SC if BL exists (true longitudinal only, for consistency)

4) Loads baseline-only-from-long sources (treated as STATIC):
   - blood_chemistry
   - restrict to BL/SC only
   - apply the same BL/SC canonicalization rule as the other baseline modalities:
       * if BL exists, drop SC
       * if only SC exists, rename SC -> BL

4b) Extracts CSF biomarkers from Current_Biospecimen_Analysis_Results.csv:
   - ABeta42, ABeta40, pTau181, tTau, CSF Alpha-synuclein
   - Filtered to CSF type, BL clinical event
   - Pivoted from long to wide, duplicates aggregated by median
   - Saved as static modality "csf_biomarkers"

5) Filtering + pivot:
       * drop rows where UNITS indicates "Stdev"
       * build feature keys as TYPE__TESTNAME__(UNITS) to avoid mixing specimen types
       * STRICT numeric parsing (won't extract digits from strings)
       * if duplicate rows exist for same PATNO×EVENT×feature: take MEDIAN
   - blood_chemistry: keep curated LTSTCODE list
       * STRICT numeric parsing
       * if duplicates exist for same PATNO×EVENT×LTSTCODE: take MEDIAN
   - pivot output for each modality: one row per PATNO×EVENT_ID, wide columns

6) Baseline-static extraction (for blood_chemistry):
   - output is one row per PATNO
   - BL preferred, SC fallback
   - NO duplicated __BL / __SC feature blocks

7) Biomarker extraction (CSF + Plasma):
   - CSF: ABeta42, ABeta40, pTau181, tTau, CSF Alpha-synuclein
   - Plasma: NFL, GFAP, Ptau217p
   - Below-LOD plasma values handled via LOD/sqrt(2) substitution

Outputs:
  processed_data/static/<modality>/data.csv.gz
  processed_data/longitudinal/<modality>/<visit>/data.csv.gz    (true longitudinal only)
  processed_data/metadata/processed_manifest.csv.gz
  processed_data/metadata/run_report.json + step reports
"""

from __future__ import annotations
import sys
from pathlib import Path
import re
import json
import argparse
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

# Resolve data locations from the shared project config so this script honours the
# same RAW_DATA_DIR as the rest of the pipeline (set it in configs/config.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import RAW_DATA_DIR, EXISTING_PROCESSED


# =============================================================================
# Paths
# =============================================================================
DATA_DIR = RAW_DATA_DIR

CODE_LIST = DATA_DIR / "Code_List_-__Annotated.csv"
DATA_DICT = DATA_DIR / "Data_Dictionary_-__Annotated.csv"

STATIC_FILES: Dict[str, Path] = {
    "demographics": DATA_DIR / "Demographics.csv",
    "genetic_consensus": DATA_DIR / "iu_genetic_consensus_20251025.csv",
    "participant_status": DATA_DIR / "Participant_Status.csv",
    "genetic_prs": DATA_DIR / "PPMI_Project_9001_20250624.csv",
}

LONG_FILES: Dict[str, Path] = {
    "age_at_visit": DATA_DIR / "Age_at_visit.csv",
    "moca": DATA_DIR / "Montreal_Cognitive_Assessment__MoCA.csv",
    "updrs1": DATA_DIR / "MDS-UPDRS_Part_I.csv",
    "updrs3": DATA_DIR / "MDS-UPDRS_Part_III.csv",
    "neurological_exam": DATA_DIR / "Neurological_Exam.csv",
    "vital_signs": DATA_DIR / "Vital_Signs.csv",
}

# Baseline-only-from-long
BASELINE_STATIC_FROM_LONG_FILES: Dict[str, Path] = {
    "blood_chemistry": DATA_DIR / "Blood_Chemistry___Hematology.csv",
}

# CSF biomarkers (long format: TESTNAME × TESTVALUE, extracted separately)
CSF_BIOSPECIMEN_FILE = DATA_DIR / "Current_Biospecimen_Analysis_Results.csv"

# Key CSF biomarkers to extract at baseline (well-established PD markers)
CSF_BIOMARKER_TESTS = [
    "ABeta42",              # Amyloid-beta 1-42 — cognitive decline trajectory
    "ABeta40",              # Amyloid-beta 1-40 — normalization denominator
    "pTau181",              # Phosphorylated tau 181 — tau pathology
    "tTau",                 # Total tau — neurodegeneration marker
    "CSF Alpha-synuclein",  # alpha-synuclein — core PD protein
]

# Key plasma biomarkers to extract at baseline (complement CSF; higher coverage)
PLASMA_BIOMARKER_TESTS = [
    "NFL",                  # Neurofilament light — axonal degeneration
    "GFAP",                 # Glial fibrillary acidic protein — astrocyte activation
    "Ptau217p",             # Phosphorylated tau 217 — tau pathology in blood
]

OUT_ROOT = EXISTING_PROCESSED
OUT_META = OUT_ROOT / "metadata"
OUT_STATIC = OUT_ROOT / "static"
OUT_LONG = OUT_ROOT / "longitudinal"

NA_VALUES = ["", "NA", "NaN", "nan", "N/A", "NULL", "null", ".", " "]


# =============================================================================
# Column candidates / allowed visits
# =============================================================================
PAT_CANDS = ["PATNO", "SUBJECT", "SUBJECT_ID", "ID"]
EVENT_CANDS = [
    "EVENT_ID", "EVENTID", "EVENT", "VISIT", "VISIT_ID", "VISITID",
    "VISCODE", "VISITCODE", "CLINICAL_EVENT"
]

ALLOWED_EVENT_RE = re.compile(r"^(SC|BL|V02|V04|V05|V06|V07|V08|V09|V10|V12)$")
BASELINE_ONLY_EVENT_RE = re.compile(r"^(SC|BL)$")

DEFAULT_VISITS_TO_EXPORT = ["BL", "SC", "V02", "V04", "V06", "V08", "V10", "V12"]
DROP_EVENT_COL_IN_PER_VISIT_EXPORT = True


# =============================================================================
# KEEP LISTS
# =============================================================================
PD_CORE_PLUS_BASIC_PANEL_CODES = sorted(set([
    "RCT8","RCT11","RCT392","RCT6","RCT13","RCT1",
    "HMT7","HMT8","HMT9","HMT10","HMT13","HMT40","HMT2","HMT3","HMT4",
    "RCT4","RCT5","RCT1407","RCT12","RCT15","RCT16","RCT18","RCT17","RCT183",
]))


# =============================================================================
# Column suggestions (SOFT)
# =============================================================================
demo_cols = ["PATNO", "SEX", "HISPLAT", "RAASIAN", "RABLACK", "RAHAWOPI", "RAINDALS", "RANOS", "RAWHITE", "RAUNKNOWN"]
gen_cons_cols = ["PATNO", "APOE", "PATHVAR_COUNT", "VAR_GENE", "LRRK2", "GBA", "VPS35", "SNCA", "PRKN", "PARK7", "PINK1"]
part_stat_cols = ["PATNO", "COHORT", "COHORT_DEFINITION", "ENROLL_STATUS", "ENROLL_AGE"]

age_cols = ["PATNO", "EVENT_ID", "AGE_AT_VISIT"]
moca_cols = ["PATNO", "EVENT_ID", "MCAALTTM", "MCACUBE", "MCACLCKC", "MCACLCKN", "MCACLCKH", "MCALION", "MCARHINO", "MCACAMEL", "MCAFDS", "MCABDS", "MCAVIGIL", "MCASER7", "MCASNTNC", "MCAVFNUM", "MCAVF", "MCAABSTR", "MCAREC1", "MCAREC2", "MCAREC3", "MCAREC4", "MCAREC5", "MCADATE", "MCAMONTH", "MCAYR", "MCADAY", "MCAPLACE", "MCACITY", "MCATOT"]
updrs1_cols = ["PATNO", "EVENT_ID", "NUPSOURC", "NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS", "NP1RTOT"]
updrs3_cols = ["PATNO", "EVENT_ID", "PDSTATE", "NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU", "NP3RIGLU", "NP3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL", "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL", "NP3LGAGR", "NP3LGAGL", "NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR", "NP3BRADY", "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON", "NP3TOT"] # BL/SC baseline target: keep UPDRS3 total only
neuro_cols = ["PATNO", "EVENT_ID", "MTRRSP", "CORDRSP", "SENRSP", "RFLXRSP", "PLRRRSP", "PLRLRSP", "CNRSP"]
vital_cols = ["PATNO", "EVENT_ID", "WGTKG", "HTCM", "TEMPC", "BPARM", "SYSSUP", "DIASUP", "HRSUP", "SYSSTND", "DIASTND", "HRSTND"]

# NOTE: DO NOT hard-subset blood chemistry; it may be wide.


# =============================================================================
# Helpers
# =============================================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_df(df: pd.DataFrame, out_fp: Path) -> None:
    ensure_dir(out_fp.parent)
    df.to_csv(out_fp, index=False, compression="gzip")

def write_json(obj: Any, out_fp: Path) -> None:
    ensure_dir(out_fp.parent)
    with out_fp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def write_text_lines(lines: List[str], out_fp: Path) -> None:
    ensure_dir(out_fp.parent)
    with out_fp.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(str(ln) + "\n")

def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False, na_values=NA_VALUES, keep_default_na=True)
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df

def safe_keep_cols(
    df: pd.DataFrame,
    cols: List[str],
    name: str,
    *,
    strict: bool = False,
    require_event: bool = False,
) -> pd.DataFrame:
    """Keep a curated subset of columns, *after* standardizing PATNO/EVENT_ID.

    Why: several PPMI tables use SUBJECT / PATNO, VISIT / EVENT_ID, etc.
    If we subset before standardization, we can silently miss PATNO and keep-all,
    which later explodes dimensionality (e.g. Neurological Exam admin fields).

    strict=True:
      - require PATNO to exist
      - optionally require EVENT_ID (for longitudinal/baseline-long tables)
      - raise instead of keep-all if overlap is too small
    """
    d = df.copy()
    d.columns = [str(c).strip().upper() for c in d.columns]

    # Standardize ID columns first (SUBJECT->PATNO, VISIT->EVENT_ID, etc.)
    try:
        d = standardize_pat_event_cols(d)
    except Exception as e:
        # Keep going, but warn — downstream may break if IDs are missing.
        print(f"[WARN] {name}: failed to standardize PATNO/EVENT_ID ({e}).")

    cols_u = [str(c).strip().upper() for c in cols]
    keep = [c for c in cols_u if c in d.columns]

    if "PATNO" not in d.columns:
        msg = f"{name}: missing PATNO column after standardization."
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}")
        return d

    if require_event and "EVENT_ID" not in d.columns:
        msg = f"{name}: missing EVENT_ID column after standardization."
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}")

    # To safely subset, we need PATNO + at least one feature (or EVENT_ID for longitudinal).
    if "PATNO" in keep and len(keep) >= 2:
        return d.loc[:, keep].copy()

    msg = f"{name}: requested keep list has too little overlap (keep={keep}). Keeping ALL columns (dangerous)."

    if strict:
        raise ValueError("[ERROR] " + msg)
    print("[WARN] " + msg)
    return d


def safe_keep_cols_soft(df: pd.DataFrame, cols: List[str], name: str) -> pd.DataFrame:
    return safe_keep_cols(df, cols, name, strict=False, require_event=False)


def safe_keep_cols_strict_long(df: pd.DataFrame, cols: List[str], name: str) -> pd.DataFrame:
    # For longitudinal-like tables we MUST have EVENT_ID to pivot/export per visit.
    return safe_keep_cols(df, cols, name, strict=True, require_event=True)

def guess_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        cu = c.upper()
        if cu in cols:
            return cu
    return None

def normalize_event_series(s: pd.Series) -> pd.Series:
    ev = s.astype("string").str.strip().str.upper()
    ev = ev.replace({
        "SCREENING": "SC",
        "BASELINE": "BL",
        "V00": "BL",
        "V0": "BL",
        "VISIT 0": "BL",
        "VISIT0": "BL",
    })

    def _canon(x):
        if pd.isna(x):
            return pd.NA
        x = str(x).strip().upper()
        if x in {"SC", "BL"}:
            return x
        m = re.match(r"^V(\d+)([A-Z]*)$", x)
        if m:
            num = m.group(1).zfill(2)
            suf = m.group(2) or ""
            return f"V{num}{suf}"
        return x

    return ev.map(_canon).astype("string")

def _norm_loose(s: str) -> str:
    s = str(s).upper().strip()
    return re.sub(r"[^A-Z0-9]+", "", s)

def _parse_numeric(x) -> float:
    """Loose numeric parser (ok for typical numeric fields, NOT ok for genotypes/qualitative assays)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace(",", ".")
    s = re.sub(r"^[<>]=?\s*", "", s)
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*[-–]\s*([+-]?\d+(?:\.\d+)?)\s*$", s)
    if m:
        a = float(m.group(1)); b = float(m.group(2))
        return 0.5 * (a + b)
    m = re.search(r"([+-]?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan

def _parse_numeric_strict(x) -> float:
    """Strict numeric parser: only accept truly numeric strings (and ranges / < >)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace(",", ".")
    s = re.sub(r"^[<>]=?\s*", "", s)

    # range like "1.2-3.4"
    if re.match(r"^[+-]?\d+(?:\.\d+)?\s*[-–]\s*[+-]?\d+(?:\.\d+)?$", s):
        a, b = re.split(r"\s*[-–]\s*", s)
        try:
            return 0.5 * (float(a) + float(b))
        except Exception:
            return np.nan

    # pure numeric only
    if re.match(r"^[+-]?\d+(?:\.\d+)?$", s):
        return float(s)

    return np.nan

def standardize_pat_event_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip().upper() for c in d.columns]
    pat = guess_col(d, PAT_CANDS)
    ev = guess_col(d, EVENT_CANDS)

    if pat is not None and pat != "PATNO":
        d = d.rename(columns={pat: "PATNO"})
    if "PATNO" in d.columns:
        d["PATNO"] = d["PATNO"].astype("string").str.strip()

    if ev is not None and ev != "EVENT_ID":
        d = d.rename(columns={ev: "EVENT_ID"})

    cols = d.columns.tolist()
    front = [c for c in ["PATNO", "EVENT_ID"] if c in cols]
    rest = [c for c in cols if c not in front]
    return d[front + rest]


def filter_long_by_events(df: pd.DataFrame, allowed_re: re.Pattern) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    d = df.copy()
    d.columns = [str(c).strip().upper() for c in d.columns]
    ev_col = guess_col(d, EVENT_CANDS)
    if ev_col is None:
        return d, {"status": "skipped_no_event_col", "rows_before": int(len(d)), "rows_after": int(len(d))}
    ev_norm = normalize_event_series(d[ev_col])
    m = ev_norm.str.match(allowed_re, na=False)
    out = d.loc[m].copy()
    return out, {
        "status": "ok",
        "event_col": ev_col,
        "rows_before": int(len(d)),
        "rows_after": int(m.sum()),
        "fraction_kept": float(m.mean()) if len(d) else 0.0,
        "unique_events_kept": sorted(ev_norm[m].dropna().unique().tolist()),
    }

def drop_sc_if_bl_exists(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply ONLY to true longitudinal tables (not baseline-only static sources)."""
    d = df.copy()
    d.columns = [str(c).strip().upper() for c in d.columns]
    pat_col = guess_col(d, PAT_CANDS)
    ev_col = guess_col(d, EVENT_CANDS)
    if pat_col is None or ev_col is None:
        return d, {"status": "skipped_missing_pat_or_event", "rows_before": int(len(d)), "rows_after": int(len(d))}

    pat = d[pat_col].astype("string").str.strip()
    evn = normalize_event_series(d[ev_col])

    has_sc = evn.eq("SC").groupby(pat, dropna=False).any()
    has_bl = evn.eq("BL").groupby(pat, dropna=False).any()
    both_pat = set((has_sc & has_bl)[lambda x: x].index.astype("string").tolist())

    mask_drop = pat.isin(both_pat) & evn.eq("SC")
    out = d.loc[~mask_drop].copy()

    return out, {
        "status": "ok",
        "rows_before": int(len(d)),
        "rows_after": int(len(out)),
        "sc_rows_dropped": int(mask_drop.sum()),
        "patients_with_both_sc_bl": int(len(both_pat)),
    }




def canonicalize_baseline_sc_to_bl(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enforce BL/SC canonicalization rule on true longitudinal tables:

      If BL exists for a subject -> keep BL (SC should already be dropped upstream)
      Else if only SC exists -> rename SC to BL

    After this step, true longitudinal tables will have *no* baseline SC rows.
    """
    d = df.copy()
    d.columns = [str(c).strip().upper() for c in d.columns]
    d = standardize_pat_event_cols(d)
    if "PATNO" not in d.columns or "EVENT_ID" not in d.columns:
        return d, {"status": "skipped_missing_pat_or_event", "rows_before": int(len(d)), "rows_after": int(len(d))}
    pat = d["PATNO"].astype("string").str.strip()
    evn = normalize_event_series(d["EVENT_ID"])
    d["EVENT_ID"] = evn

    has_bl = evn.eq("BL").groupby(pat, dropna=False).any()
    bl_pat = set(has_bl[lambda x: x].index.astype("string").tolist())

    mask_rename = evn.eq("SC") & ~pat.isin(bl_pat)
    renamed = int(mask_rename.sum())
    d.loc[mask_rename, "EVENT_ID"] = "BL"

    return d, {
        "status": "ok",
        "rows_before": int(len(df)),
        "rows_after": int(len(d)),
        "sc_renamed_to_bl": int(renamed),
        "patients_with_bl": int(len(bl_pat)),
    }


# =============================================================================
# Semantic encoders for STATIC
# =============================================================================
def encode_demographics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    d = df.copy()
    d = standardize_pat_event_cols(d)
    rep: Dict[str, Any] = {"status": "ok"}

    if "SEX" in d.columns:
        s = d["SEX"]
        if s.dtype == object or str(s.dtype).startswith("string"):
            su = s.astype("string").str.strip().str.upper()
            d["SEX"] = np.where(su.isin(["M", "MALE", "1"]), 1,
                         np.where(su.isin(["F", "FEMALE", "2"]), 0, np.nan))
        else:
            x = pd.to_numeric(s, errors="coerce")
            uniq = sorted(set(x.dropna().unique().tolist()))
            if set(uniq).issubset({0, 1}):
                d["SEX"] = x
            elif set(uniq).issubset({1, 2}):
                d["SEX"] = np.where(x == 1, 1, np.where(x == 2, 0, np.nan))
            else:
                d["SEX"] = x

    for c in ["HISPLAT", "RAASIAN", "RABLACK", "RAHAWOPI", "RAINDALS", "RANOS", "RAWHITE", "RAUNKNOWN"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    rep["columns"] = [c for c in d.columns if c != "PATNO"]
    return d.drop(columns=["EVENT_ID"], errors="ignore"), rep

def encode_participant_status(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    d = df.copy()
    d.columns = [str(c).strip().upper() for c in d.columns]
    d = standardize_pat_event_cols(d)
    rep: Dict[str, Any] = {"status": "ok"}

    if "COHORT_DEFINITION" in d.columns:
        cd = d["COHORT_DEFINITION"].astype("string").str.strip().str.upper()
        d["IS_PD"] = np.where(cd.str.contains("PARKINSON", na=False), 1,
                       np.where(cd.str.contains("HEALTHY", na=False), 0, np.nan))
    else:
        d["IS_PD"] = np.nan
        rep["warn"] = "COHORT_DEFINITION missing; IS_PD set to NaN"

    if "COHORT" in d.columns:
        d["COHORT"] = pd.to_numeric(d["COHORT"], errors="coerce")

    if "ENROLL_AGE" in d.columns:
        d["ENROLL_AGE"] = d["ENROLL_AGE"].map(_parse_numeric)

    drop = [c for c in ["COHORT_DEFINITION", "ENROLL_STATUS", "EVENT_ID"] if c in d.columns]
    d = d.drop(columns=drop, errors="ignore")

    rep["columns"] = [c for c in d.columns if c != "PATNO"]
    return d, rep


# =============================================================================
# genetic_consensus transform
# =============================================================================
def _apoe_allele_counts(apoe_val) -> tuple[float, float, float]:
    if pd.isna(apoe_val):
        return (np.nan, np.nan, np.nan)
    s = str(apoe_val).strip().upper()
    if s == "" or s in {"NA", "NAN", "NONE"}:
        return (np.nan, np.nan, np.nan)
    alleles = re.findall(r"E?([234])", s)
    if len(alleles) < 2:
        return (np.nan, np.nan, np.nan)
    alleles = alleles[:2]
    c2 = float(sum(a == "2" for a in alleles))
    c3 = float(sum(a == "3" for a in alleles))
    c4 = float(sum(a == "4" for a in alleles))
    return (c2, c3, c4)

def transform_genetic_consensus_for_diffusion(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    d = df.copy()
    d.columns = [str(c).strip().upper() for c in d.columns]
    pat = guess_col(d, PAT_CANDS)
    if pat is None:
        raise ValueError("genetic_consensus: no PATNO-like column found")

    apoe = guess_col(d, ["APOE"])
    var_gene = guess_col(d, ["VAR_GENE", "VARGENE", "VAR GENE"])
    present_pathvar = [c for c in ["PATHVAR_COUNT", "PATHVAR", "PATHVAR_", "PATHVARCOUNT"] if c in d.columns]

    if apoe is not None:
        counts = d[apoe].map(_apoe_allele_counts)
        d["APOE_E2"] = counts.map(lambda t: t[0])
        d["APOE_E3"] = counts.map(lambda t: t[1])
        d["APOE_E4"] = counts.map(lambda t: t[2])
        d = d.drop(columns=[apoe], errors="ignore")

    gene_cols = [
        c for c in d.columns
        if c not in {pat, "APOE_E2", "APOE_E3", "APOE_E4"} | set(present_pathvar) | ({var_gene} if var_gene else set())
        and c not in {"NOTES", "NOTE"}
    ]
    for c in gene_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    unknown = set()
    if var_gene is not None and gene_cols:
        vg = d[var_gene].astype("string").fillna("").str.strip().str.upper()
        mask_has = vg.ne("")
        if mask_has.any():
            d.loc[mask_has, gene_cols] = 0
            gene_set = set(gene_cols)

            def _tokens(x: str) -> List[str]:
                parts = re.split(r"[;,|/]+|\s+", x)
                out = []
                for p in parts:
                    p = p.strip().upper()
                    if not p:
                        continue
                    p_clean = re.sub(r"[^A-Z0-9]+", "", p)
                    out.append(p)
                    if p_clean != p:
                        out.append(p_clean)
                return out

            for idx, s in vg[mask_has].items():
                toks = _tokens(str(s))
                matched = False
                for t in toks:
                    if t in gene_set:
                        d.at[idx, t] = 1
                        matched = True
                if not matched:
                    unknown.add(str(s))

    drop_cols = list(present_pathvar)
    if var_gene is not None:
        drop_cols.append(var_gene)
    d = d.drop(columns=[c for c in drop_cols if c in d.columns], errors="ignore")

    d[pat] = d[pat].astype("string").str.strip()
    if pat != "PATNO":
        d = d.rename(columns={pat: "PATNO"})

    rep = {
        "pat_col": pat,
        "dropped_pathvar_cols": present_pathvar,
        "dropped_var_gene_col": var_gene,
        "n_gene_cols": int(len(gene_cols)),
        "unknown_var_gene_strings_sample": sorted(list(unknown))[:50],
    }
    return d, rep


# =============================================================================
# blood chemistry filtering (handles long AND wide blood chemistry)
# =============================================================================
def filter_blood_chemistry_only(
    dfs: Dict[str, pd.DataFrame],
    blood_codes_keep: List[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    out = dict(dfs)
    report: Dict[str, Any] = {}

    # --- blood chemistry ---
    if "blood_chemistry" in out:
        df = out["blood_chemistry"].copy()
        df.columns = [str(c).strip().upper() for c in df.columns]
        pat = guess_col(df, PAT_CANDS) or "PATNO"
        ev = guess_col(df, EVENT_CANDS) or "EVENT_ID"
        code_col = guess_col(df, ["LTSTCODE"])
        keep_codes = set(_norm_loose(c) for c in blood_codes_keep)

        if code_col is not None:
            codes = df[code_col].astype("string").map(_norm_loose)
            m = codes.isin(keep_codes)
            out["blood_chemistry"] = df.loc[m].copy()
            report["blood_chemistry"] = {"status": "ok_long", "rows_before": int(len(df)), "rows_after": int(m.sum())}
        else:
            # wide: filter columns if possible (match by column names)
            feat_cols = [c for c in df.columns if c not in {pat, ev}]
            matched = [c for c in feat_cols if _norm_loose(c) in keep_codes]
            if matched:
                out["blood_chemistry"] = df[[pat, ev] + matched].copy()
                report["blood_chemistry"] = {
                    "status": "ok_wide_cols_filtered",
                    "rows_before": int(len(df)),
                    "rows_after": int(len(df)),
                    "n_cols_kept": int(len(matched)),
                }
            else:
                out["blood_chemistry"] = df
                report["blood_chemistry"] = {"status": "wide_no_code_col_keep_all", "rows_before": int(len(df)), "rows_after": int(len(df))}

    return out, report


# =============================================================================
# Pre-pivot fixes
# =============================================================================
def fix_age_at_visit_duplicates(df_age: pd.DataFrame, round_decimals: int = 1) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    d = df_age.copy()
    d.columns = [str(c).strip().upper() for c in d.columns]
    d = standardize_pat_event_cols(d)
    if "PATNO" not in d.columns or "EVENT_ID" not in d.columns or "AGE_AT_VISIT" not in d.columns:
        return d, {"status": "skipped_missing_cols"}

    d["EVENT_ID"] = normalize_event_series(d["EVENT_ID"])
    d["_AGE_NUM"] = d["AGE_AT_VISIT"].map(_parse_numeric)

    gsize = d.groupby(["PATNO", "EVENT_ID"], dropna=False).size()
    n_dup_groups = int((gsize > 1).sum())

    out = (
        d.groupby(["PATNO", "EVENT_ID"], as_index=False, dropna=False)
         .agg({**{c: "first" for c in d.columns if c not in {"AGE_AT_VISIT", "_AGE_NUM"}}, "_AGE_NUM": "mean"})
    )
    out["AGE_AT_VISIT"] = out["_AGE_NUM"].round(round_decimals)
    out = out.drop(columns=["_AGE_NUM"], errors="ignore")
    return out, {"status": "ok", "rows_before": int(len(d)), "rows_after": int(len(out)), "groups_with_dups": n_dup_groups}

def canonicalize_updrs3_state_priority(df_updrs3: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    State-priority canonicalization.

    For each PATNO × EVENT_ID:
      - if ON exists → use ON
      - else if OFF exists → use OFF
      - else → use whatever exists (UNK)
    Then create one-hot flags:
      PDSTATE_USED_ON / PDSTATE_USED_OFF / PDSTATE_USED_UNK  (float 0/1)

    Rationale:
      - increases N (keeps OFF + UNK when ON is absent),
      - avoids mixing ON and OFF by averaging,
      - provides medically interpretable context without keeping raw PDSTATE strings.
    """
    d = df_updrs3.copy()
    d.columns = [str(c).strip().upper() for c in d.columns]
    d = standardize_pat_event_cols(d)
    if "PATNO" not in d.columns or "EVENT_ID" not in d.columns:
        return d, {"status": "skipped_missing_pat_or_event"}
    if "PDSTATE" not in d.columns:
        return d, {"status": "skipped_missing_pdstate"}

    d["EVENT_ID"] = normalize_event_series(d["EVENT_ID"])

    # Normalize PDSTATE into {ON, OFF, UNK}
    s = d["PDSTATE"].astype("string").str.strip().str.upper()
    s = s.fillna("")
    s = s.replace({
        "ON STATE": "ON", "OFF STATE": "OFF",
        "ON-MED": "ON", "OFF-MED": "OFF",
        "ONMED": "ON", "OFFMED": "OFF",
    })

    def _pdstate_tag(x: str) -> str:
        x = (x or "").strip().upper()
        if x == "":
            return "UNK"
        if x == "ON" or re.search(r"ON", x):
            return "ON"
        if x == "OFF" or re.search(r"OFF", x):
            return "OFF"
        return "UNK"

    tag = s.map(_pdstate_tag)
    d["_PDSTATE_TAG"] = tag

    rank_map = {"ON": 0, "OFF": 1, "UNK": 2}
    d["_PDSTATE_RANK"] = d["_PDSTATE_TAG"].map(rank_map).fillna(2).astype(int)

    before = int(len(d))

    # Keep only best rank rows per (PATNO, EVENT_ID)
    best = (
        d.groupby(["PATNO", "EVENT_ID"], dropna=False)["_PDSTATE_RANK"]
         .transform("min")
    )
    d = d.loc[d["_PDSTATE_RANK"] == best].copy()

    # Parse numeric columns (UPDRS items), then aggregate duplicates by median
    value_cols = [c for c in d.columns if c not in {"PATNO", "EVENT_ID", "PDSTATE", "_PDSTATE_TAG", "_PDSTATE_RANK"}]
    for c in value_cols:
        d[c] = d[c].map(_parse_numeric)

    # One-hot PDSTATE_USED (after selection)
    d["PDSTATE_USED_ON"] = (d["_PDSTATE_TAG"] == "ON").astype(float)
    d["PDSTATE_USED_OFF"] = (d["_PDSTATE_TAG"] == "OFF").astype(float)
    d["PDSTATE_USED_UNK"] = (d["_PDSTATE_TAG"] == "UNK").astype(float)

    # Collapse remaining duplicates (e.g. multiple ON rows) safely
    agg = {c: "median" for c in value_cols}
    agg.update({"PDSTATE_USED_ON": "max", "PDSTATE_USED_OFF": "max", "PDSTATE_USED_UNK": "max"})
    out = (
        d.groupby(["PATNO", "EVENT_ID"], as_index=False, dropna=False)
         .agg(agg)
    )

    # Drop raw PDSTATE
    out = out.drop(columns=["PDSTATE"], errors="ignore")

    rep = {
        "status": "ok_state_priority",
        "rows_before": before,
        "rows_after": int(len(out)),
        "kept_state_counts": {
            "ON": int((out["PDSTATE_USED_ON"] > 0.5).sum()) if "PDSTATE_USED_ON" in out.columns else 0,
            "OFF": int((out["PDSTATE_USED_OFF"] > 0.5).sum()) if "PDSTATE_USED_OFF" in out.columns else 0,
            "UNK": int((out["PDSTATE_USED_UNK"] > 0.5).sum()) if "PDSTATE_USED_UNK" in out.columns else 0,
        },
    }
    return out, rep


# =============================================================================
# Pivoting (true long + baseline sources)
# =============================================================================
def pivot_longitudinal_to_patno_visit(
    dfs_long: Dict[str, pd.DataFrame],
    *,
    long_specs: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    For long-format tables:
      - build feature keys that do not mix incompatible measurements
      - parse values (STRICT for blood_chemistry)
      - if duplicate rows exist for the same PATNO×EVENT×feature -> median
      - pivot to PATNO×EVENT wide

    IMPORTANT: never use dropna=False in pivot_table (creates phantom rows/cols).
    """
    if long_specs is None:
        long_specs = {
            "blood_chemistry": {"feature": "LTSTCODE", "value": "LSIRES"},
        }
    out: Dict[str, pd.DataFrame] = {}
    report: Dict[str, Any] = {}

    for name, df0 in dfs_long.items():
        df = df0.copy()
        df.columns = [str(c).strip().upper() for c in df.columns]
        df = standardize_pat_event_cols(df)

        if "PATNO" not in df.columns or "EVENT_ID" not in df.columns:
            out[name] = df.iloc[0:0].copy()
            report[name] = {"status": "skipped_missing_pat_or_event"}
            continue

        df["EVENT_ID"] = normalize_event_series(df["EVENT_ID"])
        spec = long_specs.get(name)

        # -----------------------------
        # long-format pivot if spec matches
        # -----------------------------
        if spec is not None:
            feat = spec["feature"].upper()
            val = spec["value"].upper()
            if feat in df.columns and val in df.columns:

                feat_s = df[feat].astype("string").str.strip()
                if name == "blood_chemistry":
                    # Codes are fine; include unit if present and varying (rare but safe)
                    if "LSIUNIT" in df.columns:
                        u = df["LSIUNIT"].astype("string").str.strip()
                        feat_s = feat_s.where(u.isna() | (u == ""), feat_s + "__" + u)
                    vals = df[val].map(_parse_numeric_strict)

                else:
                    vals = df[val].map(_parse_numeric)

                tmp = pd.DataFrame(
                    {"PATNO": df["PATNO"], "EVENT_ID": df["EVENT_ID"], "_FEAT": feat_s, "_VAL": vals}
                )
                tmp = tmp.loc[tmp["_VAL"].notna()].copy()

                if tmp.empty:
                    out[name] = tmp[["PATNO", "EVENT_ID"]].drop_duplicates().copy()
                    report[name] = {"status": "ok_empty_after_numeric_parse", "rows_before": int(len(df))}
                    continue

                # Resolve duplicates within the same PATNO×EVENT×FEAT by median
                g = tmp.groupby(["PATNO", "EVENT_ID", "_FEAT"], dropna=False)["_VAL"]
                dup_groups = int((g.size() > 1).sum())
                tmp2 = g.median().reset_index()

                wide = (
                    tmp2.pivot_table(
                        index=["PATNO", "EVENT_ID"],
                        columns="_FEAT",
                        values="_VAL",
                        aggfunc="first",
                        observed=True,
                        dropna=True,   # default; keep explicit for readability
                    )
                    .reset_index()
                )
                wide.columns = [str(c) for c in wide.columns]
                out[name] = wide
                report[name] = {
                    "status": "ok_pivot_long_median_dups",
                    "rows_before": int(len(df)),
                    "rows_after": int(len(wide)),
                    "n_features": int(wide.shape[1] - 2),
                    "dup_groups_resolved_by_median": dup_groups,
                    "n_numeric_rows_used": int(len(tmp)),
                }
                continue

        # -----------------------------
        # fallback: already-wide tables
        # -----------------------------
        value_cols = [c for c in df.columns if c not in {"PATNO", "EVENT_ID"}]
        if not value_cols:
            out[name] = df.drop_duplicates(["PATNO", "EVENT_ID"]).copy()
            report[name] = {"status": "ok_no_value_cols", "rows_before": int(len(df)), "rows_after": int(len(out[name]))}
            continue

        # Convert numeric-like columns; median for duplicates
        d2 = df.copy()
        agg: Dict[str, str] = {}
        for c in value_cols:
            x = d2[c].map(_parse_numeric)
            if x.notna().any():
                d2[c] = x
                agg[c] = "median"
            else:
                agg[c] = "first"

        out[name] = d2.groupby(["PATNO", "EVENT_ID"], as_index=False, dropna=False).agg(agg)
        report[name] = {
            "status": "ok_groupby_wide",
            "rows_before": int(len(df)),
            "rows_after": int(len(out[name])),
            "n_features": int(len(value_cols)),
        }

    return out, report


# =============================================================================
# Baseline extractor (BL preferred, SC fallback; no BL/SC suffix blocks)
# =============================================================================
def extract_baseline_static_from_pivoted(df_pivoted: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Input:  pivoted PATNO×EVENT wide table
    Output: one-row-per-PATNO baseline table, BL preferred, SC fallback.
            NO __BL / __SC duplicated feature blocks.
    """
    d = standardize_pat_event_cols(df_pivoted)
    if "PATNO" not in d.columns or "EVENT_ID" not in d.columns:
        return d.iloc[0:0].copy(), {"status": "failed_missing_patno_or_event"}

    d["PATNO"] = d["PATNO"].astype("string").str.strip()
    d["EVENT_ID"] = normalize_event_series(d["EVENT_ID"])
    d = d.loc[d["EVENT_ID"].isin(["BL", "SC"])].copy()
    if d.empty:
        return d.iloc[0:0].copy(), {"status": "empty_after_baseline_filter"}

    rank_map = {"BL": 0, "SC": 1}
    d["_BASELINE_RANK"] = d["EVENT_ID"].map(rank_map).fillna(99).astype(int)

    # If duplicates exist for the same PATNO+EVENT after upstream pivoting, keep first
    n_dups_pat_event = int(d.duplicated(subset=["PATNO", "EVENT_ID"], keep=False).sum())
    d = (
        d.sort_values(["PATNO", "_BASELINE_RANK"])
         .drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")
         .copy()
    )

    has_bl = set(d.loc[d["EVENT_ID"] == "BL", "PATNO"].astype("string").tolist())
    has_sc = set(d.loc[d["EVENT_ID"] == "SC", "PATNO"].astype("string").tolist())
    both = has_bl & has_sc

    out = (
        d.sort_values(["PATNO", "_BASELINE_RANK"])
         .drop_duplicates(subset=["PATNO"], keep="first")
         .drop(columns=["_BASELINE_RANK", "EVENT_ID"], errors="ignore")
         .reset_index(drop=True)
    )

    rep = {
        "status": "ok_bl_preferred_sc_fallback_no_suffix",
        "n_patnos_out": int(out.shape[0]),
        "n_features_out": int(out.shape[1] - 1) if "PATNO" in out.columns else int(out.shape[1]),
        "patients_with_bl": int(len(has_bl)),
        "patients_with_sc": int(len(has_sc)),
        "patients_with_both_bl_sc": int(len(both)),
        "patients_sc_fallback_only": int(len(has_sc - has_bl)),
        "duplicate_patno_event_rows_seen": int(n_dups_pat_event),
    }
    return out, rep


# =============================================================================
# CSF biomarker extraction (separate from blood_chemistry pipeline)
# =============================================================================
def extract_csf_biomarkers(
    filepath: Path,
    test_names: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract baseline CSF biomarkers from Current_Biospecimen_Analysis_Results.csv.

    The file is in long format: PATNO × CLINICAL_EVENT × TYPE × TESTNAME × TESTVALUE.
    We filter to CSF type, BL event, and key biomarker tests, then pivot to wide.
    Duplicates per PATNO×TESTNAME are aggregated by median.
    """
    if not filepath.exists():
        return pd.DataFrame(columns=["PATNO"]), {"status": "file_not_found"}

    df = pd.read_csv(
        filepath,
        usecols=["PATNO", "CLINICAL_EVENT", "TYPE", "TESTNAME", "TESTVALUE"],
        low_memory=False,
        na_values=NA_VALUES,
    )

    # Filter to CSF type
    csf = df[df["TYPE"].str.contains("CSF|Cerebrospinal", case=False, na=False)].copy()

    # Filter to baseline event
    csf = csf[csf["CLINICAL_EVENT"].str.contains("^BL$|Baseline|Screen", case=False, na=False)].copy()

    # Filter to requested biomarkers
    csf = csf[csf["TESTNAME"].isin(test_names)].copy()

    if csf.empty:
        return pd.DataFrame(columns=["PATNO"]), {"status": "empty_after_filter", "n_tests": 0}

    # Parse values as numeric (strict)
    csf["TESTVALUE"] = pd.to_numeric(csf["TESTVALUE"], errors="coerce")
    csf = csf.dropna(subset=["TESTVALUE"])

    # Clean PATNO
    csf["PATNO"] = csf["PATNO"].astype(str).str.strip()

    # Pivot: PATNO × TESTNAME -> wide, aggregate duplicates by median
    wide = csf.pivot_table(
        index="PATNO",
        columns="TESTNAME",
        values="TESTVALUE",
        aggfunc="median",
    ).reset_index()

    # Ensure clean column names (replace spaces/special chars)
    wide.columns = [str(c).replace(" ", "_").replace("-", "_") for c in wide.columns]

    n_patients = wide["PATNO"].nunique()
    per_test = {}
    for t in test_names:
        clean_name = t.replace(" ", "_").replace("-", "_")
        if clean_name in wide.columns:
            per_test[t] = int(wide[clean_name].notna().sum())
        else:
            per_test[t] = 0

    rep = {
        "status": "ok",
        "n_patients": n_patients,
        "n_features": len(wide.columns) - 1,
        "per_test_coverage": per_test,
    }
    return wide, rep


def extract_plasma_biomarkers(
    filepath: Path,
    test_names: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract baseline plasma biomarkers from Current_Biospecimen_Analysis_Results.csv.

    Similar to CSF extraction but filters to Plasma type.
    Handles below-LOD values (e.g. "<0.0560") by substituting LOD/sqrt(2).
    Duplicates per PATNO×TESTNAME are aggregated by median.
    """
    if not filepath.exists():
        return pd.DataFrame(columns=["PATNO"]), {"status": "file_not_found"}

    df = pd.read_csv(
        filepath,
        usecols=["PATNO", "CLINICAL_EVENT", "TYPE", "TESTNAME", "TESTVALUE"],
        low_memory=False,
        na_values=NA_VALUES,
    )

    # Filter to Plasma type
    plasma = df[df["TYPE"].str.contains("Plasma", case=False, na=False)].copy()

    # Filter to baseline event
    plasma = plasma[plasma["CLINICAL_EVENT"].str.contains(
        "^BL$|Baseline|Screen", case=False, na=False
    )].copy()

    # Filter to requested biomarkers
    plasma = plasma[plasma["TESTNAME"].isin(test_names)].copy()

    if plasma.empty:
        return pd.DataFrame(columns=["PATNO"]), {"status": "empty_after_filter", "n_tests": 0}

    # Parse values: handle below-LOD entries like "<0.0560"
    def _parse_plasma_value(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == "":
            return np.nan
        # Below LOD: "<0.0560" -> LOD/sqrt(2)
        m = re.match(r"^<\s*([0-9.]+)$", s)
        if m:
            return float(m.group(1)) / np.sqrt(2)
        try:
            return float(s)
        except ValueError:
            return np.nan

    plasma["TESTVALUE"] = plasma["TESTVALUE"].map(_parse_plasma_value)
    plasma = plasma.dropna(subset=["TESTVALUE"])

    # Clean PATNO
    plasma["PATNO"] = plasma["PATNO"].astype(str).str.strip()

    # Pivot: PATNO × TESTNAME -> wide, aggregate duplicates by median
    wide = plasma.pivot_table(
        index="PATNO",
        columns="TESTNAME",
        values="TESTVALUE",
        aggfunc="median",
    ).reset_index()

    # Clean column names
    wide.columns = [str(c).replace(" ", "_").replace("-", "_") for c in wide.columns]

    n_patients = wide["PATNO"].nunique()
    per_test = {}
    for t in test_names:
        clean_name = t.replace(" ", "_").replace("-", "_")
        if clean_name in wide.columns:
            per_test[t] = int(wide[clean_name].notna().sum())
        else:
            per_test[t] = 0

    rep = {
        "status": "ok",
        "n_patients": n_patients,
        "n_features": len(wide.columns) - 1,
        "per_test_coverage": per_test,
    }
    return wide, rep


# =============================================================================
# Main
# =============================================================================
def main(visits_to_export: List[str]) -> None:
    ensure_dir(OUT_META)
    ensure_dir(OUT_STATIC)
    ensure_dir(OUT_LONG)

    run_report: Dict[str, Any] = {"inputs": {}, "steps": {}, "outputs": {}}

    # --- metadata dumps ---
    for nm, fp in {"code_list": CODE_LIST, "data_dict": DATA_DICT}.items():
        if fp.exists():
            dfm = read_csv_safely(fp)
            out_fp = OUT_META / f"{nm}.csv.gz"
            write_df(dfm, out_fp)
            run_report["outputs"][nm] = {"path": str(out_fp), "shape": [int(dfm.shape[0]), int(dfm.shape[1])]}

    # --- load raw ---
    dfs_static_raw = {nm: read_csv_safely(fp) for nm, fp in STATIC_FILES.items()}
    dfs_long_raw = {nm: read_csv_safely(fp) for nm, fp in LONG_FILES.items()}
    dfs_base_raw = {nm: read_csv_safely(fp) for nm, fp in BASELINE_STATIC_FROM_LONG_FILES.items()}

    # All PATNOs flow through here; cohort restriction (e.g. to Parkinson's
    # Disease) is applied downstream in build_dataset.py.

    # --- subset static (soft) ---
    dfs_static = dict(dfs_static_raw)
    dfs_static["demographics"] = safe_keep_cols_soft(dfs_static["demographics"], demo_cols, "demographics")
    dfs_static["genetic_consensus"] = safe_keep_cols_soft(dfs_static["genetic_consensus"], gen_cons_cols, "genetic_consensus")
    dfs_static["participant_status"] = safe_keep_cols_soft(dfs_static["participant_status"], part_stat_cols, "participant_status")
    # genetic_prs: keep all (we'll drop non-numeric later)

    # --- subset longitudinal (soft) ---
    dfs_long = dict(dfs_long_raw)
    dfs_long["age_at_visit"] = safe_keep_cols_strict_long(dfs_long["age_at_visit"], age_cols, "age_at_visit")
    dfs_long["moca"] = safe_keep_cols_strict_long(dfs_long["moca"], moca_cols, "moca")
    dfs_long["updrs1"] = safe_keep_cols_strict_long(dfs_long["updrs1"], updrs1_cols, "updrs1")
    dfs_long["updrs3"] = safe_keep_cols_strict_long(dfs_long["updrs3"], updrs3_cols, "updrs3")
    dfs_long["neurological_exam"] = safe_keep_cols_strict_long(dfs_long["neurological_exam"], neuro_cols, "neurological_exam")
    dfs_long["vital_signs"] = safe_keep_cols_strict_long(dfs_long["vital_signs"], vital_cols, "vital_signs")

    # --- baseline sources: keep blood chemistry ---
    dfs_base = dict(dfs_base_raw)
    dfs_base["blood_chemistry"] = dfs_base["blood_chemistry"].copy()

    # --- semantic encodings for static ---
    dfs_static["demographics"], demo_rep = encode_demographics(dfs_static["demographics"])
    dfs_static["participant_status"], ps_rep = encode_participant_status(dfs_static["participant_status"])
    dfs_static["genetic_consensus"], gc_rep = transform_genetic_consensus_for_diffusion(dfs_static["genetic_consensus"])
    run_report["steps"]["encode_demographics"] = demo_rep
    run_report["steps"]["encode_participant_status"] = ps_rep
    run_report["steps"]["genetic_consensus_transform"] = gc_rep
    write_json(demo_rep, OUT_META / "encode_demographics_report.json")
    write_json(ps_rep, OUT_META / "encode_participant_status_report.json")
    write_json(gc_rep, OUT_META / "genetic_consensus_transform_report.json")

    # --- genetic_prs: keep PATNO + numeric columns only ---
    if "genetic_prs" in dfs_static:
        g = standardize_pat_event_cols(dfs_static["genetic_prs"]).drop(columns=["EVENT_ID"], errors="ignore")
        num_cols = []
        for c in g.columns:
            if c == "PATNO":
                continue
            x = pd.to_numeric(g[c], errors="coerce")
            if x.notna().any():
                g[c] = x
                num_cols.append(c)
        dfs_static["genetic_prs"] = g[["PATNO"] + num_cols].copy()

    # --- events filter ---
    ev_reports = {}
    for nm, df in dfs_long.items():
        dfs_long[nm], ev_reports[nm] = filter_long_by_events(df, ALLOWED_EVENT_RE)

    for nm, df in dfs_base.items():
        dfs_base[nm], ev_reports[f"baseline::{nm}"] = filter_long_by_events(df, BASELINE_ONLY_EVENT_RE)

    run_report["steps"]["events_filter"] = ev_reports
    write_json(ev_reports, OUT_META / "events_filter_report.json")

    # --- baseline-source BL/SC canonicalization (blood chemistry used as STATIC) ---
    base_scbl_report = {}
    for nm in list(dfs_base.keys()):
        dfs_base[nm], rep_drop = drop_sc_if_bl_exists(dfs_base[nm])
        dfs_base[nm], rep_canon = canonicalize_baseline_sc_to_bl(dfs_base[nm])
        base_scbl_report[nm] = {
            "drop_sc_if_bl_exists": rep_drop,
            "canonicalize_sc_to_bl": rep_canon,
        }
    run_report["steps"]["baseline_sources_bl_sc_canonicalization"] = base_scbl_report
    write_json(base_scbl_report, OUT_META / "baseline_sources_bl_sc_canonicalization_report.json")

    # --- drop SC if BL exists: ONLY for true longitudinal ---
    scbl_report = {}
    for nm in list(dfs_long.keys()):
        dfs_long[nm], scbl_report[nm] = drop_sc_if_bl_exists(dfs_long[nm])
    run_report["steps"]["drop_sc_if_bl"] = scbl_report
    write_json(scbl_report, OUT_META / "drop_sc_if_bl_report.json")


    # --- canonicalize baseline SC->BL for SC-only subjects (true longitudinal only) ---
    canon_report = {}
    for nm in list(dfs_long.keys()):
        dfs_long[nm], canon_report[nm] = canonicalize_baseline_sc_to_bl(dfs_long[nm])
    run_report["steps"]["canonicalize_sc_to_bl"] = canon_report
    write_json(canon_report, OUT_META / "canonicalize_sc_to_bl_report.json")

    # --- blood chemistry filtering ---
    combined = dict(dfs_long)
    combined.update(dfs_base)
    combined, bb_rep = filter_blood_chemistry_only(combined, PD_CORE_PLUS_BASIC_PANEL_CODES)
    dfs_long = {k: combined[k] for k in dfs_long.keys()}
    dfs_base = {k: combined[k] for k in dfs_base.keys()}
    run_report["steps"]["biospec_blood_filter"] = bb_rep
    write_json(bb_rep, OUT_META / "biospec_blood_filter_report.json")

    # --- pre-pivot fixes ---
    pre_rep = {}
    if "age_at_visit" in dfs_long:
        dfs_long["age_at_visit"], pre_rep["age_at_visit"] = fix_age_at_visit_duplicates(dfs_long["age_at_visit"])
    if "updrs3" in dfs_long:
        dfs_long["updrs3"], pre_rep["updrs3"] = canonicalize_updrs3_state_priority(dfs_long["updrs3"])
    run_report["steps"]["pre_pivot_fixes"] = pre_rep
    write_json(pre_rep, OUT_META / "pre_pivot_fixes_report.json")

    # --- pivot all long-like (true long + baseline sources) ---
    piv_inputs = dict(dfs_long)
    piv_inputs.update(dfs_base)
    dfs_piv, piv_rep = pivot_longitudinal_to_patno_visit(piv_inputs)
    run_report["steps"]["pivot_report"] = piv_rep
    write_json(piv_rep, OUT_META / "pivot_report.json")

    # --- baseline static extraction (BL + SC blocks) ---
    base_rep = {}
    for nm in list(dfs_base.keys()):
        base_df, rep = extract_baseline_static_from_pivoted(dfs_piv.get(nm, pd.DataFrame()))
        base_rep[nm] = rep
        dfs_static[nm] = base_df
    run_report["steps"]["baseline_static_extraction"] = base_rep
    write_json(base_rep, OUT_META / "baseline_static_extraction_report.json")

    # --- CSF biomarkers (separate extraction from biospecimen file) ---
    if CSF_BIOSPECIMEN_FILE.exists():
        csf_df, csf_rep = extract_csf_biomarkers(CSF_BIOSPECIMEN_FILE, CSF_BIOMARKER_TESTS)
        dfs_static["csf_biomarkers"] = csf_df
        run_report["steps"]["csf_biomarkers"] = csf_rep
        write_json(csf_rep, OUT_META / "csf_biomarkers_report.json")
        print(f"[csf] extracted {csf_df.shape[0]} patients, {csf_df.shape[1]-1} features")
    else:
        print(f"[csf] WARN: {CSF_BIOSPECIMEN_FILE} not found, skipping CSF biomarkers")

    # --- Plasma biomarkers (NFL, GFAP, Ptau217p from biospecimen file) ---
    if CSF_BIOSPECIMEN_FILE.exists():
        plasma_df, plasma_rep = extract_plasma_biomarkers(CSF_BIOSPECIMEN_FILE, PLASMA_BIOMARKER_TESTS)
        dfs_static["plasma_biomarkers"] = plasma_df
        run_report["steps"]["plasma_biomarkers"] = plasma_rep
        write_json(plasma_rep, OUT_META / "plasma_biomarkers_report.json")
        print(f"[plasma] extracted {plasma_df.shape[0]} patients, {plasma_df.shape[1]-1} features")
    else:
        print(f"[plasma] WARN: {CSF_BIOSPECIMEN_FILE} not found, skipping plasma biomarkers")

    # --- write outputs ---
    manifest_rows = []

    # static
    for modality, df in dfs_static.items():
        df_out = standardize_pat_event_cols(df).drop(columns=["EVENT_ID"], errors="ignore")
        out_fp = OUT_STATIC / modality / "data.csv.gz"
        write_df(df_out, out_fp)
        manifest_rows.append({
            "kind": "static", "modality": modality, "visit": "", "path": str(out_fp),
            "n": int(df_out.shape[0]), "p": int(df_out.shape[1])
        })
        print(f"[write] static {modality}: {df_out.shape} -> {out_fp}")

    # longitudinal per visit (true longitudinal only)
    visits_norm = [normalize_event_series(pd.Series([v])).iloc[0] for v in visits_to_export]
    visits_norm = [v for v in visits_norm if pd.notna(v)]
    run_report["steps"]["visits_to_export"] = visits_norm

    for modality in dfs_long.keys():
        df = dfs_piv.get(modality)
        if df is None or df.empty:
            continue
        df_out = standardize_pat_event_cols(df)
        if "EVENT_ID" not in df_out.columns or "PATNO" not in df_out.columns:
            continue
        for v in visits_norm:
            sub = df_out.loc[df_out["EVENT_ID"].astype("string") == str(v)].copy()
            if sub.empty:
                continue
            if DROP_EVENT_COL_IN_PER_VISIT_EXPORT:
                sub = sub.drop(columns=["EVENT_ID"], errors="ignore")
            out_fp = OUT_LONG / modality / str(v) / "data.csv.gz"
            write_df(sub, out_fp)
            manifest_rows.append({
                "kind": "longitudinal", "modality": modality, "visit": str(v), "path": str(out_fp),
                "n": int(sub.shape[0]), "p": int(sub.shape[1])
            })
            print(f"[write] long {modality}/{v}: {sub.shape} -> {out_fp}")

    # --- blood_chemistry as LONGITUDINAL modality (in addition to static baseline) ---
    # Blood chemistry has longitudinal data at V04, V06, V08, V10 (not V02).
    # We process it separately from the baseline-only pipeline to produce per-visit files.
    if "blood_chemistry" in BASELINE_STATIC_FROM_LONG_FILES:
        print("\n[blood_chem_long] Processing blood chemistry as longitudinal modality...")
        bc_raw = read_csv_safely(BASELINE_STATIC_FROM_LONG_FILES["blood_chemistry"])
        # Filter to all allowed longitudinal events
        bc_long, bc_ev_rep = filter_long_by_events(bc_raw, ALLOWED_EVENT_RE)
        # Drop SC if BL exists
        bc_long, _ = drop_sc_if_bl_exists(bc_long)
        # Canonicalize SC -> BL for SC-only subjects
        bc_long, _ = canonicalize_baseline_sc_to_bl(bc_long)
        # Apply LTSTCODE filtering (same curated codes as static)
        bc_dict = {"blood_chemistry": bc_long}
        bc_dict, _ = filter_blood_chemistry_only(bc_dict, PD_CORE_PLUS_BASIC_PANEL_CODES)
        bc_long = bc_dict["blood_chemistry"]
        # Pivot long->wide
        bc_piv_dict, bc_piv_rep = pivot_longitudinal_to_patno_visit({"blood_chemistry": bc_long})
        bc_pivoted = bc_piv_dict.get("blood_chemistry")

        if bc_pivoted is not None and not bc_pivoted.empty:
            bc_out = standardize_pat_event_cols(bc_pivoted)
            if "EVENT_ID" in bc_out.columns and "PATNO" in bc_out.columns:
                bc_visit_count = 0
                for v in visits_norm:
                    sub = bc_out.loc[bc_out["EVENT_ID"].astype("string") == str(v)].copy()
                    if sub.empty:
                        continue
                    if DROP_EVENT_COL_IN_PER_VISIT_EXPORT:
                        sub = sub.drop(columns=["EVENT_ID"], errors="ignore")
                    out_fp = OUT_LONG / "blood_chemistry" / str(v) / "data.csv.gz"
                    write_df(sub, out_fp)
                    manifest_rows.append({
                        "kind": "longitudinal", "modality": "blood_chemistry",
                        "visit": str(v), "path": str(out_fp),
                        "n": int(sub.shape[0]), "p": int(sub.shape[1])
                    })
                    bc_visit_count += 1
                    print(f"[write] long blood_chemistry/{v}: {sub.shape} -> {out_fp}")
                run_report["steps"]["blood_chemistry_longitudinal"] = {
                    "status": "ok",
                    "visits_written": bc_visit_count,
                    "pivot_report": bc_piv_rep.get("blood_chemistry", {}),
                }
                print(f"[blood_chem_long] wrote {bc_visit_count} visits")
        else:
            print("[blood_chem_long] WARN: no data after pivot")
            run_report["steps"]["blood_chemistry_longitudinal"] = {"status": "empty_after_pivot"}

    manifest = pd.DataFrame(manifest_rows).sort_values(["kind", "modality", "visit"]).reset_index(drop=True)
    manifest_fp = OUT_META / "processed_manifest.csv.gz"
    write_df(manifest, manifest_fp)
    write_json(run_report, OUT_META / "run_report.json")
    print(f"\n[done] manifest -> {manifest_fp}")
    print(f"[done] run_report -> {OUT_META / 'run_report.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--visits", nargs="+", default=DEFAULT_VISITS_TO_EXPORT)
    args = ap.parse_args()
    main(visits_to_export=args.visits)