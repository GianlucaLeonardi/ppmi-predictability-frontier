#!/usr/bin/env python3
"""
Validate that the benchmark environment is correctly set up.

Checks:
  1. Required packages are importable
  2. Raw data files exist
  3. Processed data directory exists with expected structure
  4. Config loads without errors
  5. Path consistency
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_imports():
    print("Checking imports...")
    required = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "matplotlib", "seaborn"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  {pkg}: OK")
        except ImportError:
            print(f"  {pkg}: MISSING")
            missing.append(pkg)
    return missing


def check_config():
    print("\nChecking config...")
    try:
        from configs.config import (
            RAW_DATA_DIR, EXISTING_PROCESSED, TARGETS, REGIMES,
            FORECAST_HORIZONS, VISIT_SCHEDULE, MODELS_REGRESSION, MODELS_CLASSIFICATION,
            MODALITY_FAMILIES,
        )
        print(f"  RAW_DATA_DIR: {RAW_DATA_DIR} ({'exists' if RAW_DATA_DIR.exists() else 'MISSING'})")
        print(f"  EXISTING_PROCESSED: {EXISTING_PROCESSED} ({'exists' if EXISTING_PROCESSED.exists() else 'MISSING'})")
        print(f"  Targets: {len(TARGETS)}")
        print(f"  Regimes: {[r.name for r in REGIMES]}")
        print(f"  Horizons: {FORECAST_HORIZONS}")
        print(f"  Visit schedule: {VISIT_SCHEDULE}")
        print(f"  Regression models: {list(MODELS_REGRESSION.keys())}")
        print(f"  Classification models: {list(MODELS_CLASSIFICATION.keys())}")
        print(f"  Modality families: {[m.name for m in MODALITY_FAMILIES]}")
        return True
    except Exception as e:
        print(f"  CONFIG ERROR: {e}")
        return False


def check_raw_data():
    print("\nChecking raw data files...")
    from configs.config import RAW_DATA_DIR

    expected_files = [
        "MDS-UPDRS_Part_III.csv",
        "MDS-UPDRS_Part_I.csv",
        "Montreal_Cognitive_Assessment__MoCA.csv",
        "Vital_Signs.csv",
        "Participant_Status.csv",
        "Demographics.csv",
    ]

    missing = []
    for f in expected_files:
        path = RAW_DATA_DIR / f
        if path.exists():
            print(f"  {f}: OK")
        else:
            print(f"  {f}: MISSING")
            missing.append(f)
    return missing


def check_processed_data():
    print("\nChecking processed data...")
    from configs.config import EXISTING_PROCESSED

    expected = [
        "static/demographics/data.csv.gz",
        "static/participant_status/data.csv.gz",
        "static/genetic_consensus/data.csv.gz",
        "longitudinal/updrs3/BL/data.csv.gz",
        "longitudinal/moca/BL/data.csv.gz",
        "metadata/patno_splits.csv.gz",
    ]

    missing = []
    for f in expected:
        path = EXISTING_PROCESSED / f
        if path.exists():
            print(f"  {f}: OK")
        else:
            print(f"  {f}: MISSING")
            missing.append(f)
    return missing


def main():
    print("=" * 60)
    print("PPMI Benchmark Setup Validation")
    print("=" * 60)

    missing_pkgs = check_imports()
    config_ok = check_config()
    missing_raw = check_raw_data()
    missing_proc = check_processed_data()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_ok = True
    if missing_pkgs:
        print(f"FAIL: Missing packages: {missing_pkgs}")
        all_ok = False
    else:
        print("OK: All packages available")

    if not config_ok:
        print("FAIL: Config errors")
        all_ok = False
    else:
        print("OK: Config loads correctly")

    if missing_raw:
        print(f"FAIL: Missing raw data: {missing_raw}")
        all_ok = False
    else:
        print("OK: Raw data present")

    if missing_proc:
        print(f"WARN: Missing processed data: {missing_proc}")
        print("  (This is OK if running preprocessing from scratch)")

    if all_ok:
        print("\nAll checks passed. Ready to run the pipeline.")
        return 0
    else:
        print("\nSome checks failed. Fix issues before running.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
