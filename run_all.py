#!/usr/bin/env python3
"""
PPMI Predictability-Frontier Benchmark: full pipeline entrypoint.

Runs the analysis pipeline end-to-end (build datasets, frontier benchmark,
ablation, statistical tests, calibration, SHAP, exploratory analysis).

Usage:
    python run_all.py                   # full pipeline (builds fresh by default)
    python run_all.py --step 2          # resume from step 2
    python run_all.py --max-tasks 5     # debug: only 5 tasks
    python run_all.py --max-seeds 2     # debug: only 2 CV seeds
    python run_all.py --no-force        # reuse existing build artifacts
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import PROCESSED_DATA_DIR, RESULTS_DIR
from utils.logging_utils import add_file_handler, get_logger

log = get_logger("run_all")


def main():
    parser = argparse.ArgumentParser(description="Full benchmark pipeline v2")
    parser.add_argument("--step", type=int, default=1, help="Start from step N (1-10)")
    parser.add_argument("--step-to", type=int, default=10,
                        help="Stop after step N inclusive (1-10). Use --step N --step-to N "
                             "to run a single step.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit tasks (debug)")
    parser.add_argument("--max-seeds", type=int, default=None, help="Limit CV seeds (debug)")
    parser.add_argument("--force", dest="force", action="store_true", default=True,
                        help="Force rebuild of datasets in Step 1 (default: True)")
    parser.add_argument("--no-force", dest="force", action="store_false",
                        help="Reuse existing build artifacts; skip Step 1 rebuild if possible")
    parser.add_argument("--no-predictions", action="store_true", help="Skip saving predictions")
    parser.add_argument("--fresh", action="store_true",
                        help="Purge stale pipeline outputs before running (only applies "
                             "when --step <= 1; ignored on resume to avoid wiping needed inputs)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summaries").mkdir(exist_ok=True)
    add_file_handler(log, RESULTS_DIR / "logs" / "run_all.log")

    t0_total = time.time()

    log.info("=" * 60)
    log.info("PPMI Predictability-Frontier Benchmark v2")
    from configs.config import N_OUTER_FOLDS as _N_OUTER, N_SEEDS as _N_SEEDS
    log.info("R^2-first | %d-fold CV x %d seeds",
             _N_OUTER, args.max_seeds or _N_SEEDS)
    log.info("Starting from step %d", args.step)
    log.info("=" * 60)

    # Full-purge pre-flight (only when --fresh AND starting from step 1).
    # Remove stale pipeline outputs so a fresh run starts from a clean state.
    if args.fresh and args.step <= 1:
        import shutil as _shutil
        _proj = Path(__file__).resolve().parent
        log.warning("--fresh: purging stale pipeline outputs from %s", _proj)
        for _d in ["predictions", "shards", "figures", "tables", "summaries",
                   "exploratory"]:
            _shutil.rmtree(RESULTS_DIR / _d, ignore_errors=True)
        for _f in ["frontier_results.csv", "frontier_results_per_fold.csv",
                   "r2_spearman_divergences.csv", "calibration_detail.json",
                   "qc_report.json"]:
            (RESULTS_DIR / _f).unlink(missing_ok=True)
        for _abl in (RESULTS_DIR).glob("ablation_*.csv"):
            _abl.unlink(missing_ok=True)
        log.warning("--fresh: purge complete")
    elif args.fresh:
        log.warning("--fresh ignored because --step=%d (only applies at step 1)", args.step)

    # Pre-flight: purge stale artifacts
    import glob as _glob
    for _v12_dir in _glob.glob(str(RESULTS_DIR / "predictions" / "*__V12")):
        import shutil as _shutil
        _shutil.rmtree(_v12_dir, ignore_errors=True)
        log.info("Pre-flight: removed stale V12 prediction dir %s", Path(_v12_dir).name)
    for _v12_dir in _glob.glob(str(PROCESSED_DATA_DIR / "tasks" / "*__V12")):
        import shutil as _shutil
        _shutil.rmtree(_v12_dir, ignore_errors=True)
        log.info("Pre-flight: removed stale V12 task dir %s", Path(_v12_dir).name)
    # Remove old split-based task files
    for _old in _glob.glob(str(PROCESSED_DATA_DIR / "tasks" / "*" / "train.csv.gz")):
        Path(_old).unlink(missing_ok=True)
    for _old in _glob.glob(str(PROCESSED_DATA_DIR / "tasks" / "*" / "val.csv.gz")):
        Path(_old).unlink(missing_ok=True)
    for _old in _glob.glob(str(PROCESSED_DATA_DIR / "tasks" / "*" / "test.csv.gz")):
        Path(_old).unlink(missing_ok=True)

    # Remove old patient_splits.csv (no longer used)
    _old_splits = PROCESSED_DATA_DIR / "patient_splits.csv"
    if _old_splits.exists():
        _old_splits.unlink()
        log.info("Pre-flight: removed old patient_splits.csv (CV replaces fixed splits)")

    # Step 1: Build datasets
    if args.step <= 1 <= args.step_to:
        log.info("STEP 1: Building datasets (full cohort, no splits)...")
        t0 = time.time()
        from data_preprocessing.build_dataset import run_pipeline
        manifest = run_pipeline(force=args.force)
        log.info("Step 1 done in %.1fs. %d tasks built.", time.time() - t0, len(manifest))

    # Step 2: Frontier benchmark (N_OUTER_FOLDS-fold CV x N_SEEDS seeds)
    if args.step <= 2 <= args.step_to:
        from configs.config import N_OUTER_FOLDS, N_SEEDS
        log.info("STEP 2: Running frontier benchmark (%d-fold CV x %d seeds)...",
                 N_OUTER_FOLDS, args.max_seeds or N_SEEDS)
        t0 = time.time()
        from evaluation.frontier import run_frontier
        results = run_frontier(
            max_tasks=args.max_tasks,
            max_seeds=args.max_seeds,
            save_predictions=not args.no_predictions,
        )
        log.info("Step 2 done in %.1fs. %d aggregated results.", time.time() - t0, len(results))

    # Step 3: Modality ablation
    if args.step <= 3 <= args.step_to:
        log.info("STEP 3: Running modality ablation...")
        t0 = time.time()
        from evaluation.ablation import run_ablation_suite
        ablation_configs = [
            ("baseline_multimodal", "V06"),
            ("baseline_multimodal", "V08"),
            ("baseline_plus_12m", "V08"),
            ("rolling", "V08"),
        ]
        for regime, horizon in ablation_configs:
            try:
                run_ablation_suite(regime=regime, horizon=horizon)
            except Exception as e:
                log.warning("Ablation %s/%s failed: %s", regime, horizon, e)
        log.info("Step 3 done in %.1fs.", time.time() - t0)

    # Step 4: Survivorship
    if args.step <= 4 <= args.step_to:
        log.info("STEP 4: Survivorship analysis...")
        t0 = time.time()
        from evaluation.survivorship import characterize_survivorship
        characterize_survivorship()
        log.info("Step 4 done in %.1fs.", time.time() - t0)

    # Step 5: Statistical tests (fold-aware, R^2-first)
    if args.step <= 5 <= args.step_to:
        log.info("STEP 5: Statistical tests (fold-aware, R^2 + Spearman)...")
        t0 = time.time()
        from evaluation.statistical_tests import run_all_statistical_tests
        run_all_statistical_tests()
        log.info("Step 5 done in %.1fs.", time.time() - t0)

    # Step 6: QC
    if args.step <= 6 <= args.step_to:
        log.info("STEP 6: QC checks...")
        t0 = time.time()
        from evaluation.qc_checks import run_all_qc_checks
        report = run_all_qc_checks()
        qc_status = report.get("overall", "UNKNOWN")
        log.info("Step 6 done in %.1fs. QC status: %s", time.time() - t0, qc_status)
        if qc_status == "FAIL":
            log.error("QC FAILED - review results/qc_report.json")

    # Step 7: LEDD (Levodopa Equivalent Daily Dose) sensitivity analysis.
    # Compares representative cross-modal tasks with vs. without LEDD as a
    # longitudinal feature; populates LEDD_MEAN_DELTA and LEDD_MAX_ABS_DELTA.
    if args.step <= 7 <= args.step_to:
        log.info("STEP 7: LEDD sensitivity analysis...")
        t0 = time.time()
        try:
            from scripts.sensitivity_ledd import main as ledd_main
            ledd_main()
            log.info("Step 7 done in %.1fs.", time.time() - t0)
        except Exception as e:
            log.warning("Step 7 (sensitivity_ledd) failed (non-fatal): %s", e)

    # Step 8: Calibration analysis
    if args.step <= 8 <= args.step_to:
        log.info("STEP 8: Calibration analysis...")
        t0 = time.time()
        try:
            from evaluation.calibration import run_calibration_analysis
            run_calibration_analysis()
            log.info("Step 8 done in %.1fs.", time.time() - t0)
        except Exception as e:
            log.warning("Step 8 (calibration) failed (non-fatal): %s", e)

    # Step 9: SHAP feature importance
    if args.step <= 9 <= args.step_to:
        log.info("STEP 9: SHAP feature importance...")
        t0 = time.time()
        try:
            from scripts.run_shap import run_shap_analysis
            run_shap_analysis()
            log.info("Step 9 done in %.1fs.", time.time() - t0)
        except Exception as e:
            log.warning("Step 9 (SHAP) failed (non-fatal): %s", e)

    # Step 10: Exploratory distributional analysis (supplementary)
    if args.step <= 10 <= args.step_to:
        log.info("STEP 10: Exploratory analysis (supplementary)...")
        t0 = time.time()
        try:
            from evaluation.exploratory import run_exploratory_analysis
            run_exploratory_analysis()
            log.info("Step 10 done in %.1fs.", time.time() - t0)
        except Exception as e:
            log.warning("Step 10 (exploratory) failed (non-fatal): %s", e)

    elapsed = time.time() - t0_total
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE in %.1fs (%.1f min)", elapsed, elapsed / 60)
    log.info("=" * 60)
    log.info("Key outputs:")
    log.info("  Frontier results (aggregated) : %s", RESULTS_DIR / "frontier_results.csv")
    log.info("  Frontier results (per-fold)   : %s", RESULTS_DIR / "frontier_results_per_fold.csv")
    log.info("  R^2/Spearman divergences      : %s", RESULTS_DIR / "r2_spearman_divergences.csv")
    log.info("  Exploratory analysis           : %s", RESULTS_DIR / "exploratory")
    log.info("  Figures                        : %s", RESULTS_DIR / "figures")
    log.info("  Tables                         : %s", RESULTS_DIR / "tables")
    log.info("  QC report                      : %s", RESULTS_DIR / "qc_report.json")


if __name__ == "__main__":
    main()
