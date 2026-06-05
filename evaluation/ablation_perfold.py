"""
Per-fold modality ablation with full nested CV.

Drops one modality family at a time, re-tunes XGBoost via the main frontier
inner-CV machinery, and pairs the result with the full-feature per-fold R^2 in
`frontier_results_per_fold.csv` for fold-aware paired permutation tests.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import (
    CV_SEEDS,
    MODALITY_FAMILIES,
    MODELS_REGRESSION,
    N_OUTER_FOLDS,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from data_preprocessing.build_dataset import create_cv_folds
from evaluation.frontier import (
    load_task,
    _pop_raw_cols,
    _get_xy,
    _normalize_fold,
)
from evaluation.metrics import regression_metrics
from models.ml_models import get_model
from utils.logging_utils import get_logger

log = get_logger("ablation_perfold")


# Directory where per-shard CSVs land (analogous to results/shards/ for Step 2).
ABLATION_SHARDS_DIR = RESULTS_DIR / "ablation_shards"


def _identify_modality_columns(feat_names: List[str], modality_name: str) -> List[int]:
    return [i for i, name in enumerate(feat_names) if name.split("__")[0] == modality_name]


def _iter_outer_folds(patnos, y_all, task_type: str) -> Iterator[Tuple[int, int, dict]]:
    """Yield (seed, fold_idx, fold_dict) for the canonical 5x5 grid."""
    for seed in CV_SEEDS:
        folds = create_cv_folds(patnos, y_all, task_type, n_folds=N_OUTER_FOLDS, seed=seed)
        for fold_idx, fold in enumerate(folds):
            yield seed, fold_idx, fold


def _full_r2_lookup(per_fold_df: pd.DataFrame, task_key: str) -> Dict[Tuple[int, int], float]:
    """Map (seed, fold) -> full XGBoost R^2 from frontier_results_per_fold.csv."""
    sub = per_fold_df[
        (per_fold_df["task_key"] == task_key)
        & (per_fold_df["model"] == "xgboost")
    ]
    return {
        (int(row.seed), int(row.fold)): float(row.r2)
        for row in sub.itertuples(index=False)
        if np.isfinite(row.r2)
    }


def run_ablation_perfold_for_task(
    task_key: str,
    task_meta: Dict,
    full_r2_per_fold: Dict[Tuple[int, int], float],
    model_name: str = "xgboost",
) -> List[Dict]:
    """Per-fold leave-one-modality-out ablation; full-model R^2 looked up from full_r2_per_fold."""
    full_df = load_task(task_key)
    if full_df is None or len(full_df) < 20:
        log.warning("Task %s missing or too small; skipping", task_key)
        return []

    task_type = task_meta.get("task_type", "regression")
    if task_type != "regression":
        return []

    _ = _pop_raw_cols(full_df)
    feat_cols = [c for c in full_df.columns if c not in ("PATNO", "target")]
    y_all = full_df["target"].values.astype(np.float32)
    patnos = full_df["PATNO"].values

    model_cfg = MODELS_REGRESSION.get(model_name, {})

    rows: List[Dict] = []
    n_outer = len(CV_SEEDS) * N_OUTER_FOLDS

    for seed, fold_idx, fold in _iter_outer_folds(patnos, y_all, task_type):
        full_r2 = full_r2_per_fold.get((seed, fold_idx))
        if full_r2 is None:
            log.warning("  no full-model R^2 for %s seed=%d fold=%d -- skipping fold",
                        task_key, seed, fold_idx)
            continue

        train_df = full_df.iloc[fold["train_idx"]].reset_index(drop=True)
        test_df = full_df.iloc[fold["test_idx"]].reset_index(drop=True)

        train_normed, test_normed, _ = _normalize_fold(train_df, test_df, feat_cols)
        _, X_train_nan, y_train, fn = _get_xy(train_normed)
        _, X_test_nan, y_test, _ = _get_xy(test_normed)

        for mf in MODALITY_FAMILIES:
            drop_indices = _identify_modality_columns(fn, mf.name)
            if not drop_indices:
                continue
            keep_indices = [i for i in range(X_train_nan.shape[1]) if i not in drop_indices]
            if not keep_indices:
                continue

            log.info("    seed=%d fold=%d modality=%s n_dropped=%d ...",
                     seed, fold_idx, mf.name, len(drop_indices))
            t0 = time.time()
            try:
                model = get_model(model_name, task_type=task_type, config=model_cfg, seed=seed)
                model.fit(X_train_nan[:, keep_indices], y_train)
                y_pred = model.predict(X_test_nan[:, keep_indices])
                abl_metrics = regression_metrics(y_test, y_pred)
                ablated_r2 = float(abl_metrics.get("r2", float("nan")))
                fit_s = time.time() - t0
                delta = (full_r2 - ablated_r2) if np.isfinite(ablated_r2) else float("nan")
                log.info("    seed=%d fold=%d modality=%s done in %.1fs full_r2=%.4f ablated_r2=%.4f delta=%.4f",
                         seed, fold_idx, mf.name, fit_s, full_r2, ablated_r2, delta)
            except Exception as e:
                log.warning("    FAILED seed=%d fold=%d modality=%s: %s",
                            seed, fold_idx, mf.name, e)
                ablated_r2 = float("nan")
                delta = float("nan")

            rows.append({
                "task_key": task_key,
                "target": task_meta["target"],
                "target_display": task_meta["target_display"],
                "target_domain": task_meta["target_domain"],
                "horizon": task_meta["horizon"],
                "horizon_months": task_meta["horizon_months"],
                "regime": task_meta["regime"],
                "dropped_modality": mf.name,
                "dropped_modality_display": mf.display,
                "n_dropped_features": len(drop_indices),
                "seed": int(seed),
                "fold": int(fold_idx),
                "full_r2": full_r2,
                "ablated_r2": ablated_r2,
                "delta_r2": delta,
            })

    log.info("  task %s done: %d rows produced (expected ~%d = %d outer folds x %d modalities)",
             task_key, len(rows), n_outer * len(MODALITY_FAMILIES),
             n_outer, len(MODALITY_FAMILIES))
    return rows


def run_shard(
    target: str,
    shard_idx: int,
    manifest_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    incremental: bool = True,
) -> Path:
    """Run one ablation shard (one target, all regime/horizon combos) to its own shard CSV."""
    if manifest_path is None:
        manifest_path = PROCESSED_DATA_DIR / "task_manifest.csv"
    if output_path is None:
        ABLATION_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = ABLATION_SHARDS_DIR / f"{shard_idx:04d}__{target}.csv"

    manifest = pd.read_csv(manifest_path)
    sub = manifest[
        (manifest["task_type"] == "regression")
        & (manifest["target"] == target)
    ]
    # Restrict to the canonical ablation grid.
    from scripts.run_ablation_perfold_shard import ABLATION_GRID
    sub = sub[sub.apply(lambda r: (r["regime"], r["horizon"]) in ABLATION_GRID, axis=1)]

    if sub.empty:
        log.warning("Shard %d (target=%s): no matching ablation tuples", shard_idx, target)
        return output_path

    log.info("Shard %d start: target=%s n_tuples=%d", shard_idx, target, len(sub))

    per_fold_path = RESULTS_DIR / "frontier_results_per_fold.csv"
    if not per_fold_path.exists():
        raise FileNotFoundError(
            f"Required full-model per-fold reference not found at {per_fold_path}. "
            "Run step 2 (frontier) first."
        )
    log.info("Loading full-model per-fold reference: %s", per_fold_path)
    per_fold_df = pd.read_csv(per_fold_path,
                              usecols=["task_key", "model", "seed", "fold", "r2"])

    all_rows: List[Dict] = []
    for tuple_idx, (_, row) in enumerate(sub.iterrows(), start=1):
        task_key = row["task_key"]
        full_lookup = _full_r2_lookup(per_fold_df, task_key)
        if not full_lookup:
            log.warning("  no full-model rows for %s; skipping tuple", task_key)
            continue
        log.info("[%d/%d] tuple: %s", tuple_idx, len(sub), task_key)
        rows = run_ablation_perfold_for_task(
            task_key=task_key,
            task_meta=row.to_dict(),
            full_r2_per_fold=full_lookup,
        )
        all_rows.extend(rows)

        if incremental and all_rows:
            pd.DataFrame(all_rows).to_csv(output_path, index=False)
            log.info("  incremental flush: %s (%d rows so far)",
                     output_path.name, len(all_rows))

    pd.DataFrame(all_rows).to_csv(output_path, index=False)
    log.info("Shard %d done: target=%s total_rows=%d -> %s",
             shard_idx, target, len(all_rows), output_path)
    return output_path


# Legacy single-process entry point for non-sharded / interactive use.
def run_ablation_perfold_suite(
    regimes: Optional[List[str]] = None,
    horizons: Optional[List[str]] = None,
    task_keys: Optional[List[str]] = None,
    manifest_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run per-fold ablation across the requested (regime, horizon, task) grid and write one CSV."""
    if manifest_path is None:
        manifest_path = PROCESSED_DATA_DIR / "task_manifest.csv"
    if output_path is None:
        output_path = RESULTS_DIR / "tables" / "ablation_perfold_xgboost.csv"

    manifest = pd.read_csv(manifest_path)
    sub = manifest[manifest["task_type"] == "regression"].copy()
    if regimes is not None:
        sub = sub[sub["regime"].isin(regimes)]
    if horizons is not None:
        sub = sub[sub["horizon"].isin(horizons)]
    if task_keys is not None:
        sub = sub[sub["task_key"].isin(task_keys)]

    log.info("Per-fold ablation suite: %d (task, regime, horizon) tuples", len(sub))

    per_fold_path = RESULTS_DIR / "frontier_results_per_fold.csv"
    if not per_fold_path.exists():
        raise FileNotFoundError(
            f"Required full-model per-fold reference not found at {per_fold_path}. "
            "Run step 2 (frontier) first."
        )
    per_fold_df = pd.read_csv(per_fold_path,
                              usecols=["task_key", "model", "seed", "fold", "r2"])

    all_rows: List[Dict] = []
    for _, row in sub.iterrows():
        task_key = row["task_key"]
        full_lookup = _full_r2_lookup(per_fold_df, task_key)
        if not full_lookup:
            log.warning("No full-model rows for %s; skipping", task_key)
            continue
        log.info("Per-fold ablation: %s", task_key)
        rows = run_ablation_perfold_for_task(
            task_key=task_key,
            task_meta=row.to_dict(),
            full_r2_per_fold=full_lookup,
        )
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    log.info("Saved per-fold ablation: %s (%d rows)", output_path, len(out_df))
    return out_df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--regime", action="append", default=None)
    p.add_argument("--horizon", action="append", default=None)
    p.add_argument("--task-key", action="append", default=None)
    args = p.parse_args()

    run_ablation_perfold_suite(
        regimes=args.regime,
        horizons=args.horizon,
        task_keys=args.task_key,
    )
