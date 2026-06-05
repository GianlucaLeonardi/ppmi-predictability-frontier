"""
Central configuration for the PPMI Predictability-Frontier Benchmark: the single
source of truth for cohort, targets, horizons, regimes, modalities, models, and
validation settings used by all downstream modules.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# -- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_COHERENCE_ROOT = PROJECT_ROOT.parent.parent  # longitudinal_coherence/
RAW_DATA_DIR = _COHERENCE_ROOT / "data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
EXISTING_PROCESSED = RAW_DATA_DIR / "processed_data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
REPORTS_DIR = RESULTS_DIR / "reports"
EXPORTS_DIR = RESULTS_DIR / "exports"
LOGS_DIR = RESULTS_DIR / "logs"
EXPLORATORY_DIR = RESULTS_DIR / "exploratory"

# -- Random seed --------------------------------------------------------------

SEED = 42

# -- Cross-validation settings ------------------------------------------------
# Patient-level stratified K-fold outer CV (no held-out val set); HP tuning uses
# inner CV within each training fold. Setup: 5 outer x 5 seeds x 3 inner x 15.

N_OUTER_FOLDS = 5
N_SEEDS = 5
CV_SEEDS = [0, 42, 123, 2024, 7]

# Inner CV for hyperparameter tuning (nested inside each outer fold)
N_INNER_FOLDS = 3

# Randomized hyperparameter search: configurations sampled from the grid for
# tree-based models (RF, XGBoost). Linear models keep exhaustive grid search.
N_RANDOM_SEARCH_SAMPLES = 15

# -- PPMI visit schedule  ----

VISIT_MONTH_MAP = {
    "SC":  -1,
    "BL":   0,
    "V01":  3,
    "V02":  6,
    "V03":  9,
    "V04": 12,
    "V05": 18,
    "V06": 24,
    "V07": 30,
    "V08": 36,
    "V09": 42,
    "V10": 48,
    "V11": 54,
    "V12": 60,
}

# Standard analysis visits (available for most instruments)
VISIT_ORDER = ["BL", "V02", "V04", "V06", "V08", "V10", "V12"]

VISIT_SCHEDULE = {v: VISIT_MONTH_MAP[v] for v in VISIT_ORDER}

# -- Cohort -------------------------------------------------------------------

COHORT_LABEL_PD = "Parkinson's Disease"
COHORT_PRIMARY = [COHORT_LABEL_PD]

# -- Target outcomes ----------------------------------------------------------

@dataclass
class TargetSpec:
    name: str               # short identifier
    display: str            # human-readable label for plots
    domain: str             # motor | cognitive | nonmotor | autonomic | milestone
    source_table: str       # longitudinal modality key
    column: str             # column in processed data
    task_type: str          # regression | classification
    direction: str          # higher_worse | lower_worse
    min_patients: int = 60  # per-task retention threshold: task kept iff
                            # full-cohort n_total >= 2 * min_patients. With
                            # stratified K-fold at N_OUTER_FOLDS=5, this
                            # means each test fold holds ~(2*min_patients/5)
                            # patients. (See build_dataset.run_pipeline.)
    primary_metric: str = ""  # primary metric for this task type


# Regression targets
TARGETS_REGRESSION = [
    # Motor
    TargetSpec("updrs3_total",   "UPDRS-III Total",       "motor",     "updrs3", "NP3TOT",
               "regression", "higher_worse", primary_metric="r2"),
    TargetSpec("updrs3_tremor",  "UPDRS-III Tremor",      "motor",     "updrs3", "TREMOR_SUBSCORE",
               "regression", "higher_worse", primary_metric="r2"),
    TargetSpec("updrs3_brady",   "UPDRS-III Bradykinesia","motor",     "updrs3", "BRADY_SUBSCORE",
               "regression", "higher_worse", primary_metric="r2"),
    TargetSpec("updrs3_pigd",    "UPDRS-III PIGD",        "motor",     "updrs3", "PIGD_SUBSCORE",
               "regression", "higher_worse", primary_metric="r2"),
    # Cognitive
    TargetSpec("moca_total",     "MoCA Total",            "cognitive", "moca",   "MCATOT",
               "regression", "lower_worse", primary_metric="r2"),
    TargetSpec("moca_delayed",   "MoCA Delayed Recall",   "cognitive", "moca",   "DELAYED_RECALL_SUM",
               "regression", "lower_worse", primary_metric="r2"),
    # Non-motor
    TargetSpec("updrs1_total",   "UPDRS-I Total",         "nonmotor",  "updrs1", "NP1RTOT",
               "regression", "higher_worse", primary_metric="r2"),
    # Autonomic
    TargetSpec("ortho_sys",      "Orthostatic SBP Drop",  "autonomic", "vital_signs", "ORTHO_SYS_DROP",
               "regression", "higher_worse", primary_metric="r2"),
    TargetSpec("ortho_dia",      "Orthostatic DBP Drop",  "autonomic", "vital_signs", "ORTHO_DIA_DROP",
               "regression", "higher_worse", primary_metric="r2"),
]

# Classification targets

TARGETS_CLASSIFICATION = [
    TargetSpec("motor_worsen",    "Motor Worsening",       "milestone", "updrs3", "MOTOR_WORSEN",
               "classification", "higher_worse", primary_metric="auprc"),
    TargetSpec("cognitive_impair","Cognitive Impairment",   "milestone", "moca",   "MCI_FLAG",
               "classification", "lower_worse",  primary_metric="auprc"),
]

# Ranking target: patients ranked by magnitude of motor change from baseline
TARGETS_RANKING = [
    TargetSpec("motor_rank",      "Motor Change Rank",     "motor",     "updrs3", "MOTOR_RANK",
               "ranking", "higher_worse", primary_metric="spearman"),
]

TARGETS = TARGETS_REGRESSION + TARGETS_CLASSIFICATION + TARGETS_RANKING

TARGET_DOMAINS = ["motor", "cognitive", "nonmotor", "autonomic", "milestone"]

# -- Worsening thresholds (clinically meaningful change) ----------------------
MOTOR_WORSEN_THRESHOLD = 5     # MDS-UPDRS III increase of >= 5 points
MOCA_IMPAIRMENT_CUTOFF = 26    # MoCA < 26

# -- Prediction horizons -----------------------------------------------------

FORECAST_HORIZONS_PRIMARY = ["V04", "V06", "V08"]       # 12m, 24m, 36m
FORECAST_HORIZONS_SENSITIVITY = ["V10"]                  # 48m (V12 excluded: OFF-state UPDRS-III test n < 60)
FORECAST_HORIZONS = FORECAST_HORIZONS_PRIMARY + FORECAST_HORIZONS_SENSITIVITY

# -- Information regimes ------------------------------------------------------

@dataclass
class RegimeSpec:
    name: str
    display: str
    history_visits: Optional[List[str]]   # None = dynamic rolling

REGIMES = [
    RegimeSpec("baseline_only",       "Baseline only",               ["BL"]),
    RegimeSpec("baseline_multimodal", "Baseline + static multimodal",["BL"]),
    RegimeSpec("baseline_plus_12m",   "Baseline + 12 months",        ["BL", "V02", "V04"]),
    RegimeSpec("rolling",             "Rolling history",             None),  # all visits < horizon
]

# -- Feature modality families ------------------------------------------------

@dataclass
class ModalityFamily:
    name: str
    display: str
    kind: str           # static | longitudinal
    source_key: str
    is_default: bool    # included in default feature set

MODALITY_FAMILIES = [
    ModalityFamily("demographics",       "Demographics",         "static",       "demographics",        True),
    ModalityFamily("participant_status",  "Participant Status",   "static",       "participant_status",  True),
    ModalityFamily("genetic_consensus",   "Genetic Variants",     "static",       "genetic_consensus",   True),
    ModalityFamily("genetic_prs",         "Genetic PRS",          "static",       "genetic_prs",         False),
    ModalityFamily("csf_biomarkers",      "CSF Biomarkers",       "static",       "csf_biomarkers",      False),
    ModalityFamily("plasma_biomarkers",   "Plasma Biomarkers",    "static",       "plasma_biomarkers",   False),
    ModalityFamily("updrs3",              "UPDRS-III (Motor)",    "longitudinal", "updrs3",              True),
    ModalityFamily("updrs1",              "UPDRS-I (Non-motor)",  "longitudinal", "updrs1",              True),
    ModalityFamily("moca",                "MoCA (Cognitive)",     "longitudinal", "moca",                True),
    ModalityFamily("vital_signs",         "Vital Signs",          "longitudinal", "vital_signs",         True),
    ModalityFamily("blood_chemistry",     "Blood Chemistry",      "longitudinal", "blood_chemistry",     False),
    ModalityFamily("ledd",                "LEDD (Medication)",    "longitudinal", "ledd",                False),
]

DEFAULT_MODALITIES = [m.name for m in MODALITY_FAMILIES if m.is_default]

# -- Derived sub-scores from UPDRS Part III examiner-rated items only.

UPDRS3_TREMOR_ITEMS = [
    "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML",
    "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL",
    "NP3RTALJ", "NP3RTCON",
]

UPDRS3_PIGD_ITEMS = [
    "NP3GAIT", "NP3FRZGT", "NP3PSTBL",
]

UPDRS3_BRADY_ITEMS = [
    "NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL",
    "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL",
    "NP3LGAGR", "NP3LGAGL", "NP3BRADY",
]

# -- Models -------------------------------------------------------------------
# Hyperparameter grids are defined here for nested CV tuning.
# The inner loop selects the best hyperparameters; the outer loop evaluates.

MODELS_REGRESSION = {
    "population_mean": {"type": "naive", "params": {}},
    "locf":            {"type": "naive", "params": {}},
    "lme":             {"type": "statistical", "params": {}},
    "ridge":           {"type": "linear",
                        "param_grid": {"alpha": list(__import__('numpy').logspace(-2, 5, 15))}},
    "elastic_net":     {"type": "linear",
                        "param_grid": {"alpha": list(__import__('numpy').logspace(-2, 3, 11)),
                                       "l1_ratio": [0.1, 0.5, 0.9, 0.95, 1.0]}},
    "random_forest":   {"type": "tree",
                        "search": "random",
                        "n_random_samples": N_RANDOM_SEARCH_SAMPLES,
                        "param_grid": {"n_estimators": [200, 500, 800, 1000],
                                       "max_depth": [8, 12, 16, None],
                                       "min_samples_leaf": [1, 3, 5, 10]},
                        "fixed_params": {"max_features": "sqrt", "random_state": SEED, "n_jobs": -1}},
    "xgboost":         {"type": "tree",
                        "search": "random",
                        "n_random_samples": N_RANDOM_SEARCH_SAMPLES,
                        "param_grid": {"n_estimators": [200, 500, 800, 1000],
                                       "max_depth": [3, 4, 6, 8],
                                       "learning_rate": [0.005, 0.01, 0.05, 0.1]},
                        # XGBoost runs on GPU (device="cuda", tree_method="hist").
                        "fixed_params": {
                            "tree_method": "hist",
                            "device": "cuda",
                            "random_state": SEED,
                            "subsample": 0.8, "colsample_bytree": 0.8}},
}

MODELS_CLASSIFICATION = {
    "majority_class":       {"type": "naive", "params": {}},
    "logistic_regression":  {"type": "linear",
                             "param_grid": {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}},
    "random_forest_clf":    {"type": "tree",
                             "search": "random",
                             "n_random_samples": N_RANDOM_SEARCH_SAMPLES,
                             "param_grid": {"n_estimators": [200, 500, 800, 1000],
                                            "max_depth": [8, 12, 16, None],
                                            "min_samples_leaf": [1, 3, 5, 10]},
                             "fixed_params": {"class_weight": "balanced",
                                              "random_state": SEED, "n_jobs": -1}},
    "xgboost_clf":          {"type": "tree",
                             "search": "random",
                             "n_random_samples": N_RANDOM_SEARCH_SAMPLES,
                             "param_grid": {"n_estimators": [200, 500, 800, 1000],
                                            "max_depth": [3, 4, 6, 8],
                                            "learning_rate": [0.005, 0.01, 0.05, 0.1]},
                             # XGBoost runs on GPU (device="cuda", tree_method="hist").
                             "fixed_params": {"scale_pos_weight": "auto",
                                              "tree_method": "hist",
                                              "device": "cuda",
                                              "random_state": SEED,
                                              "subsample": 0.8, "colsample_bytree": 0.8}},
}

# -- Validation / metrics -----------------------------------------------------
# R^2 is the PRIMARY regression metric. Spearman is the main secondary.

METRICS_REGRESSION = ["r2", "spearman", "mae", "rmse", "pearson"]
PRIMARY_METRIC_REGRESSION = "r2"
SECONDARY_METRIC_REGRESSION = "spearman"

METRICS_CLASSIFICATION = [
    "auroc", "auprc", "f1", "mcc", "balanced_accuracy",
    "sensitivity", "specificity", "ppv", "npv", "brier",
]

# R^2/Spearman divergence detection threshold:
# Flag tasks where two models have |delta R^2| < this but |delta Spearman| > 0.05
R2_SPEARMAN_DIVERGENCE_R2_THRESH = 0.02
R2_SPEARMAN_DIVERGENCE_SPEARMAN_THRESH = 0.05

# Bootstrap settings (not used in main pipeline — CIs come from CV distribution).
# Retained for ad-hoc analysis and statistical test bootstrapping.
N_BOOTSTRAP = 2000
BOOTSTRAP_CI = 0.95
N_PERMUTATIONS = 5000
ALPHA_STATISTICAL = 0.05

# -- Preprocessing ------------------------------------------------------------

WINSOR_QUANTILES = (0.005, 0.995)
Z_CLIP = 8.0
MIN_NON_NAN_FRAC = 0.10   # drop columns with >90% missing

# Number of stratification bins for continuous regression targets in CV
N_STRATIFICATION_BINS = 5

# -- QC thresholds ------------------------------------------------------------

MIN_N_WARNING = 50
MIN_N_EXCLUDE = 30

# -- Plotting -----------------------------------------------------------------

FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"

PALETTE = {
    # Okabe-Ito colour-blind-safe categorical palette (coherent across all figures)
    "motor":     "#D55E00",
    "cognitive": "#0072B2",
    "nonmotor":  "#009E73",
    "autonomic": "#CC79A7",
    "milestone": "#999999",
    "biomarker": "#56B4E9",
}

REGIME_MARKERS = {
    "baseline_only":       "o",
    "baseline_multimodal": "^",
    "baseline_plus_12m":   "s",
    "rolling":             "D",
}

REGIME_COLORS = {
    # Okabe-Ito colour-blind-safe. The two high-information regimes (+12m, rolling)
    # get the maximally-distinct orange/blue pair so their traces and significance
    # asterisks never read as similar; baseline_only=grey, multimodal=reddish-purple.
    "baseline_only":       "#999999",
    "baseline_multimodal": "#CC79A7",
    "baseline_plus_12m":   "#E69F00",
    "rolling":             "#0072B2",
}
