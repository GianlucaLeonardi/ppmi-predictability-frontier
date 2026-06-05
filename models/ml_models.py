"""
ML model wrappers for the benchmark. Each model exposes fit/predict, with
hyperparameter tuning via nested inner CV (R^2 for regression, neg log-loss
for classification).
"""

import warnings
from itertools import product
from typing import Any, Dict, Optional

import numpy as np

from configs.config import SEED, N_INNER_FOLDS
from utils.logging_utils import get_logger

log = get_logger(__name__)


# =============================================================================
# Inner CV helper for hyperparameter tuning
# =============================================================================

def _inner_cv_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_class,
    param_grid: Dict[str, list],
    fixed_params: Dict[str, Any],
    scoring: str = "r2",
    n_folds: int = N_INNER_FOLDS,
    sampler_seed: int = SEED,
    use_nan_x: bool = False,
    search: str = "grid",
    n_random_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Hyperparameter search with inner K-fold CV, returning the best param combination."""
    # Generate all parameter combinations
    param_names = sorted(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    combinations = list(product(*param_values))

    if not combinations:
        return dict(fixed_params)

    # Randomized search: sample a subset of combinations with a fixed seed.
    if search == "random" and n_random_samples is not None and n_random_samples < len(combinations):
        sampler = np.random.default_rng(sampler_seed + 10007)
        sampled_idx = sampler.choice(len(combinations), size=n_random_samples, replace=False)
        combinations = [combinations[int(i)] for i in sorted(sampled_idx)]

    # Create inner folds with stratification for both classification and regression.
    rng = np.random.default_rng(SEED)
    n = len(X_train)

    if scoring == "neg_log_loss":
        # Classification: stratify by class label
        strata = y_train.astype(int)
    else:
        # Regression/ranking: stratify by quantile bins
        from configs.config import N_STRATIFICATION_BINS
        finite_mask = np.isfinite(y_train)
        strata = np.zeros(n, dtype=int)
        if finite_mask.sum() > N_STRATIFICATION_BINS:
            quantiles = np.linspace(0, 100, N_STRATIFICATION_BINS + 1)
            bin_edges = np.percentile(y_train[finite_mask], quantiles)
            strata[finite_mask] = np.clip(
                np.digitize(y_train[finite_mask], bin_edges[1:-1]),
                0, N_STRATIFICATION_BINS - 1,
            )

    fold_bins = [[] for _ in range(n_folds)]
    for stratum_val in np.unique(strata):
        stratum_idx = np.where(strata == stratum_val)[0]
        rng.shuffle(stratum_idx)
        for i, idx in enumerate(stratum_idx):
            fold_bins[i % n_folds].append(idx)
    inner_folds = []
    for i in range(n_folds):
        val_idx = np.array(fold_bins[i])
        train_idx = np.concatenate([np.array(fold_bins[j]) for j in range(n_folds) if j != i])
        inner_folds.append((train_idx, val_idx))

    best_score = -np.inf
    best_combo = combinations[0]

    for combo in combinations:
        params = dict(zip(param_names, combo))
        params.update(fixed_params)

        fold_scores = []
        for train_idx, val_idx in inner_folds:
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va, y_va = X_train[val_idx], y_train[val_idx]

            try:
                model = model_class(**params)
                if use_nan_x:
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_va)
                else:
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_va)

                if scoring == "r2":
                    from sklearn.metrics import r2_score
                    score = r2_score(y_va, y_pred)
                elif scoring == "neg_mse":
                    from sklearn.metrics import mean_squared_error
                    score = -mean_squared_error(y_va, y_pred)
                elif scoring == "neg_log_loss":
                    from sklearn.metrics import log_loss
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_va)
                        score = -log_loss(y_va, y_prob)
                    else:
                        score = -np.inf
                else:
                    score = -np.inf

                fold_scores.append(score)
            except Exception:
                fold_scores.append(-np.inf)

        mean_score = np.mean(fold_scores) if fold_scores else -np.inf
        if mean_score > best_score:
            best_score = mean_score
            best_combo = combo

    best_params = dict(zip(param_names, best_combo))
    best_params.update(fixed_params)
    return best_params


# =============================================================================
# Regression models
# =============================================================================

class RidgeModel:
    def __init__(self, param_grid=None, fixed_params=None):
        self.param_grid = param_grid or {"alpha": list(np.logspace(-2, 5, 15))}
        self.fixed_params = fixed_params or {}
        self.model_ = None
        self.best_params_ = {}

    def fit(self, X_train, y_train, **kw):
        from sklearn.linear_model import Ridge

        best = _inner_cv_search(
            X_train, y_train,
            Ridge,
            self.param_grid,
            {**self.fixed_params, "random_state": SEED},
            scoring="r2",
        )
        self.best_params_ = best
        self.model_ = Ridge(**best)
        self.model_.fit(X_train, y_train)

        if best.get("alpha") == self.param_grid["alpha"][-1]:
            log.warning("Ridge: best alpha=%.1f is at grid boundary", best["alpha"])
        return self

    def predict(self, X):
        return self.model_.predict(X)


class ElasticNetModel:
    def __init__(self, param_grid=None, fixed_params=None):
        self.param_grid = param_grid or {
            "alpha": list(np.logspace(-2, 3, 11)),
            "l1_ratio": [0.1, 0.5, 0.9],
        }
        self.fixed_params = fixed_params or {}
        self.model_ = None
        self.best_params_ = {}

    def fit(self, X_train, y_train, **kw):
        from sklearn.linear_model import ElasticNet

        best = _inner_cv_search(
            X_train, y_train,
            ElasticNet,
            self.param_grid,
            {**self.fixed_params, "max_iter": 5000, "random_state": SEED},
            scoring="r2",
        )
        self.best_params_ = best
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = ElasticNet(**best)
            self.model_.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class RandomForestRegressorModel:
    def __init__(self, param_grid=None, fixed_params=None,
                 search="grid", n_random_samples=None, outer_seed=None):
        self.param_grid = param_grid or {
            "n_estimators": [200, 300, 500, 800],
            "max_depth": [8, 12, 16],
            "min_samples_leaf": [3, 5, 10],
        }
        self.fixed_params = fixed_params or {"max_features": "sqrt", "random_state": SEED, "n_jobs": -1}
        self.search = search
        self.n_random_samples = n_random_samples
        self.outer_seed = outer_seed
        self.model_ = None
        self.best_params_ = {}

    def fit(self, X_train, y_train, **kw):
        from sklearn.ensemble import RandomForestRegressor

        best = _inner_cv_search(
            X_train, y_train,
            RandomForestRegressor,
            self.param_grid,
            self.fixed_params,
            scoring="r2",
            sampler_seed=SEED,
            search=self.search,
            n_random_samples=self.n_random_samples,
        )
        self.best_params_ = best
        self.model_ = RandomForestRegressor(**best)
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def feature_importances(self):
        return self.model_.feature_importances_ if self.model_ else np.array([])


class XGBoostRegModel:
    def __init__(self, param_grid=None, fixed_params=None,
                 search="grid", n_random_samples=None, outer_seed=None):
        self.param_grid = param_grid or {
            "n_estimators": [200, 300, 500, 800],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
        }
        self.fixed_params = fixed_params or {"tree_method": "hist", "random_state": SEED}
        self.search = search
        self.n_random_samples = n_random_samples
        self.outer_seed = outer_seed
        self.model_ = None
        self.best_params_ = {}

    def fit(self, X_train, y_train, **kw):
        from xgboost import XGBRegressor

        best = _inner_cv_search(
            X_train, y_train,
            XGBRegressor,
            self.param_grid,
            {**self.fixed_params, "verbosity": 0},
            scoring="r2",
            use_nan_x=True,
            sampler_seed=SEED,
            search=self.search,
            n_random_samples=self.n_random_samples,
        )
        self.best_params_ = best
        self.model_ = XGBRegressor(**best)
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def feature_importances(self):
        return self.model_.feature_importances_ if self.model_ else np.array([])


# =============================================================================
# Classification models
# =============================================================================

class LogisticRegressionModel:
    def __init__(self, param_grid=None, fixed_params=None):
        self.param_grid = param_grid or {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
        self.fixed_params = fixed_params or {}
        self.model_ = None
        self.best_params_ = {}

    def fit(self, X_train, y_train, **kw):
        from sklearn.linear_model import LogisticRegression

        best = _inner_cv_search(
            X_train, y_train,
            LogisticRegression,
            self.param_grid,
            {**self.fixed_params, "max_iter": 5000, "random_state": SEED,
             "solver": "lbfgs"},
            scoring="neg_log_loss",
        )
        self.best_params_ = best
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = LogisticRegression(**best)
            self.model_.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        proba = self.model_.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]  # binary: return P(class=1)
        return proba  # multiclass: return full matrix


class RandomForestClassifierModel:
    def __init__(self, param_grid=None, fixed_params=None,
                 search="grid", n_random_samples=None, outer_seed=None):
        self.param_grid = param_grid or {
            "n_estimators": [200, 300, 500, 800],
            "max_depth": [8, 12, 16],
            "min_samples_leaf": [3, 5, 10],
        }
        self.fixed_params = fixed_params or {
            "class_weight": "balanced", "random_state": SEED, "n_jobs": -1,
        }
        self.search = search
        self.n_random_samples = n_random_samples
        self.outer_seed = outer_seed
        self.model_ = None
        self.best_params_ = {}

    def fit(self, X_train, y_train, **kw):
        from sklearn.ensemble import RandomForestClassifier

        best = _inner_cv_search(
            X_train, y_train,
            RandomForestClassifier,
            self.param_grid,
            self.fixed_params,
            scoring="neg_log_loss",
            sampler_seed=SEED,
            search=self.search,
            n_random_samples=self.n_random_samples,
        )
        self.best_params_ = best
        self.model_ = RandomForestClassifier(**best)
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        proba = self.model_.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def feature_importances(self):
        return self.model_.feature_importances_ if self.model_ else np.array([])


class XGBoostClassifierModel:
    def __init__(self, param_grid=None, fixed_params=None,
                 search="grid", n_random_samples=None, outer_seed=None):
        self.param_grid = param_grid or {
            "n_estimators": [200, 300, 500, 800],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
        }
        self.fixed_params = fixed_params or {
            "scale_pos_weight": "auto",
            "tree_method": "hist", "random_state": SEED,
        }
        self.search = search
        self.n_random_samples = n_random_samples
        self.outer_seed = outer_seed
        self.model_ = None
        self.best_params_ = {}

    def fit(self, X_train, y_train, **kw):
        from xgboost import XGBClassifier

        fp = dict(self.fixed_params)
        # Auto scale_pos_weight for binary
        if fp.get("scale_pos_weight") == "auto":
            n_classes = len(np.unique(y_train[np.isfinite(y_train)]))
            if n_classes == 2:
                n_pos = (y_train == 1).sum()
                n_neg = (y_train == 0).sum()
                fp["scale_pos_weight"] = float(n_neg / n_pos) if n_pos > 0 else 1.0
            else:
                fp.pop("scale_pos_weight")  # not applicable for multiclass

        best = _inner_cv_search(
            X_train, y_train,
            XGBClassifier,
            self.param_grid,
            {**fp, "eval_metric": "logloss", "verbosity": 0},
            scoring="neg_log_loss",
            use_nan_x=True,
            sampler_seed=SEED,
            search=self.search,
            n_random_samples=self.n_random_samples,
        )
        self.best_params_ = best
        self.model_ = XGBClassifier(**best)
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        proba = self.model_.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def feature_importances(self):
        return self.model_.feature_importances_ if self.model_ else np.array([])


# =============================================================================
# Model factory
# =============================================================================

def get_model(name: str, task_type: str = "regression", config: dict = None, seed: int = None):
    """Return a model instance by name, configured from `config` and outer CV `seed`."""
    from models.baselines import PopulationMeanBaseline, LastObservationCarriedForward, MajorityClassBaseline
    config = config or {}
    pg = config.get("param_grid", None)
    fp = config.get("fixed_params", None)
    search = config.get("search", "grid")
    n_random_samples = config.get("n_random_samples", None)
    tree_kw = dict(param_grid=pg, fixed_params=fp,
                   search=search, n_random_samples=n_random_samples,
                   outer_seed=seed)

    # Regression models
    if task_type == "regression":
        if name == "population_mean":
            return PopulationMeanBaseline()
        elif name == "locf":
            return LastObservationCarriedForward()
        elif name == "lme":
            from models.lme_model import LinearMixedEffectsModel
            return LinearMixedEffectsModel()
        elif name == "ridge":
            return RidgeModel(param_grid=pg, fixed_params=fp)
        elif name == "elastic_net":
            return ElasticNetModel(param_grid=pg, fixed_params=fp)
        elif name == "random_forest":
            return RandomForestRegressorModel(**tree_kw)
        elif name == "xgboost":
            return XGBoostRegModel(**tree_kw)

    # Classification models
    elif task_type in ("classification",):
        if name == "majority_class":
            return MajorityClassBaseline()
        elif name == "logistic_regression":
            return LogisticRegressionModel(param_grid=pg, fixed_params=fp)
        elif name == "random_forest_clf":
            return RandomForestClassifierModel(**tree_kw)
        elif name == "xgboost_clf":
            return XGBoostClassifierModel(**tree_kw)

    # Ranking models: reuse regression models (predict continuous, evaluate by rank correlation)
    elif task_type == "ranking":
        if name == "population_mean":
            return PopulationMeanBaseline()
        elif name == "locf":
            return LastObservationCarriedForward()
        elif name == "lme":
            from models.lme_model import LinearMixedEffectsModel
            return LinearMixedEffectsModel()
        elif name == "ridge":
            return RidgeModel(param_grid=pg, fixed_params=fp)
        elif name == "elastic_net":
            return ElasticNetModel(param_grid=pg, fixed_params=fp)
        elif name == "random_forest":
            return RandomForestRegressorModel(**tree_kw)
        elif name == "xgboost":
            return XGBoostRegModel(**tree_kw)

    raise ValueError(f"Unknown model '{name}' for task_type '{task_type}'")
