"""
Naive baseline models.

These are the models that any ML approach must beat to be interesting.
"""

import numpy as np
import pandas as pd


class PopulationMeanBaseline:
    """Predict the training-set mean for every patient."""

    def __init__(self):
        self.mean_ = None

    def fit(self, X, y, **kwargs):
        self.mean_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class MajorityClassBaseline:
    """Predict the majority class for every patient."""

    def __init__(self):
        self.majority_ = None
        self.class_probs_ = None  # class probability vector (binary: scalar; multiclass: matrix)

    def fit(self, X, y, **kwargs):
        valid = y[np.isfinite(y)]
        vals, counts = np.unique(valid, return_counts=True)
        self.majority_ = vals[np.argmax(counts)]
        n_classes = len(vals)
        if n_classes == 2:
            # Binary: store P(class=1)
            self.class_probs_ = float(np.mean(valid == 1))
        else:
            # Multiclass: store per-class prevalence vector
            total = len(valid)
            self.class_probs_ = np.array([np.sum(valid == c) / total for c in sorted(vals)])
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)

    def predict_proba(self, X):
        n = len(X)
        if isinstance(self.class_probs_, np.ndarray):
            # Multiclass: return (n, n_classes) matrix
            return np.tile(self.class_probs_, (n, 1))
        # Binary: return P(class=1) scalar array
        return np.full(n, self.class_probs_)


class LastObservationCarriedForward:
    """LOCF: predict each patient's most recent raw target value, falling back to the training-set mean when missing."""

    def __init__(self):
        self._fallback_mean = None

    def fit(self, y_train, **kwargs):
        self._fallback_mean = float(np.nanmean(y_train))
        return self

    def predict(self, raw_df):
        """Return most-recent raw target value per patient."""
        if raw_df is None or (isinstance(raw_df, pd.DataFrame) and raw_df.empty):
            raise ValueError("LOCF requires raw target values; none provided")

        n = len(raw_df)
        preds = np.full(n, self._fallback_mean)

        # Iterate earliest → latest; later visits overwrite earlier ones.
        # Result: each patient gets their most recent non-NaN value.
        for col in raw_df.columns:
            vals = raw_df[col].values.astype(float)
            valid = np.isfinite(vals)
            preds[valid] = vals[valid]

        return preds
