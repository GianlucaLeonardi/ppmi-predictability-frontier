"""
Linear mixed-effects model for longitudinal prediction on raw-scale targets,
predicting at the forecast horizon via BLUPs, with LOCF fallback.
"""

import warnings
import numpy as np
import pandas as pd

from configs.config import VISIT_SCHEDULE
from utils.logging_utils import get_logger

log = get_logger(__name__)

try:
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    log.warning("statsmodels not available; LME model will fall back to population mean")


class LinearMixedEffectsModel:
    """LME for longitudinal prediction on raw-scale targets, with LOCF fallback
    when fewer than 2 history visits are available."""

    def __init__(self):
        self._fe_intercept = 0.0
        self._fe_slope = 0.0
        self._re_cov = None
        self._resid_var = 1.0
        self._horizon_months = None
        self._fitted = False
        self._fallback_pred = 0.0
        # Raw-column metadata (for BLUP prediction on test patients)
        self._raw_col_visits = []   # list of (col_name, months) pairs

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, feature_names=None, target_column="",
            horizon_months=None, X_train_raw=None,
            raw_target_train=None, **kwargs):
        """Fit the LME model on raw-scale target history, falling back to LOCF."""
        self._horizon_months = horizon_months or 36.0
        self._fallback_pred = float(np.nanmean(y_train))

        # Parse raw column metadata
        self._raw_col_visits = []
        if raw_target_train is not None and not raw_target_train.empty:
            for col in raw_target_train.columns:
                parts = col.split("__")
                # format: __raw__{visit}__{target_col}
                if len(parts) >= 4 and parts[1] == "raw":
                    visit = parts[2]
                    if visit in VISIT_SCHEDULE:
                        self._raw_col_visits.append(
                            (col, float(VISIT_SCHEDULE[visit]))
                        )
            self._raw_col_visits.sort(key=lambda x: x[1])

        has_raw = len(self._raw_col_visits) >= 2

        if not has_raw or not HAS_STATSMODELS:
            self._fitted = False
            self._fit_locf_fallback(raw_target_train, y_train)
            return

        # Build long-format data from raw target values (correct common scale)
        long_records = []
        raw_vals = raw_target_train.values  # (n_patients, n_visits)
        raw_cols = raw_target_train.columns.tolist()
        col_months = {col: months for col, months in self._raw_col_visits}

        for i in range(len(raw_vals)):
            for j, col in enumerate(raw_cols):
                if col in col_months:
                    val = raw_vals[i, j]
                    if np.isfinite(val):
                        long_records.append({
                            "patient": i,
                            "months": col_months[col],
                            "value": float(val),
                        })

        if len(long_records) < 20:
            self._fitted = False
            self._fit_locf_fallback(raw_target_train, y_train)
            return

        long_df = pd.DataFrame(long_records)
        n_patients_with_data = long_df["patient"].nunique()
        if n_patients_with_data < 10:
            self._fitted = False
            self._fit_locf_fallback(raw_target_train, y_train)
            return

        # Fit MixedLM on raw-scale values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exog = long_df[["months"]].copy()
                exog.insert(0, "Intercept", 1.0)
                md = MixedLM(
                    endog=long_df["value"],
                    exog=exog,
                    groups=long_df["patient"],
                    exog_re=exog[["Intercept", "months"]],
                )
                result = md.fit(reml=True, method="lbfgs", maxiter=300)

            if not result.converged:
                log.warning("LME did not converge in %d iterations; "
                            "estimates may be unreliable", 300)

            self._fe_intercept = float(result.fe_params["Intercept"])
            self._fe_slope = float(result.fe_params["months"])
            self._re_cov = np.array(result.cov_re)
            self._resid_var = float(result.scale)
            self._fitted = True
            log.debug("LME fitted (raw scale): intercept=%.4f, slope=%.6f, "
                      "n_obs=%d, n_groups=%d, converged=%s",
                      self._fe_intercept, self._fe_slope,
                      len(long_df), n_patients_with_data, result.converged)
        except Exception as e:
            log.debug("LME fitting failed (%s); using LOCF fallback", e)
            self._fitted = False
            self._fit_locf_fallback(raw_target_train, y_train)

    # ------------------------------------------------------------------
    def predict(self, X_test, X_test_raw=None, raw_target_test=None):
        """Predict at the horizon using BLUPs, or the LOCF fallback if unfitted."""
        if not self._fitted:
            return self._predict_fallback(raw_target_test)

        return self._predict_raw(raw_target_test)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_raw(self, raw_target_df):
        """Predict raw-scale value at horizon for each patient using BLUPs."""
        n = len(raw_target_df) if raw_target_df is not None else 0
        pop_pred = self._fe_intercept + self._fe_slope * self._horizon_months
        preds = np.full(max(n, 1), pop_pred)

        if n == 0 or self._re_cov is None:
            return preds[:n] if n > 0 else preds

        D = self._re_cov
        sigma2 = max(self._resid_var, 1e-6)

        raw_vals = raw_target_df.values
        raw_cols = raw_target_df.columns.tolist()
        col_months = {col: months for col, months in self._raw_col_visits}

        for i in range(n):
            obs_months = []
            obs_values = []
            for j, col in enumerate(raw_cols):
                if col in col_months:
                    val = raw_vals[i, j]
                    if np.isfinite(val):
                        obs_months.append(col_months[col])
                        obs_values.append(float(val))

            if len(obs_months) == 0:
                continue

            obs_months_arr = np.array(obs_months)
            obs_values_arr = np.array(obs_values)
            n_obs = len(obs_months_arr)

            # Random effects design: [1, months]
            Z = np.column_stack([np.ones(n_obs), obs_months_arr])

            # Residuals from fixed effects
            fixed = self._fe_intercept + self._fe_slope * obs_months_arr
            resid = obs_values_arr - fixed

            # BLUP: b_hat = D Z' (Z D Z' + sigma2 I)^{-1} resid
            try:
                V = Z @ D @ Z.T + sigma2 * np.eye(n_obs)
                V_inv = np.linalg.solve(V, np.eye(n_obs))
                b_hat = D @ Z.T @ V_inv @ resid

                # Predict at horizon
                z_h = np.array([1.0, self._horizon_months])
                preds[i] = pop_pred + z_h @ b_hat
            except np.linalg.LinAlgError:
                pass

        return preds

    # ------------------------------------------------------------------
    # LOCF fallback (used when LME cannot be fitted)
    # ------------------------------------------------------------------

    def _fit_locf_fallback(self, raw_target_df, y_train):
        """Store training-set mean; prediction uses raw carry-forward."""
        self._fallback_pred = float(np.nanmean(y_train))

    def _predict_fallback(self, raw_target_df):
        """LOCF-style fallback: most recent raw value, or training-set mean."""
        if raw_target_df is None:
            raise ValueError("LME fallback requires raw_target_df (cannot determine n_test)")
        n = len(raw_target_df)
        preds = np.full(n, self._fallback_pred)
        if not raw_target_df.empty:
            for col in raw_target_df.columns:
                vals = raw_target_df[col].values.astype(float)
                valid = np.isfinite(vals)
                preds[valid] = vals[valid]
        return preds
