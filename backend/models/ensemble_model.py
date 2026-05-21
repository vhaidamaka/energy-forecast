"""
Stacking Ensemble predictor.

Does NOT train new neural models — loads predictions from already-finished runs,
fits a Ridge regression meta-learner on top, and blends them.

Workflow:
  1. Load test_predicted_json from each base_run_id
  2. Align all predictions to the same timestamps (inner join)
  3. Fit Ridge(alpha) on stacked test predictions → actual values
  4. Blend forecasts using the same learned coefficients
  5. Evaluate blended predictions vs actual

HyperParams (all optional):
  base_run_ids : list[int]   — run IDs to ensemble (required, set at runtime)
  alpha        : float       — Ridge regularisation strength (default 1.0)
  blend_method : str         — "ridge" | "mean" | "weighted_mape"
                               ridge         — learned weights (default)
                               mean          — simple average
                               weighted_mape — weight = 1 / MAPE of each model
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from models.base import BasePredictor, PredictionResult
from core.seq_utils import steps_per_day as steps_per_day_fn


class EnsemblePredictor(BasePredictor):

    def fit_predict(self, df: pd.DataFrame, horizon_days: int) -> PredictionResult:
        """
        df is passed for compatibility but is not used for training.
        Actual data is loaded from the DB via base_run_ids.
        """
        from core.database import SessionLocal, Run, Result as DBResult

        run_ids   = self.hyperparams.get("base_run_ids", [])
        alpha     = float(self.hyperparams.get("alpha", 1.0))
        method    = self.hyperparams.get("blend_method", "ridge")

        if not run_ids:
            raise ValueError("ensemble requires 'base_run_ids' hyperparameter")

        self.log(f"[Ensemble] Loading {len(run_ids)} base runs: {run_ids}")
        self.log(f"[Ensemble] Method={method}  alpha={alpha}")

        db = SessionLocal()
        try:
            runs_data = []
            for rid in run_ids:
                run = db.query(Run).filter(Run.id == rid).first()
                if not run or not run.result:
                    raise ValueError(f"Run #{rid} not found or has no result")
                r = run.result
                if not r.test_predicted_json or not r.actual_json or not r.forecast_json:
                    raise ValueError(
                        f"Run #{rid} ({run.model}) is missing test_predicted_json or forecast_json. "
                        "Re-run it to generate these fields."
                    )
                runs_data.append({
                    "id":        rid,
                    "model":     run.model,
                    "mape":      r.mape,
                    "test_pred": r.test_predicted_json,
                    "actual":    r.actual_json,
                    "forecast":  r.forecast_json,
                })
                self.log(f"[Ensemble]  ✓ Run #{rid} ({run.model.upper()}) MAPE={r.mape:.2f}%")
        finally:
            db.close()

        # ── 1. Build test-set stacked matrix ─────────────────────────────────
        # Align by timestamp (inner join across all models)
        test_dfs = []
        for rd in runs_data:
            pred_map = {d["timestamp"]: d["predicted"] for d in rd["test_pred"]}
            test_dfs.append(pd.Series(pred_map, name=f"run_{rd['id']}"))

        stacked_test = pd.concat(test_dfs, axis=1).dropna()
        self.log(f"[Ensemble] Aligned test set: {len(stacked_test)} points × {len(runs_data)} models")

        # Actual values aligned to same timestamps
        actual_map = {d["timestamp"]: d["actual"] for d in runs_data[0]["actual"]}
        y_actual = np.array([actual_map[ts] for ts in stacked_test.index if ts in actual_map])
        X_test   = stacked_test.values[:len(y_actual)]

        if len(y_actual) == 0:
            raise ValueError("No overlapping timestamps found across base runs")

        # ── 2. Fit meta-learner ───────────────────────────────────────────────
        if method == "mean":
            weights = np.ones(len(runs_data)) / len(runs_data)
            blend_test = X_test @ weights
            self.log(f"[Ensemble] Simple mean blend")

        elif method == "weighted_mape":
            mapes   = np.array([rd["mape"] for rd in runs_data])
            inv     = 1.0 / np.maximum(mapes, 0.1)
            weights = inv / inv.sum()
            blend_test = X_test @ weights
            for rd, w in zip(runs_data, weights):
                self.log(f"[Ensemble]   {rd['model']:15s} weight={w:.3f}")

        else:  # ridge (default)
            scaler  = StandardScaler()
            X_scaled = scaler.fit_transform(X_test)
            ridge   = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_scaled, y_actual)
            weights  = ridge.coef_
            blend_test = ridge.predict(X_scaled)
            for rd, w in zip(runs_data, weights):
                self.log(f"[Ensemble]   {rd['model']:15s} coef={w:.4f}")
            self.log(f"[Ensemble]   intercept={ridge.intercept_:.4f}")

        # ── 3. Metrics ────────────────────────────────────────────────────────
        mae, rmse, mape = self.calc_metrics(y_actual, blend_test)
        self.log(f"[Ensemble] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

        # Improvement vs best base model
        best_base_mape = min(rd["mape"] for rd in runs_data)
        delta = best_base_mape - mape
        self.log(f"[Ensemble] Best base MAPE={best_base_mape:.2f}%  Δ={delta:+.2f}%")

        # ── 4. Blend forecasts ────────────────────────────────────────────────
        forecast_dfs = []
        for rd in runs_data:
            fmap = {d["timestamp"]: d["predicted"] for d in rd["forecast"]}
            forecast_dfs.append(pd.Series(fmap, name=f"run_{rd['id']}"))

        stacked_fc = pd.concat(forecast_dfs, axis=1).dropna()
        X_fc = stacked_fc.values

        if method == "ridge":
            X_fc_scaled  = scaler.transform(X_fc)
            blend_fc     = ridge.predict(X_fc_scaled)
        else:
            blend_fc     = X_fc @ weights[:len(runs_data)]

        blend_fc = np.maximum(blend_fc, 0)   # clip negatives

        # ── 5. Build timestamps ───────────────────────────────────────────────
        test_timestamps = pd.DatetimeIndex(stacked_test.index[:len(y_actual)])
        fc_timestamps   = pd.DatetimeIndex(stacked_fc.index)

        # Training history: show per-model MAPE as a comparison bar
        history = [
            {"epoch": i+1, "model": rd["model"], "mape": rd["mape"], "loss": rd["mape"]/100}
            for i, rd in enumerate(sorted(runs_data, key=lambda x: x["mape"]))
        ]

        return PredictionResult(
            mae=mae, rmse=rmse, mape=mape,
            forecast=self.series_to_records(fc_timestamps, blend_fc, "predicted"),
            actual=self.series_to_records(test_timestamps, y_actual, "actual"),
            test_predicted=self.series_to_records(test_timestamps, blend_test, "predicted"),
            training_history=history,
        )
