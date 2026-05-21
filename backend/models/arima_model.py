"""
ARIMA / SARIMAX model for energy forecasting.
Uses statsmodels under the hood.
"""

import pandas as pd
import numpy as np
from models.base import BasePredictor, PredictionResult
from core.seq_utils import steps_per_day as steps_per_day_fn
from core.train_utils import get_split


class ArimaPredictor(BasePredictor):

    def fit_predict(self, df: pd.DataFrame, horizon_days: int) -> PredictionResult:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        p = self.hyperparams.get("p", 2)
        d = self.hyperparams.get("d", 1)
        q = self.hyperparams.get("q", 2)
        seasonal = self.hyperparams.get("seasonal", False)
        m = self.hyperparams.get("seasonal_m", 24)

        series = df["energy"].dropna()
        freq = pd.infer_freq(series.index) or "h"

        # Detect horizon steps from days + inferred freq
        horizon_steps = horizon_days * steps_per_day_fn(series.to_frame())

        # Use last 80% for training, 20% for in-sample validation
        train_split = float(self.hyperparams.get("train_split", 0.8))
        split = get_split(len(series), train_split)
        train = series.iloc[:split]
        test = series.iloc[split:]

        self.log(f"[ARIMA] Training on {len(train)} steps, validating on {len(test)} steps")
        self.log(f"[ARIMA] Order=({p},{d},{q}), seasonal={seasonal}, m={m}")

        order = (p, d, q)
        seasonal_order = (
            self.hyperparams.get("seasonal_P", 1),
            self.hyperparams.get("seasonal_D", 1),
            self.hyperparams.get("seasonal_Q", 1),
            m,
        ) if seasonal else (0, 0, 0, 0)

        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.log("[ARIMA] Fitting model...")
        fitted = model.fit(disp=False)
        self.log(f"[ARIMA] AIC={fitted.aic:.2f}, BIC={fitted.bic:.2f}")

        # In-sample test predictions
        test_pred = fitted.predict(start=split, end=split + len(test) - 1)
        actual_vals = test.values
        pred_vals = test_pred.values

        mae, rmse, mape = self.calc_metrics(actual_vals, pred_vals)
        self.log(f"[ARIMA] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

        # Future forecast
        self.log(f"[ARIMA] Forecasting {horizon_steps} steps ({horizon_days} days)...")
        forecast_result = fitted.get_forecast(steps=horizon_steps)
        forecast_mean = forecast_result.predicted_mean

        # Build future timestamps
        last_ts = series.index[-1]
        future_idx = pd.date_range(start=last_ts, periods=horizon_steps + 1, freq=freq)[1:]

        return PredictionResult(
            mae=mae,
            rmse=rmse,
            mape=mape,
            forecast=self.series_to_records(future_idx, forecast_mean.values, "predicted"),
            actual=self.series_to_records(test.index, actual_vals, "actual"),
            test_predicted=self.series_to_records(test.index, pred_vals, "predicted"),
            training_history=None,
        )

