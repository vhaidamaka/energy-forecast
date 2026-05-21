"""
Vanilla LSTM model for energy forecasting.
"""

import numpy as np
import pandas as pd
from models.base import BasePredictor, PredictionResult
from core.seq_utils import build_sequences, scale_data, inverse_scale, steps_per_day as steps_per_day_fn
from core.train_utils import get_split, build_callbacks, build_optimizer


class LstmPredictor(BasePredictor):

    def _build_model(self, input_shape):
        import keras
        from keras import layers

        u1    = self.hyperparams.get("units_1", 64)
        u2    = self.hyperparams.get("units_2", 32)
        dr    = self.hyperparams.get("dropout", 0.2)
        act   = self.hyperparams.get("dense_activation", "relu")
        dense = self.hyperparams.get("dense_units", 16)

        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(u1, return_sequences=True),
            layers.Dropout(dr),
            layers.LSTM(u2),
            layers.Dropout(dr),
            layers.Dense(dense, activation=act),
            layers.Dense(1),
        ])
        model.compile(optimizer=build_optimizer(self.hyperparams), loss="mse", metrics=["mae"])
        return model

    def fit_predict(self, df: pd.DataFrame, horizon_days: int) -> PredictionResult:
        import tensorflow as tf
        tf.random.set_seed(42)
        np.random.seed(42)

        lookback    = self.hyperparams.get("lookback", 24)
        epochs      = self.hyperparams.get("epochs", 50)
        batch       = self.hyperparams.get("batch_size", 32)
        val_split   = float(self.hyperparams.get("validation_split", 0.1))
        train_split = float(self.hyperparams.get("train_split", 0.8))

        feature_cols = _get_feature_cols(df)
        self.log(f"[LSTM] Features: {feature_cols}")
        self.log(f"[LSTM] Lookback={lookback}  Epochs={epochs}  Batch={batch}  "
                 f"TrainSplit={train_split}  ValSplit={val_split}")

        X_s, y_s, scaler_X, scaler_y = scale_data(df, feature_cols)
        X_seq, y_seq = build_sequences(X_s, y_s, lookback)

        split = get_split(len(X_seq), train_split)
        X_tr, X_te = X_seq[:split], X_seq[split:]
        y_tr, y_te = y_seq[:split], y_seq[split:]

        model = self._build_model((X_tr.shape[1], X_tr.shape[2]))
        self.log(f"[LSTM] Params: {model.count_params():,}")

        history_records = []
        cbs = build_callbacks(self.hyperparams, self.log, history_records)

        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch,
                  validation_split=val_split, callbacks=cbs, verbose=0)

        self.log(f"[LSTM] Trained {len(history_records)} epochs")

        y_pred_s = model.predict(X_te, verbose=0)
        y_actual = inverse_scale(scaler_y, y_te)
        y_pred   = inverse_scale(scaler_y, y_pred_s)

        mae, rmse, mape = self.calc_metrics(y_actual.flatten(), y_pred.flatten())
        self.log(f"[LSTM] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

        test_timestamps = df.index[split + lookback: split + lookback + len(y_te)]

        horizon_steps = horizon_days * steps_per_day_fn(df)
        self.log(f"[LSTM] Forecasting {horizon_steps} steps...")

        future_preds = _recursive_forecast(
            model, X_seq[-1], scaler_X, scaler_y, df, feature_cols, lookback, horizon_steps
        )

        freq = pd.infer_freq(df.index) or "h"
        future_idx = pd.date_range(start=df.index[-1], periods=horizon_steps + 1, freq=freq)[1:]

        return PredictionResult(
            mae=mae, rmse=rmse, mape=mape,
            forecast=self.series_to_records(future_idx, future_preds, "predicted"),
            actual=self.series_to_records(test_timestamps, y_actual.flatten(), "actual"),
            test_predicted=self.series_to_records(test_timestamps, y_pred.flatten(), "predicted"),
            training_history=history_records,
        )


# ── Shared helpers ────────────────────────────────────────────────────────────

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    preferred = ["energy", "voltage", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                 "is_weekend", "month"]
    return [c for c in preferred if c in df.columns]


def _recursive_forecast(model, last_seq, scaler_X, scaler_y, df, feature_cols, lookback, steps):
    """
    Recursive multi-step forecast with correct time feature advancement.

    At each step we:
      1. Predict from the current scaled sequence
      2. Inverse-scale → real energy value
      3. Compute the ACTUAL timestamp for this future step
      4. Recompute all cyclical time features (hour_sin/cos, dow_sin/cos,
         is_weekend, month) for that future timestamp
      5. Re-scale the updated feature row and slide the window

    This fixes the flat-line problem caused by frozen time features:
    without updating them, the model always sees the same hour/day context
    and converges to the training mean within 1–2 days.
    """
    import pandas as pd

    preds = []
    seq = last_seq.copy()
    energy_idx   = feature_cols.index("energy")   if "energy"   in feature_cols else 0
    freq         = pd.infer_freq(df.index) or "h"
    last_ts      = df.index[-1]

    # Build a lookup: feature_col → index in feature_cols
    col_idx = {c: i for i, c in enumerate(feature_cols)}

    for step in range(steps):
        inp    = seq.reshape(1, lookback, seq.shape[-1])
        pred_s = model.predict(inp, verbose=0)
        pred_val = float(scaler_y.inverse_transform(pred_s)[0, 0])
        preds.append(pred_val)

        # ── Next timestamp ────────────────────────────────────────────────────
        future_ts = last_ts + (step + 1) * pd.tseries.frequencies.to_offset(freq)

        # ── Build next feature row in ORIGINAL (unscaled) space ──────────────
        # Start from the last known unscaled row so stable features (voltage)
        # are preserved, then overwrite with correct values.
        last_unscaled = scaler_X.inverse_transform(seq[-1:]).flatten()
        next_unscaled = last_unscaled.copy()

        # Update energy with the new prediction
        next_unscaled[energy_idx] = pred_val

        # Update all time features we know how to compute
        _update_time_features(next_unscaled, col_idx, future_ts)

        # Re-scale the whole row → back into model input space
        next_row = scaler_X.transform(next_unscaled.reshape(1, -1)).flatten()
        seq = np.vstack([seq[1:], next_row])

    return np.array(preds)


def _update_time_features(row: np.ndarray, col_idx: dict, ts) -> None:
    """
    Update cyclical time features in-place in unscaled feature row.
    Only updates columns that actually exist in the feature set.
    """
    hour = ts.hour
    dow  = ts.dayofweek   # 0=Monday … 6=Sunday

    if "hour_sin"   in col_idx: row[col_idx["hour_sin"]]   = np.sin(2 * np.pi * hour / 24)
    if "hour_cos"   in col_idx: row[col_idx["hour_cos"]]   = np.cos(2 * np.pi * hour / 24)
    if "dow_sin"    in col_idx: row[col_idx["dow_sin"]]    = np.sin(2 * np.pi * dow  / 7)
    if "dow_cos"    in col_idx: row[col_idx["dow_cos"]]    = np.cos(2 * np.pi * dow  / 7)
    if "is_weekend" in col_idx: row[col_idx["is_weekend"]] = float(dow >= 5)
    if "month"      in col_idx: row[col_idx["month"]]      = float(ts.month)
    if "hour"       in col_idx: row[col_idx["hour"]]       = float(hour)
    if "day_of_week" in col_idx: row[col_idx["day_of_week"]] = float(dow)
    if "day"        in col_idx: row[col_idx["day"]]        = float(ts.day)



class _LoggingCallback:
    def __init__(self, log_fn, records: list):
        from tensorflow.keras.callbacks import Callback

        class _CB(Callback):
            def on_epoch_end(self_, epoch, logs=None):
                logs = logs or {}
                loss     = logs.get("loss", 0)
                val_loss = logs.get("val_loss", 0)
                mae      = logs.get("mae", 0)
                log_fn(f"Epoch {epoch+1}: loss={loss:.4f}  val_loss={val_loss:.4f}  mae={mae:.4f}")
                records.append({"epoch": epoch + 1, "loss": loss, "val_loss": val_loss, "mae": mae})

        self._cb = _CB()

    def __iter__(self):
        return iter([self._cb])

    def __getattr__(self, name):
        return getattr(self._cb, name)
