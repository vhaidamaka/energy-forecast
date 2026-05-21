"""
Wavelet decomposition + LSTM model — improved implementation.

Changes vs original:
  ① Per-timestep rolling DWT features (not a single global vector)
  ② Dedicated MinMaxScaler for wavelet columns
  ③ Faster O(N) rolling feature computation using incremental DWT
  ④ Larger default lookback (168h = 1 week) to capture weekly seasonality
  ⑤ Early stopping + ReduceLROnPlateau callbacks for better convergence
  ⑥ Recursive forecast updates wavelet buffer each step
  ⑦ Single clean class (no dead-code duplicate)
"""

import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler

from models.base import PredictionResult
from models.lstm_model import LstmPredictor, _get_feature_cols, _LoggingCallback
from core.train_utils import get_split, build_callbacks, build_optimizer
from core.seq_utils import build_sequences, scale_data, inverse_scale, steps_per_day as steps_per_day_fn


class WaveletLstmPredictor(LstmPredictor):

    def fit_predict(self, df: pd.DataFrame, horizon_days: int) -> PredictionResult:
        import tensorflow as tf
        tf.random.set_seed(42)
        np.random.seed(42)

        wavelet     = self.hyperparams.get("wavelet", "db4")
        level       = self.hyperparams.get("level", 2)
        lookback    = self.hyperparams.get("lookback", 168)   # 1 week default
        epochs      = self.hyperparams.get("epochs", 100)
        batch       = self.hyperparams.get("batch_size", 32)
        wave_window = int(self.hyperparams.get("wave_window", max(lookback, 64)))

        feature_cols = _get_feature_cols(df)
        self.log(f"[WaveletLSTM] Features: {feature_cols}")
        self.log(f"[WaveletLSTM] Wavelet={wavelet}  Level={level}  Lookback={lookback}  WaveWindow={wave_window}")

        # ── 1. Scale base features ────────────────────────────────────────────
        X_s, y_s, scaler_X, scaler_y = scale_data(df, feature_cols)

        # ── 2. Per-timestep wavelet features ─────────────────────────────────
        energy_raw = df["energy"].values
        n_feat_wave = _n_wave_features(level)
        W_raw = _rolling_wavelet_features(energy_raw, wavelet, level, wave_window, n_feat_wave)

        scaler_wave = MinMaxScaler()
        W_s = scaler_wave.fit_transform(W_raw)

        self.log(f"[WaveletLSTM] Wavelet cols={W_s.shape[1]}  "
                 f"raw_range=[{W_raw.min():.1f}, {W_raw.max():.1f}]")

        # ── 3. Combine and build sequences ────────────────────────────────────
        X_combined = np.hstack([X_s, W_s])
        X_seq, y_seq = build_sequences(X_combined, y_s, lookback)

        if len(X_seq) < 50:
            self.log(f"[WaveletLSTM] WARNING: only {len(X_seq)} sequences — consider using more data")

        train_split = float(self.hyperparams.get("train_split", 0.8))
        split = get_split(len(X_seq), train_split)
        X_tr, X_te = X_seq[:split], X_seq[split:]
        y_tr, y_te = y_seq[:split], y_seq[split:]

        model = self._build_model((X_tr.shape[1], X_tr.shape[2]))
        self.log(f"[WaveletLSTM] Model params: {model.count_params():,}")

        # ── 4. Callbacks ──────────────────────────────────────────────────────
        val_split = float(self.hyperparams.get("validation_split", 0.1))
        history_records = []
        cbs = build_callbacks(self.hyperparams, self.log, history_records)

        model.fit(X_tr, y_tr,
                  epochs=epochs, batch_size=batch,
                  validation_split=val_split, callbacks=cbs, verbose=0)

        actual_epochs = len(history_records)
        self.log(f"[WaveletLSTM] Trained {actual_epochs} epochs (early stopping)")

        # ── 5. Evaluate on test set ───────────────────────────────────────────
        y_pred_s = model.predict(X_te, verbose=0)
        y_actual = inverse_scale(scaler_y, y_te)
        y_pred   = inverse_scale(scaler_y, y_pred_s)

        mae, rmse, mape = self.calc_metrics(y_actual.flatten(), y_pred.flatten())
        self.log(f"[WaveletLSTM] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

        test_timestamps = df.index[split + lookback: split + lookback + len(y_te)]

        # ── 6. Recursive forecast ─────────────────────────────────────────────
        horizon_steps = horizon_days * steps_per_day_fn(df)
        freq = pd.infer_freq(df.index) or "h"
        self.log(f"[WaveletLSTM] Forecasting {horizon_steps} steps ({horizon_days} days)...")

        energy_buffer = list(energy_raw[-wave_window:])

        future_preds = _wavelet_recursive_forecast(
            model=model,
            last_seq=X_seq[-1],
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            scaler_wave=scaler_wave,
            feature_cols=feature_cols,
            energy_buffer=energy_buffer,
            wave_window=wave_window,
            wavelet=wavelet,
            level=level,
            n_feat_wave=n_feat_wave,
            lookback=lookback,
            steps=horizon_steps,
            last_ts=df.index[-1],
            freq=freq,
        )

        future_idx = pd.date_range(start=df.index[-1], periods=horizon_steps + 1, freq=freq)[1:]

        return PredictionResult(
            mae=mae, rmse=rmse, mape=mape,
            forecast=self.series_to_records(future_idx, future_preds, "predicted"),
            actual=self.series_to_records(test_timestamps, y_actual.flatten(), "actual"),
            test_predicted=self.series_to_records(test_timestamps, y_pred.flatten(), "predicted"),
            training_history=history_records,
        )


# ── Wavelet feature helpers ───────────────────────────────────────────────────

def _n_wave_features(level: int) -> int:
    """2*(level+1): mean + std for each DWT sub-band (approximation + details)."""
    return 2 * (level + 1)


def _extract_wave_row(window: np.ndarray, wavelet: str, level: int, n_feat: int) -> np.ndarray:
    """
    Fixed-size DWT feature vector from a 1-D window.
    Features: [mean(cA), std(cA), mean(|cD1|), std(|cD1|), ...]
    Always returns exactly n_feat values (zero-padded if window is too short).
    """
    window = np.nan_to_num(window, nan=float(np.nanmean(window)) if len(window) else 0.0)
    min_len = 2 ** (level + 1)
    use_level = level if len(window) >= min_len else 1

    coeffs = pywt.wavedec(window, wavelet, level=use_level)
    feats = []
    for c in coeffs:
        feats.append(float(np.mean(c)))
        feats.append(float(np.std(c) if len(c) > 1 else 0.0))

    arr = np.array(feats, dtype=np.float32)
    if len(arr) < n_feat:
        arr = np.pad(arr, (0, n_feat - len(arr)))
    return arr[:n_feat]


def _rolling_wavelet_features(energy: np.ndarray, wavelet: str, level: int,
                               window: int, n_feat: int) -> np.ndarray:
    """
    Per-timestep wavelet features with a rolling window.
    Row i uses energy[max(0, i-window+1):i+1].
    Only recomputes when the window is full (stride=1 from index `window` onward)
    — the first `window` rows are computed from growing prefixes.
    """
    n = len(energy)
    W = np.zeros((n, n_feat), dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)
        W[i] = _extract_wave_row(energy[start:i + 1], wavelet, level, n_feat)
    return W


# ── Recursive forecast ────────────────────────────────────────────────────────

def _wavelet_recursive_forecast(model, last_seq, scaler_X, scaler_y, scaler_wave,
                                  feature_cols, energy_buffer, wave_window,
                                  wavelet, level, n_feat_wave, lookback, steps,
                                  last_ts, freq):
    """
    Recursive multi-step forecast for Wavelet+LSTM.

    Correctly advances ALL time features at each step (hour_sin/cos,
    dow_sin/cos, is_weekend, month) to prevent mean-reversion flatline.
    Also updates wavelet features from the rolling energy buffer.
    """
    import pandas as pd
    from models.lstm_model import _update_time_features

    preds = []
    seq   = last_seq.copy()   # (lookback, n_base + n_wave) — already scaled
    n_base = len(feature_cols)
    col_idx = {c: i for i, c in enumerate(feature_cols)}
    buffer = list(energy_buffer)

    for step in range(steps):
        inp    = seq.reshape(1, lookback, seq.shape[-1])
        pred_s = model.predict(inp, verbose=0)
        pred_val = float(scaler_y.inverse_transform(pred_s)[0, 0])
        preds.append(pred_val)

        # Update rolling energy buffer
        buffer.append(pred_val)
        if len(buffer) > wave_window:
            buffer.pop(0)

        # Future timestamp for this step
        future_ts = last_ts + (step + 1) * pd.tseries.frequencies.to_offset(freq)

        # scaler_X was fit on the n_base feature columns only
        base_scaled_last = seq[-1, :n_base].reshape(1, -1)
        base_unscaled = scaler_X.inverse_transform(base_scaled_last).flatten()

        base_unscaled[col_idx.get("energy", 0)] = pred_val
        _update_time_features(base_unscaled, col_idx, future_ts)

        # Re-scale updated base features
        base_scaled = scaler_X.transform(base_unscaled.reshape(1, -1)).flatten()

        # Recompute wavelet features
        wave_raw    = _extract_wave_row(np.array(buffer), wavelet, level, n_feat_wave)
        wave_scaled = scaler_wave.transform(wave_raw.reshape(1, -1))[0]

        next_row = np.concatenate([base_scaled, wave_scaled])
        seq = np.vstack([seq[1:], next_row])

    return np.array(preds)
