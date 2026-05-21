"""
Shared sequence building and scaling utilities for all LSTM-based models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_data(df: pd.DataFrame, feature_cols: list[str]):
    """Scale features and target. Returns X_scaled, y_scaled, scaler_X, scaler_y."""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = df[feature_cols].values
    y = df["energy"].values.reshape(-1, 1)

    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y)

    return X_s, y_s, scaler_X, scaler_y


def build_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """Create sliding window sequences."""
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)


def inverse_scale(scaler: MinMaxScaler, arr: np.ndarray) -> np.ndarray:
    arr = arr.reshape(-1, 1)
    return scaler.inverse_transform(arr)


# ── Steps-per-day helper (used by all models) ─────────────────────────────────

_FREQ_STEPS: dict[str, int] = {
    "min": 1440, "t": 1440,
    "5min": 288, "5t": 288,
    "10min": 144, "10t": 144,
    "15min": 96, "15t": 96,
    "30min": 48, "30t": 48,
    "h": 24,
    "d": 1,
}


def steps_per_day(df: "pd.DataFrame") -> int:
    """
    Derive steps-per-day reliably from the DatetimeIndex frequency.
    Falls back to median timedelta if pd.infer_freq returns None (sparse index).
    Never uses rows/date-span which breaks on gappy data.
    """
    import pandas as pd

    freq = pd.infer_freq(df.index)

    if freq is not None:
        fl = freq.lower()
        # Match longest suffix first so "30min" beats "min"
        for key in sorted(_FREQ_STEPS, key=len, reverse=True):
            if fl.endswith(key):
                prefix = fl[: len(fl) - len(key)] or "1"
                try:
                    mult = int(prefix)
                except ValueError:
                    mult = 1
                return max(1, _FREQ_STEPS[key] // mult)

    # Fallback: use median gap between consecutive timestamps
    if len(df) >= 2:
        delta = pd.Series(df.index).diff().dropna().median()
        minutes = delta.total_seconds() / 60
        if minutes > 0:
            return max(1, round(1440 / minutes))

    return 24  # safe default: hourly
