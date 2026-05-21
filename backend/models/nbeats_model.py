"""
N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.
Oreshkin et al., 2019 — https://arxiv.org/abs/1905.10437

Architecture:
  • Stack of "blocks", each containing a small fully-connected network
  • Each block outputs TWO things:
      backcast  — what it can explain of its own input window
      forecast  — its contribution to the multi-step horizon
  • Residual connections: each block's input = previous block's input − previous backcast
    (the unexplained residual propagates forward, like differentiable boosting)
  • Final forecast = sum of all block forecasts

Stack types:
  "generic"     — unconstrained basis (data-driven)
  "trend"       — polynomial basis (θ maps to polynomial coefficients)
  "seasonality" — Fourier basis (θ maps to sine/cosine coefficients)

For energy data the recommended stack order is [trend, seasonality, generic]:
  • trend stack     → captures slow drift (e.g. seasonal load growth)
  • seasonality     → captures 24h / 168h cycles explicitly
  • generic         → mops up anything remaining
"""

import numpy as np
import pandas as pd

from models.base import BasePredictor, PredictionResult
from core.seq_utils import scale_data, inverse_scale, steps_per_day as steps_per_day_fn
from core.train_utils import get_split, build_callbacks, build_optimizer
from models.lstm_model import _get_feature_cols, _update_time_features


class NBeatsPredictor(BasePredictor):
    """N-BEATS predictor. Uses only the energy column (univariate by design)."""

    def fit_predict(self, df: pd.DataFrame, horizon_days: int) -> PredictionResult:
        import tensorflow as tf
        tf.random.set_seed(42)
        np.random.seed(42)

        # ── Hyperparameters ────────────────────────────────────────────────────
        lookback        = int(self.hyperparams.get("lookback", 168))      # input window
        horizon_steps   = horizon_days * steps_per_day_fn(df)             # forecast steps H
        epochs          = int(self.hyperparams.get("epochs", 100))
        batch           = int(self.hyperparams.get("batch_size", 32))
        train_split     = float(self.hyperparams.get("train_split", 0.8))
        val_split       = float(self.hyperparams.get("validation_split", 0.1))
        # Stack config: comma-separated, e.g. "trend,seasonality,generic"
        stack_types_str = self.hyperparams.get("stack_types", "trend,seasonality,generic")
        stack_types     = [s.strip() for s in stack_types_str.split(",")]
        num_blocks      = int(self.hyperparams.get("num_blocks", 3))       # blocks per stack
        layer_width     = int(self.hyperparams.get("layer_width", 256))    # FC units per layer
        num_layers      = int(self.hyperparams.get("num_layers", 4))       # FC layers per block
        dropout         = float(self.hyperparams.get("dropout", 0.0))      # usually 0 for N-BEATS

        self.log(f"[N-BEATS] lookback={lookback}  horizon={horizon_steps}  "
                 f"stacks={stack_types}  blocks/stack={num_blocks}  "
                 f"layer_width={layer_width}  num_layers={num_layers}")

        # ── Scale (energy only — N-BEATS is univariate) ───────────────────────
        from sklearn.preprocessing import MinMaxScaler
        energy = df["energy"].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        energy_s = scaler.fit_transform(energy).flatten()

        # ── Build sliding windows ─────────────────────────────────────────────
        # Each sample: X = lookback values, y = next horizon_steps values
        X_list, y_list = [], []
        for i in range(lookback, len(energy_s) - horizon_steps + 1):
            X_list.append(energy_s[i - lookback: i])
            y_list.append(energy_s[i: i + horizon_steps])
        X_all = np.array(X_list, dtype=np.float32)   # (N, lookback)
        y_all = np.array(y_list, dtype=np.float32)   # (N, horizon_steps)

        split = get_split(len(X_all), train_split)
        X_tr, X_te = X_all[:split], X_all[split:]
        y_tr, y_te = y_all[:split], y_all[split:]

        self.log(f"[N-BEATS] Train={len(X_tr)}  Test={len(X_te)}  Params computing...")

        # ── Build Keras model ─────────────────────────────────────────────────
        model = _build_nbeats(
            lookback=lookback,
            horizon=horizon_steps,
            stack_types=stack_types,
            num_blocks=num_blocks,
            layer_width=layer_width,
            num_layers=num_layers,
            dropout=dropout,
            optimizer=build_optimizer(self.hyperparams),
        )
        self.log(f"[N-BEATS] Parameters: {model.count_params():,}")

        # ── Train ─────────────────────────────────────────────────────────────
        history_records = []
        cbs = build_callbacks(self.hyperparams, self.log, history_records)

        model.fit(
            X_tr, y_tr,
            epochs=epochs, batch_size=batch,
            validation_split=val_split,
            callbacks=cbs, verbose=0,
        )
        self.log(f"[N-BEATS] Trained {len(history_records)} epochs")

        # ── Evaluate on test set ──────────────────────────────────────────────
        y_pred_s = model.predict(X_te, verbose=0)          # (N_te, horizon_steps)
        y_pred_s = np.clip(y_pred_s, 0, 1)                  # keep in [0,1] before inverse

        # Inverse-scale: reshape (N, H) → (N*H, 1) → inverse → reshape back
        y_actual_flat = scaler.inverse_transform(y_te.reshape(-1, 1)).reshape(len(y_te), -1)
        y_pred_flat   = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).reshape(len(y_pred_s), -1)

        # Metrics on step-1 predictions (most meaningful for comparison with other models)
        mae, rmse, mape = self.calc_metrics(y_actual_flat[:, 0], y_pred_flat[:, 0])
        self.log(f"[N-BEATS] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

        # Build test timestamps
        # test sequences start at index: split + lookback in original df
        test_start_idx = split + lookback
        test_timestamps = df.index[test_start_idx: test_start_idx + len(y_te)]

        # ── Future forecast ───────────────────────────────────────────────────
        self.log(f"[N-BEATS] Forecasting {horizon_steps} steps ({horizon_days} days)...")

        # Use the very last lookback window from the full dataset
        last_window = energy_s[-lookback:].reshape(1, -1).astype(np.float32)
        forecast_s  = model.predict(last_window, verbose=0)[0]    # (horizon_steps,)
        forecast_s  = np.clip(forecast_s, 0, 1)
        forecast    = scaler.inverse_transform(forecast_s.reshape(-1, 1)).flatten()

        freq = pd.infer_freq(df.index) or "h"
        future_idx = pd.date_range(start=df.index[-1], periods=horizon_steps + 1, freq=freq)[1:]

        return PredictionResult(
            mae=mae, rmse=rmse, mape=mape,
            forecast=self.series_to_records(future_idx, forecast, "predicted"),
            actual=self.series_to_records(
                test_timestamps, y_actual_flat[:, 0], "actual"
            ),
            test_predicted=self.series_to_records(
                test_timestamps, y_pred_flat[:, 0], "predicted"
            ),
            training_history=history_records,
        )


# ── N-BEATS Keras model builder ──────────────────────────────────────────────

def _build_nbeats(lookback, horizon, stack_types, num_blocks,
                  layer_width, num_layers, dropout, optimizer):
    """
    Build the full N-BEATS model as a Keras functional graph.

    Returns a model: input (lookback,) → output (horizon,)

    Internally the computation graph is:
        residual = x
        total_forecast = zeros(horizon)
        for each stack:
            for each block:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                total_forecast += block_forecast
        return total_forecast
    """
    import keras
    from keras import layers

    x_input = keras.Input(shape=(lookback,), name="input")

    # We track the residual signal and accumulated forecast using Keras tensors
    residual = x_input
    forecasts = []

    for stack_idx, stack_type in enumerate(stack_types):
        for block_idx in range(num_blocks):
            block_name = f"{stack_type}_s{stack_idx}_b{block_idx}"

            # ── Shared FC trunk ───────────────────────────────────────────────
            h = residual
            for layer_idx in range(num_layers):
                h = layers.Dense(layer_width, activation="relu",
                                 name=f"{block_name}_fc{layer_idx}")(h)
                if dropout > 0:
                    h = layers.Dropout(dropout)(h)

            # ── Basis expansion: backcast_theta and forecast_theta ────────────
            if stack_type == "trend":
                degree = int(np.ceil(lookback / horizon)) + 1
                degree = min(degree, 5)    # cap to avoid over-parameterisation
                backcast_theta = layers.Dense(
                    degree + 1, name=f"{block_name}_bc_theta")(h)
                forecast_theta = layers.Dense(
                    degree + 1, name=f"{block_name}_fc_theta")(h)
                # Map theta → time series via polynomial basis
                backcast  = _polynomial_basis_layer(lookback,  degree, f"{block_name}_bc")(backcast_theta)
                block_fct = _polynomial_basis_layer(horizon,   degree, f"{block_name}_fc")(forecast_theta)

            elif stack_type == "seasonality":
                n_harmonics = int(np.ceil(horizon / 2))
                theta_dim = 2 * n_harmonics              # sin + cos coefficients
                backcast_theta = layers.Dense(
                    theta_dim, name=f"{block_name}_bc_theta")(h)
                forecast_theta = layers.Dense(
                    theta_dim, name=f"{block_name}_fc_theta")(h)
                backcast  = _fourier_basis_layer(lookback,  n_harmonics, f"{block_name}_bc")(backcast_theta)
                block_fct = _fourier_basis_layer(horizon,   n_harmonics, f"{block_name}_fc")(forecast_theta)

            else:  # generic — unconstrained linear projection
                backcast  = layers.Dense(lookback, name=f"{block_name}_bc")(h)
                block_fct = layers.Dense(horizon,  name=f"{block_name}_fc")(h)

            # ── Residual connection ───────────────────────────────────────────
            residual = layers.Subtract(name=f"{block_name}_residual")([residual, backcast])
            forecasts.append(block_fct)

    # Sum all block forecasts
    if len(forecasts) == 1:
        output = forecasts[0]
    else:
        output = layers.Add(name="total_forecast")(forecasts)

    model = keras.Model(inputs=x_input, outputs=output, name="nbeats")
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


# ── Interpretable basis layers ────────────────────────────────────────────────

def _polynomial_basis_layer(length: int, degree: int, name: str):
    """
    Returns a Dense-equivalent layer that maps theta (degree+1,) → series (length,).
    The weight matrix T is the Vandermonde matrix of normalised time indices.
    Weights are non-trainable — only theta is learned.
    """
    import keras
    import tensorflow as tf

    t = np.linspace(0, 1, length, dtype=np.float32)               # (length,)
    T = np.stack([t ** i for i in range(degree + 1)], axis=1)     # (length, degree+1)

    class PolyLayer(keras.layers.Layer):
        def __init__(self, basis_matrix, **kwargs):
            super().__init__(**kwargs)
            self.basis = tf.constant(basis_matrix, dtype=tf.float32)

        def call(self, theta):
            # theta: (batch, degree+1)  basis: (length, degree+1)
            # output: (batch, length)
            return tf.matmul(theta, tf.transpose(self.basis))

        def get_config(self):
            cfg = super().get_config()
            return cfg

    return PolyLayer(T, name=name)


def _fourier_basis_layer(length: int, n_harmonics: int, name: str):
    """
    Returns a layer that maps theta (2*n_harmonics,) → series (length,)
    via Fourier (sine+cosine) basis.
    """
    import keras
    import tensorflow as tf

    t = np.linspace(0, 1, length, dtype=np.float32)
    # Build (length, 2*n_harmonics) matrix: [cos(2π*1*t), ..., sin(2π*1*t), ...]
    cols = []
    for h in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * np.pi * h * t))
        cols.append(np.sin(2 * np.pi * h * t))
    F = np.stack(cols, axis=1)                                     # (length, 2*n_harmonics)

    class FourierLayer(keras.layers.Layer):
        def __init__(self, basis_matrix, **kwargs):
            super().__init__(**kwargs)
            self.basis = tf.constant(basis_matrix, dtype=tf.float32)

        def call(self, theta):
            return tf.matmul(theta, tf.transpose(self.basis))

        def get_config(self):
            cfg = super().get_config()
            return cfg

    return FourierLayer(F, name=name)
