"""
Vanilla Transformer encoder model for energy forecasting.
Keras 3 / TF 2.18+ compatible — positional encoding via a proper Keras layer.

Architecture:
  1. Linear input projection  (n_features → d_model)
  2. Sinusoidal positional encoding  (learned-free, added as a Keras layer)
  3. N × Transformer encoder blocks
       MultiHeadAttention → Add&Norm → FFN(relu) → Add&Norm
  4. GlobalAveragePooling1D
  5. Dense(32, relu) → Dropout → Dense(1)
"""

import numpy as np
import pandas as pd

import keras
from keras import layers, ops

from models.base import BasePredictor, PredictionResult
from models.lstm_model import _get_feature_cols, _LoggingCallback, _recursive_forecast
from core.train_utils import get_split, build_callbacks, build_optimizer
from core.seq_utils import build_sequences, scale_data, inverse_scale, steps_per_day as steps_per_day_fn


# ── Positional Encoding as a proper Keras layer ───────────────────────────────

class SinusoidalPositionalEncoding(layers.Layer):
    """
    Adds fixed sinusoidal positional encoding to the input.
    Encoding is pre-computed in __init__ and stored as a non-trainable weight.
    """
    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model

        # Pre-compute encoding matrix (seq_len, d_model)
        positions = np.arange(seq_len)[:, np.newaxis]
        dims      = np.arange(d_model)[np.newaxis, :]
        angles    = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        # Store as (1, seq_len, d_model) — broadcast over batch
        self._pe = angles[np.newaxis, :, :].astype(np.float32)

    def call(self, x):
        return x + self._pe      # keras.ops.add would also work

    def get_config(self):
        return {**super().get_config(),
                "seq_len": self.seq_len,
                "d_model": self.d_model}


# ── Transformer encoder block ─────────────────────────────────────────────────

def _transformer_block(x, d_model, n_heads, ffn_dim, dropout, block_idx):
    """One Transformer encoder block (functional style, pure Keras layers)."""
    # Multi-head self-attention
    attn = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads,
        dropout=dropout,
        name=f"attn_{block_idx}"
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"attn_norm_{block_idx}")(
        layers.Add(name=f"attn_add_{block_idx}")([x, attn])
    )

    # Feed-forward sub-layer
    ffn = layers.Dense(ffn_dim, activation="relu", name=f"ffn1_{block_idx}")(x)
    ffn = layers.Dropout(dropout, name=f"ffn_drop_{block_idx}")(ffn)
    ffn = layers.Dense(d_model, name=f"ffn2_{block_idx}")(ffn)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"ffn_norm_{block_idx}")(
        layers.Add(name=f"ffn_add_{block_idx}")([x, ffn])
    )
    return x


# ── Predictor ─────────────────────────────────────────────────────────────────

class TransformerPredictor(BasePredictor):

    def _build_model(self, input_shape):
        seq_len, n_features = input_shape
        d_model  = self.hyperparams.get("d_model", 64)
        n_heads  = self.hyperparams.get("n_heads", 4)
        n_layers = self.hyperparams.get("n_layers", 2)
        ffn_dim  = self.hyperparams.get("ffn_dim", 128)
        dr       = self.hyperparams.get("dropout", 0.1)
        lr       = self.hyperparams.get("learning_rate", 0.001)

        if d_model % n_heads != 0:
            d_model = max(n_heads, (d_model // n_heads) * n_heads)
            self.log(f"[Transformer] Adjusted d_model → {d_model} (divisible by n_heads={n_heads})")

        inp = layers.Input(shape=(seq_len, n_features), name="input")

        # Input projection + positional encoding
        x = layers.Dense(d_model, name="input_proj")(inp)
        x = SinusoidalPositionalEncoding(seq_len, d_model, name="pos_enc")(x)
        x = layers.Dropout(dr, name="input_drop")(x)

        # Encoder stack
        for i in range(n_layers):
            x = _transformer_block(x, d_model, n_heads, ffn_dim, dr, i)

        # Pooling + head
        x = layers.GlobalAveragePooling1D(name="pool")(x)
        x = layers.Dense(32, activation="relu", name="head_dense")(x)
        x = layers.Dropout(dr, name="head_drop")(x)
        out = layers.Dense(1, name="output")(x)

        model = keras.Model(inp, out, name="Transformer")
        model.compile(
            optimizer=build_optimizer(self.hyperparams),
            loss="mse",
            metrics=["mae"]
        )

        self.log(
            f"[Transformer] d_model={d_model}  n_heads={n_heads}  "
            f"n_layers={n_layers}  ffn_dim={ffn_dim}  params={model.count_params():,}"
        )
        return model

    def fit_predict(self, df: pd.DataFrame, horizon_days: int) -> PredictionResult:
        import tensorflow as tf
        tf.random.set_seed(42)
        np.random.seed(42)

        lookback = self.hyperparams.get("lookback", 48)
        epochs   = self.hyperparams.get("epochs", 50)
        batch    = self.hyperparams.get("batch_size", 32)

        feature_cols = _get_feature_cols(df)
        self.log(f"[Transformer] Features: {feature_cols}")
        self.log(f"[Transformer] Lookback={lookback}  Epochs={epochs}  Batch={batch}")

        X_s, y_s, scaler_X, scaler_y = scale_data(df, feature_cols)
        X_seq, y_seq = build_sequences(X_s, y_s, lookback)

        train_split = float(self.hyperparams.get("train_split", 0.8))
        split = get_split(len(X_seq), train_split)
        X_tr, X_te = X_seq[:split], X_seq[split:]
        y_tr, y_te = y_seq[:split], y_seq[split:]

        model = self._build_model((X_tr.shape[1], X_tr.shape[2]))
        self.log(f"[Transformer] Total params: {model.count_params():,}")
        val_split = float(self.hyperparams.get("validation_split", 0.1))
        history_records = []
        cbs = build_callbacks(self.hyperparams, self.log, history_records)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch,
                  validation_split=val_split, callbacks=cbs, verbose=0)
        self.log(f'[Transformer] Trained {len(history_records)} epochs')

        y_pred_s = model.predict(X_te, verbose=0)
        y_actual = inverse_scale(scaler_y, y_te)
        y_pred   = inverse_scale(scaler_y, y_pred_s)

        mae, rmse, mape = self.calc_metrics(y_actual.flatten(), y_pred.flatten())
        self.log(f"[Transformer] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

        test_timestamps = df.index[split + lookback: split + lookback + len(y_te)]

        horizon_steps = horizon_days * steps_per_day_fn(df)
        self.log(f"[Transformer] Forecasting {horizon_steps} future steps...")

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
