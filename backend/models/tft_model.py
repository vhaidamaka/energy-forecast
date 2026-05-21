"""
Temporal Fusion Transformer (TFT) for energy forecasting.
Keras 3 / TF 2.18+ compatible — all ops go through keras.layers or keras.ops.

Architecture (Lim et al., 2021):
  1. Variable Selection Network  — soft per-feature importance weights
  2. GRN (Gated Residual Network) — core building block
  3. LSTM sequence encoder       — temporal context
  4. Static enrichment            — global context injected per-timestep
  5. Multi-Head Self-Attention
  6. Position-wise GRN
  7. Dense(1) output head
"""

import numpy as np
import pandas as pd

import keras
from keras import layers, ops

from models.base import BasePredictor, PredictionResult
from models.lstm_model import _get_feature_cols, _LoggingCallback, _recursive_forecast
from core.train_utils import get_split, build_callbacks, build_optimizer
from core.seq_utils import build_sequences, scale_data, inverse_scale, steps_per_day as steps_per_day_fn


# ── Custom Keras layers (no raw tf.* calls) ───────────────────────────────────

class TileLayer(layers.Layer):
    """Tile a tensor along axis=1 by seq_len, used for broadcasting context."""
    def __init__(self, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len

    def call(self, x):
        # x: (batch, 1, d) → (batch, seq_len, d)
        return ops.tile(x, [1, self.seq_len, 1])

    def get_config(self):
        return {**super().get_config(), "seq_len": self.seq_len}


class ExpandDimsLayer(layers.Layer):
    """Insert a dimension at axis=1."""
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return ops.expand_dims(x, axis=self.axis)

    def get_config(self):
        return {**super().get_config(), "axis": self.axis}


class LastTimestepLayer(layers.Layer):
    """Select the last timestep: (batch, seq, d) → (batch, d)."""
    def call(self, x):
        return x[:, -1, :]


class GRNLayer(layers.Layer):
    """
    Gated Residual Network:
      h     = ELU( Dense(x) )
      gate  = sigmoid( Dense(h) )
      lin   = Dense(h)
      out   = LayerNorm( residual_proj(x) + gate * lin )
    """
    def __init__(self, d_model, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate

        self.dense1    = layers.Dense(d_model, activation="elu")
        self.drop1     = layers.Dropout(dropout_rate)
        self.gate_d    = layers.Dense(d_model, activation="sigmoid")
        self.linear_d  = layers.Dense(d_model)
        self.multiply  = layers.Multiply()
        self.res_proj  = layers.Dense(d_model, use_bias=False)
        self.add       = layers.Add()
        self.norm      = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        residual = self.res_proj(x)
        h    = self.dense1(x)
        h    = self.drop1(h, training=training)
        gate = self.gate_d(h)
        lin  = self.linear_d(h)
        h    = self.multiply([gate, lin])
        return self.norm(self.add([residual, h]))

    def get_config(self):
        return {**super().get_config(),
                "d_model": self.d_model,
                "dropout_rate": self.dropout_rate}


class VSNLayer(layers.Layer):
    """
    Variable Selection Network:
      weights = softmax( Dense( GlobalAvgPool(proj(x)) ) )  # per-feature importance
      out     = GRN( weighted_input_projection )
    """
    def __init__(self, n_features, d_model, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_features   = n_features
        self.d_model      = d_model
        self.dropout_rate = dropout_rate

        self.proj        = layers.Dense(d_model)
        self.pool        = layers.GlobalAveragePooling1D()
        self.weight_dense = layers.Dense(n_features, activation="softmax")
        self.multiply    = layers.Multiply()
        self.weighted_proj = layers.Dense(d_model)
        self.grn         = GRNLayer(d_model, dropout_rate)

    def call(self, x, training=False):
        # x: (batch, seq_len, n_features)
        proj = self.proj(x)                           # (batch, seq_len, d_model)
        ctx  = self.pool(proj)                        # (batch, d_model)
        w    = self.weight_dense(ctx)                 # (batch, n_features)
        # Expand w to (batch, seq_len, n_features) via RepeatVector
        w_exp = ops.expand_dims(w, axis=1)            # (batch, 1, n_features)
        w_exp = ops.tile(w_exp, [1, ops.shape(x)[1], 1])  # (batch, seq_len, n_features)
        weighted = self.multiply([x, w_exp])
        out = self.weighted_proj(weighted)
        return self.grn(out, training=training)

    def get_config(self):
        return {**super().get_config(),
                "n_features": self.n_features,
                "d_model": self.d_model,
                "dropout_rate": self.dropout_rate}


class StaticEnrichmentLayer(layers.Layer):
    """
    Global context (mean-pooled LSTM output) broadcast back to every timestep,
    added to x and passed through a GRN.
    """
    def __init__(self, d_model, seq_len, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model      = d_model
        self.seq_len      = seq_len
        self.dropout_rate = dropout_rate

        self.ctx_pool  = layers.GlobalAveragePooling1D()
        self.ctx_proj  = layers.Dense(d_model)
        self.expand    = ExpandDimsLayer(axis=1)
        self.tile      = TileLayer(seq_len)
        self.add       = layers.Add()
        self.grn       = GRNLayer(d_model, dropout_rate)

    def call(self, x, training=False):
        ctx = self.ctx_pool(x)                  # (batch, d_model)
        ctx = self.ctx_proj(ctx)                # (batch, d_model)
        ctx = self.expand(ctx)                  # (batch, 1, d_model)
        ctx = self.tile(ctx)                    # (batch, seq_len, d_model)
        enriched = self.add([x, ctx])
        return self.grn(enriched, training=training)

    def get_config(self):
        return {**super().get_config(),
                "d_model": self.d_model,
                "seq_len": self.seq_len,
                "dropout_rate": self.dropout_rate}


# ── TFT Predictor ─────────────────────────────────────────────────────────────

class TFTPredictor(BasePredictor):

    def _build_model(self, input_shape):
        seq_len, n_features = input_shape
        d_model     = self.hyperparams.get("d_model", 64)
        n_heads     = self.hyperparams.get("n_heads", 4)
        lstm_layers = self.hyperparams.get("lstm_layers", 1)
        dr          = self.hyperparams.get("dropout", 0.1)
        lr          = self.hyperparams.get("learning_rate", 0.001)

        # d_model must be divisible by n_heads
        if d_model % n_heads != 0:
            d_model = max(n_heads, (d_model // n_heads) * n_heads)
            self.log(f"[TFT] Adjusted d_model → {d_model} (divisible by n_heads={n_heads})")

        inp = layers.Input(shape=(seq_len, n_features), name="input")

        # 1. Variable Selection Network
        x = VSNLayer(n_features, d_model, dr, name="vsn")(inp)

        # 2. LSTM encoder
        for i in range(lstm_layers):
            x = layers.LSTM(d_model, return_sequences=True,
                            dropout=dr, name=f"lstm_{i}")(x)

        # 3. Static enrichment
        x = StaticEnrichmentLayer(d_model, seq_len, dr, name="static_enrich")(x)

        # 4. Temporal self-attention + residual
        attn = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads,
            dropout=dr, name="temporal_attn"
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6, name="attn_norm")(
            layers.Add(name="attn_add")([x, attn])
        )

        # 5. Position-wise GRN
        x = GRNLayer(d_model, dr, name="poswise_grn")(x)

        # 6. Output head — last timestep only
        x = LastTimestepLayer(name="last_step")(x)
        x = layers.Dense(32, activation="relu", name="head_dense")(x)
        x = layers.Dropout(dr, name="head_drop")(x)
        out = layers.Dense(1, name="output")(x)

        model = keras.Model(inp, out, name="TFT")
        model.compile(
            optimizer=build_optimizer(self.hyperparams),
            loss="mse",
            metrics=["mae"]
        )

        self.log(
            f"[TFT] d_model={d_model}  n_heads={n_heads}  "
            f"lstm_layers={lstm_layers}  params={model.count_params():,}"
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
        self.log(f"[TFT] Features: {feature_cols}")
        self.log(f"[TFT] Lookback={lookback}  Epochs={epochs}  Batch={batch}")

        X_s, y_s, scaler_X, scaler_y = scale_data(df, feature_cols)
        X_seq, y_seq = build_sequences(X_s, y_s, lookback)

        train_split = float(self.hyperparams.get("train_split", 0.8))
        split = get_split(len(X_seq), train_split)
        X_tr, X_te = X_seq[:split], X_seq[split:]
        y_tr, y_te = y_seq[:split], y_seq[split:]

        model = self._build_model((X_tr.shape[1], X_tr.shape[2]))
        val_split = float(self.hyperparams.get("validation_split", 0.1))
        history_records = []
        cbs = build_callbacks(self.hyperparams, self.log, history_records)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch,
                  validation_split=val_split, callbacks=cbs, verbose=0)
        self.log(f'[TFT] Trained {len(history_records)} epochs')

        y_pred_s = model.predict(X_te, verbose=0)
        y_actual = inverse_scale(scaler_y, y_te)
        y_pred   = inverse_scale(scaler_y, y_pred_s)

        mae, rmse, mape = self.calc_metrics(y_actual.flatten(), y_pred.flatten())
        self.log(f"[TFT] MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

        test_timestamps = df.index[split + lookback: split + lookback + len(y_te)]

        horizon_steps = horizon_days * steps_per_day_fn(df)
        self.log(f"[TFT] Forecasting {horizon_steps} future steps...")

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
