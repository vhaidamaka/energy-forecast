from models.base import BasePredictor
from models.arima_model import ArimaPredictor
from models.lstm_model import LstmPredictor
from models.bilstm_model import BiLstmPredictor
from models.gru_model import GruPredictor
from models.bigru_model import BiGruPredictor
from models.wavelet_lstm_model import WaveletLstmPredictor
from models.transformer_model import TransformerPredictor
from models.tft_model import TFTPredictor
from models.nbeats_model import NBeatsPredictor
from models.ensemble_model import EnsemblePredictor
from typing import Callable

REGISTRY: dict[str, type[BasePredictor]] = {
    "nbeats":       NBeatsPredictor,
    "ensemble":     EnsemblePredictor,
    "arima":        ArimaPredictor,
    "lstm":         LstmPredictor,
    "bilstm":       BiLstmPredictor,
    "gru":          GruPredictor,
    "bigru":        BiGruPredictor,
    "wavelet_lstm": WaveletLstmPredictor,
    "transformer":  TransformerPredictor,
    "tft":          TFTPredictor,
}

# ── Shared training params (added to every neural model) ─────────────────────
_SHARED_TRAIN = {
    "train_split":       0.8,    # fraction of sequences used for training
    "validation_split":  0.1,    # fraction of training used for validation
    "optimizer":         "adam", # adam | adamw | sgd | rmsprop
    "dense_units":       16,     # head dense layer width
    "dense_activation":  "relu", # activation for dense head
    "early_stopping":    True,   # enable EarlyStopping callback
    "es_patience":       10,     # epochs to wait before stopping
    "reduce_lr":         True,   # enable ReduceLROnPlateau callback
    "lr_factor":         0.5,    # LR reduction factor
    "lr_patience":       5,      # epochs to wait before reducing LR
}

DEFAULT_HYPERPARAMS: dict[str, dict] = {
    "ensemble": {
        "base_run_ids":  [],          # list of run IDs to blend — set at runtime
        "alpha":         1.0,         # Ridge regularisation strength
        "blend_method":  "ridge",     # ridge | mean | weighted_mape
    },
    "nbeats": {
        "lookback": 168,           # input window (1 week of hourly data)
        "stack_types": "trend,seasonality,generic",
        "num_blocks": 3,           # blocks per stack
        "layer_width": 256,        # FC units per layer inside each block
        "num_layers": 4,           # FC layers per block
        "dropout": 0.0,            # N-BEATS typically uses no dropout
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        **_SHARED_TRAIN,
    },
    "arima": {
        "p": 2, "d": 1, "q": 2,
        "seasonal": False, "seasonal_m": 24,
        "seasonal_P": 1, "seasonal_D": 1, "seasonal_Q": 1,
        "train_split": 0.8,
    },
    "lstm": {
        "units_1": 64, "units_2": 32, "dropout": 0.2, "lookback": 24,
        "epochs": 50, "batch_size": 32, "learning_rate": 0.001,
        **_SHARED_TRAIN,
    },
    "bilstm": {
        "units_1": 64, "units_2": 32, "dropout": 0.2, "lookback": 24,
        "epochs": 50, "batch_size": 32, "learning_rate": 0.001,
        "merge_mode": "concat",
        **_SHARED_TRAIN,
    },
    "gru": {
        "units_1": 64, "units_2": 32, "dropout": 0.2, "lookback": 24,
        "epochs": 50, "batch_size": 32, "learning_rate": 0.001,
        **_SHARED_TRAIN,
    },
    "bigru": {
        "units_1": 64, "units_2": 32, "dropout": 0.2, "lookback": 24,
        "epochs": 50, "batch_size": 32, "learning_rate": 0.001,
        "merge_mode": "concat",
        **_SHARED_TRAIN,
    },
    "wavelet_lstm": {
        "units_1": 64, "units_2": 32, "dropout": 0.2, "lookback": 168,
        "epochs": 100, "batch_size": 32, "learning_rate": 0.001,
        "wavelet": "db4", "level": 2, "wave_window": 168,
        **_SHARED_TRAIN,
    },
    "transformer": {
        "d_model": 64, "n_heads": 4, "n_layers": 2, "ffn_dim": 128,
        "dropout": 0.1, "lookback": 48,
        "epochs": 50, "batch_size": 32, "learning_rate": 0.001,
        **_SHARED_TRAIN,
    },
    "tft": {
        "d_model": 64, "n_heads": 4, "lstm_layers": 1,
        "dropout": 0.1, "lookback": 48,
        "epochs": 50, "batch_size": 32, "learning_rate": 0.001,
        **_SHARED_TRAIN,
    },
}


def get_predictor(model_name: str, hyperparams: dict,
                  log_callback: Callable | None = None) -> BasePredictor:
    cls = REGISTRY.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(REGISTRY)}")
    merged = {**DEFAULT_HYPERPARAMS.get(model_name, {}), **hyperparams}
    return cls(merged, log_callback)
