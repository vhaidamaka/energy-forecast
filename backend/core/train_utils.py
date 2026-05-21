"""
Shared training utilities used by all neural models.
Centralises: callback construction, split logic, optimizer selection.
"""

import numpy as np


def get_split(n: int, train_split: float) -> int:
    """Return the index at which training data ends."""
    train_split = max(0.5, min(0.95, train_split))
    return int(n * train_split)


def build_callbacks(hyperparams: dict, log_fn, history_records: list) -> list:
    """
    Build Keras callbacks from hyperparams dict.
    Always includes the logging callback.
    Optionally adds EarlyStopping and ReduceLROnPlateau.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from models.lstm_model import _LoggingCallback

    cbs = list(_LoggingCallback(log_fn, history_records))

    if hyperparams.get("early_stopping", True):
        patience = int(hyperparams.get("es_patience", 10))
        cbs.append(EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ))

    if hyperparams.get("reduce_lr", True):
        cbs.append(ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(hyperparams.get("lr_factor", 0.5)),
            patience=int(hyperparams.get("lr_patience", 5)),
            min_lr=1e-6,
            verbose=0,
        ))

    return cbs


def build_optimizer(hyperparams: dict):
    """Build a Keras optimizer from hyperparams."""
    import keras
    lr = float(hyperparams.get("learning_rate", 0.001))
    name = hyperparams.get("optimizer", "adam").lower()

    if name == "adamw":
        wd = float(hyperparams.get("weight_decay", 1e-4))
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    elif name == "sgd":
        momentum = float(hyperparams.get("momentum", 0.9))
        return keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True)
    elif name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    else:  # adam (default)
        return keras.optimizers.Adam(learning_rate=lr)
