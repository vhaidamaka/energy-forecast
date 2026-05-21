"""
GRU (Gated Recurrent Unit) model for energy forecasting.
GRU is faster to train than LSTM with comparable accuracy on many time-series tasks
— it has fewer parameters (no separate cell state) which helps on smaller datasets.
Inherits the full fit_predict / recursive forecast pipeline from LstmPredictor,
only overrides _build_model.
"""

from models.lstm_model import LstmPredictor
from core.train_utils import build_optimizer


class GruPredictor(LstmPredictor):

    def _build_model(self, input_shape):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout, Input

        u1 = self.hyperparams.get("units_1", 64)
        u2 = self.hyperparams.get("units_2", 32)
        dr = self.hyperparams.get("dropout", 0.2)

        model = Sequential([
            Input(shape=input_shape),
            GRU(u1, return_sequences=True),
            Dropout(dr),
            GRU(u2),
            Dropout(dr),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer=build_optimizer(self.hyperparams), loss="mse", metrics=["mae"])

        self.log(f"[GRU] units=({u1},{u2})  dropout={dr}  lr={self.hyperparams.get('learning_rate', 0.001)}")
        return model
