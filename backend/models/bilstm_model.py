"""
Bidirectional LSTM model for energy forecasting.
Inherits from LstmPredictor, only overrides _build_model.
"""

from models.lstm_model import LstmPredictor
from core.train_utils import build_optimizer


class BiLstmPredictor(LstmPredictor):

    def _build_model(self, input_shape):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input

        u1    = self.hyperparams.get("units_1", 64)
        u2    = self.hyperparams.get("units_2", 32)
        dr    = self.hyperparams.get("dropout", 0.2)
        merge = self.hyperparams.get("merge_mode", "concat")

        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(u1, return_sequences=True), merge_mode=merge),
            Dropout(dr),
            Bidirectional(LSTM(u2), merge_mode=merge),
            Dropout(dr),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer=build_optimizer(self.hyperparams), loss="mse", metrics=["mae"])

        self.log(f"[BiLSTM] merge_mode={merge}")
        return model
