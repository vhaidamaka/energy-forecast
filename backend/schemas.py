from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime


# ── Dataset ──────────────────────────────────────────────────────────────────

class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None

class DatasetCreate(DatasetBase):
    pass

class DatasetOut(BaseModel):
    id: int
    name: str
    original_filename: str
    file_format: str
    uploaded_at: datetime
    rows: Optional[int]
    columns: Optional[list[str]]
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    granularity: Optional[str]
    energy_column: Optional[str]
    description: Optional[str]

    model_config = {"from_attributes": True}


# ── Hyperparams ───────────────────────────────────────────────────────────────

class EnsembleParams(BaseModel):
    base_run_ids: list[int] = Field(default_factory=list,
                                    description="IDs of already-trained runs to blend")
    alpha: float = Field(1.0, ge=0.001, le=1000.0,
                         description="Ridge regularisation strength")
    blend_method: str = Field("ridge", pattern="^(ridge|mean|weighted_mape)$")


class NBeatsParams(BaseModel):
    lookback: int = Field(168, ge=24, le=720)
    stack_types: str = Field("trend,seasonality,generic",
                             description="Comma-separated: trend | seasonality | generic")
    num_blocks: int = Field(3, ge=1, le=8)
    layer_width: int = Field(256, ge=32, le=1024)
    num_layers: int = Field(4, ge=2, le=8)
    dropout: float = Field(0.0, ge=0.0, le=0.5)
    epochs: int = Field(100, ge=5, le=500)
    batch_size: int = Field(32, ge=8, le=256)
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1)
    # Shared training params
    train_split: float = Field(0.8, ge=0.5, le=0.95)
    validation_split: float = Field(0.1, ge=0.05, le=0.3)
    optimizer: str = Field("adam", pattern="^(adam|adamw|sgd|rmsprop)$")
    early_stopping: bool = True
    es_patience: int = Field(10, ge=1, le=50)
    reduce_lr: bool = True
    lr_factor: float = Field(0.5, ge=0.1, le=0.9)
    lr_patience: int = Field(5, ge=1, le=30)


class ArimaParams(BaseModel):
    p: int = Field(2, ge=0, le=10)
    d: int = Field(1, ge=0, le=2)
    q: int = Field(2, ge=0, le=10)
    seasonal: bool = False
    seasonal_P: int = Field(1, ge=0, le=3)
    seasonal_D: int = Field(1, ge=0, le=2)
    seasonal_Q: int = Field(1, ge=0, le=3)
    seasonal_m: int = Field(24, ge=2, le=365)
    train_split: float = Field(0.8, ge=0.5, le=0.95)

class LstmParams(BaseModel):
    units_1: int = Field(64, ge=8, le=256)
    units_2: int = Field(32, ge=8, le=128)
    dropout: float = Field(0.2, ge=0.0, le=0.5)
    lookback: int = Field(24, ge=4, le=336)
    epochs: int = Field(50, ge=5, le=500)
    batch_size: int = Field(32, ge=8, le=256)
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1)
    # Training config
    train_split: float = Field(0.8, ge=0.5, le=0.95)
    validation_split: float = Field(0.1, ge=0.05, le=0.3)
    optimizer: str = Field("adam", pattern="^(adam|adamw|sgd|rmsprop)$")
    dense_units: int = Field(16, ge=8, le=256)
    dense_activation: str = Field("relu", pattern="^(relu|tanh|elu|gelu|selu)$")
    # Callbacks
    early_stopping: bool = True
    es_patience: int = Field(10, ge=1, le=50)
    reduce_lr: bool = True
    lr_factor: float = Field(0.5, ge=0.1, le=0.9)
    lr_patience: int = Field(5, ge=1, le=30)

class BiLstmParams(LstmParams):
    merge_mode: str = Field("concat", pattern="^(concat|sum|mul|ave)$")

class GruParams(LstmParams):
    pass  # Same hyperparams as LSTM — GRU uses identical interface

class BiGruParams(LstmParams):
    merge_mode: str = Field("concat", pattern="^(concat|sum|mul|ave)$")

class WaveletLstmParams(LstmParams):
    wavelet: str = Field("db4", pattern="^(db4|db2|haar|sym8|coif1)$")
    level: int = Field(2, ge=1, le=5)
    wave_window: int = Field(168, ge=32, le=512)


class TransformerParams(LstmParams):
    d_model: int = Field(64, ge=16, le=512)
    n_heads: int = Field(4, ge=1, le=16)
    n_layers: int = Field(2, ge=1, le=8)
    ffn_dim: int = Field(128, ge=32, le=1024)

class TFTParams(LstmParams):
    d_model: int = Field(64, ge=16, le=512)
    n_heads: int = Field(4, ge=1, le=16)
    lstm_layers: int = Field(1, ge=1, le=4)


# ── Run ───────────────────────────────────────────────────────────────────────

class RunCreate(BaseModel):
    dataset_id: int
    model: str = Field(..., pattern="^(ensemble|nbeats|arima|lstm|bilstm|gru|bigru|wavelet_lstm|transformer|tft)$")
    hyperparams: dict[str, Any]
    horizon_days: int = Field(7, ge=1, le=90)
    name: Optional[str] = None

class BatchRunCreate(BaseModel):
    """Create one run per model, all with default hyperparams."""
    dataset_id: int
    models: list[str] = Field(
        default=["nbeats","arima","lstm","bilstm","gru","bigru","wavelet_lstm","transformer","tft"],
        description="List of model keys to run"
    )
    horizon_days: int = Field(7, ge=1, le=90)
    name_prefix: Optional[str] = None   # e.g. "Zorya batch" → "Zorya batch — LSTM"


class RunOut(BaseModel):
    id: int
    dataset_id: int
    model: str
    hyperparams: dict[str, Any]
    horizon_days: int
    status: str
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    error_message: Optional[str]
    name: Optional[str]

    model_config = {"from_attributes": True}




# ── Result ────────────────────────────────────────────────────────────────────

class ResultOut(BaseModel):
    id: int
    run_id: int
    mae: Optional[float]
    rmse: Optional[float]
    mape: Optional[float]
    forecast_json: Optional[list[dict]]
    actual_json: Optional[list[dict]]
    test_predicted_json: Optional[list[dict]]
    training_history: Optional[list[dict]]

    model_config = {"from_attributes": True}


class RunWithResult(RunOut):
    result: Optional[ResultOut] = None
    dataset: Optional[DatasetOut] = None
    duration_seconds: Optional[float] = None

    model_config = {"from_attributes": True}

    @classmethod
    def from_run(cls, run) -> "RunWithResult":
        # model_validate with from_attributes, then build a plain dict
        # so we can inject duration_seconds before construction
        data = cls.model_validate(run).model_dump()
        if getattr(run, "started_at", None) and getattr(run, "finished_at", None):
            data["duration_seconds"] = (
                run.finished_at - run.started_at
            ).total_seconds()
        return cls.model_validate(data)
