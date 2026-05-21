from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Union
import os

BASE_DIR = Path(__file__).parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "Energy Predictor"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # In Docker these are overridden via environment variables to /app/storage/*
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/energy_predictor.db"
    DATA_DIR: Path = BASE_DIR / "data"
    RUNS_DIR: Path = BASE_DIR / "runs"

    # Comma-separated origins supported via env var:
    # CORS_ORIGINS="http://localhost:5173,http://localhost:80"
    CORS_ORIGINS: Union[list[str], str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:80",
        "http://localhost",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def get_cors_origins(self) -> list[str]:
        if isinstance(self.CORS_ORIGINS, str):
            return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]
        return self.CORS_ORIGINS

settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.RUNS_DIR.mkdir(parents=True, exist_ok=True)
