import warnings
import os

# Suppress Keras/TF deprecation warnings in logs
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', message='.*input_shape.*')
warnings.filterwarnings('ignore', message='.*input_dim.*')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from core.database import init_db
from api.routes.datasets import router as datasets_router
from api.routes.runs import router as runs_router
from api.routes.export import router as export_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Energy Consumption Prediction API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router, prefix="/api")
app.include_router(runs_router, prefix="/api")
app.include_router(export_router, prefix="/api")


@app.on_event("startup")
def startup():
    init_db()
    print(f"✓ {settings.APP_NAME} v{settings.VERSION} started")
    print(f"  Data dir: {settings.DATA_DIR}")
    print(f"  DB: {settings.DATABASE_URL}")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": settings.VERSION}


@app.get("/api/models/defaults")
def model_defaults():
    from models import DEFAULT_HYPERPARAMS
    return DEFAULT_HYPERPARAMS


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
