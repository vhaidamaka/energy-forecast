"""
Training job runner.

Two launch paths:
  1. launch_training(run_id, loop) — used by single POST /runs/{id}/start
     Registers a queue, starts a daemon thread, streams logs via SSE.

  2. launch_training_batch(run_ids, loop) — used by POST /runs/batch
     Registers a queue for EACH run, then runs them sequentially in ONE
     thread so they don't fight for RAM.  Each run's queue is live while
     that run trains; stream_logs works identically for batch runs.
"""

import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session
from core.database import SessionLocal, Run, Result
from core.preprocessing import load_and_preprocess
from models import get_predictor


# ── Queue registry ────────────────────────────────────────────────────────────

_log_queues: dict[int, asyncio.Queue] = {}


def get_log_queue(run_id: int) -> asyncio.Queue | None:
    return _log_queues.get(run_id)


def register_log_queue(run_id: int, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _log_queues[run_id] = q
    return q


def unregister_log_queue(run_id: int):
    _log_queues.pop(run_id, None)


# ── Single-run launch (existing path) ────────────────────────────────────────

def launch_training(run_id: int, loop: asyncio.AbstractEventLoop):
    """Start a single training run in a daemon thread with SSE logging."""
    thread = threading.Thread(
        target=_train_worker, args=(run_id, loop), daemon=True
    )
    thread.start()


# ── Batch launch ──────────────────────────────────────────────────────────────

def launch_training_batch(run_ids: list[int], loop: asyncio.AbstractEventLoop):
    """
    Register a queue for every run, then execute all sequentially in one
    background thread.  Each run's queue is populated while it trains so
    stream_logs works exactly like a single run.
    """
    for run_id in run_ids:
        register_log_queue(run_id, loop)

    def run_all():
        for run_id in run_ids:
            _train_worker(run_id, loop)

    thread = threading.Thread(target=run_all, daemon=True)
    thread.start()


# ── Core worker (shared by both paths) ───────────────────────────────────────

def _train_worker(run_id: int, loop: asyncio.AbstractEventLoop | None):
    db = SessionLocal()
    queue = _log_queues.get(run_id)

    def log(msg: str):
        print(f"[run:{run_id}] {msg}", flush=True)
        if queue and loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(queue.put(msg), loop)

    def _signal_end():
        if queue and loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    try:
        run: Run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            _signal_end()
            return

        run.status = "training"
        run.started_at = datetime.now(timezone.utc)
        db.commit()

        dataset = run.dataset
        file_path = Path(dataset.file_path)

        log(f"Loading dataset '{dataset.name}' from {file_path.name}...")
        df, _ = load_and_preprocess(file_path)
        log(f"Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")

        predictor = get_predictor(run.model, run.hyperparams, log_callback=log)
        log(f"Starting {run.model.upper()} training — horizon={run.horizon_days} days")

        result = predictor.fit_predict(df, run.horizon_days)

        db_result = Result(
            run_id=run.id,
            mae=result.mae,
            rmse=result.rmse,
            mape=result.mape,
            forecast_json=result.forecast,
            actual_json=result.actual,
            test_predicted_json=result.test_predicted,
            training_history=result.training_history,
        )
        db.add(db_result)

        run.status = "done"
        run.finished_at = datetime.now(timezone.utc)
        db.commit()

        log(f"✓ Training complete — MAE={result.mae:.4f} RMSE={result.rmse:.4f} MAPE={result.mape:.2f}%")

    except Exception as exc:
        import traceback
        msg = f"✗ Training failed: {exc}\n{traceback.format_exc()}"
        log(msg)
        try:
            run = db.query(Run).filter(Run.id == run_id).first()
            if run:
                run.status = "failed"
                run.error_message = str(exc)
                run.finished_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
    finally:
        _signal_end()
        db.close()
