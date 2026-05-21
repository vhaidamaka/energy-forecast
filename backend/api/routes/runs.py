import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from core.database import get_db, Run, Dataset
from core.training_runner import launch_training, launch_training_batch, register_log_queue, unregister_log_queue, get_log_queue
from schemas import RunCreate, BatchRunCreate, RunOut, RunWithResult

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("/", response_model=RunOut, status_code=201)
def create_run(payload: RunCreate, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")

    run = Run(
        dataset_id=payload.dataset_id,
        model=payload.model,
        hyperparams=payload.hyperparams,
        horizon_days=payload.horizon_days,
        name=payload.name,
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


@router.post("/{run_id}/start", response_model=RunOut)
async def start_run(run_id: int, db: Session = Depends(get_db)):
    """
    Must be async so asyncio.get_running_loop() works correctly.
    Python 3.13 / uvloop raises RuntimeError if get_event_loop() is called
    from a worker thread (which is where sync endpoints run).
    """
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(404, "Run not found")
    if run.status not in ("pending", "failed"):
        raise HTTPException(400, f"Run is already in status '{run.status}'")

    # Get the running loop from the async context — always safe here
    loop = asyncio.get_running_loop()
    register_log_queue(run_id, loop)
    launch_training(run_id, loop)

    db.refresh(run)
    return run


@router.get("/{run_id}/logs")
async def stream_logs(run_id: int, db: Session = Depends(get_db)):
    """
    Server-Sent Events stream for live training logs.

    Wait up to 30 s for the queue to appear (batch runs may queue behind
    other models).  While waiting, send SSE comment heartbeats every 5 s
    so nginx / browsers don't close the connection prematurely.
    If training already finished before we connected, return event:done.
    During streaming, send a heartbeat comment every 15 s to keep the
    connection alive through nginx's proxy_read_timeout.
    """
    # ── Wait for queue (with heartbeats so the connection stays open) ─────────
    queue = None
    for tick in range(60):                    # 60 × 0.5 s = 30 s max wait
        queue = get_log_queue(run_id)
        if queue is not None:
            break
        if tick % 10 == 0:                    # heartbeat every 5 s
            pass                              # handled inside StreamingResponse
        await asyncio.sleep(0.5)

    if queue is None:
        # Training finished before we connected — check DB status
        run = db.query(Run).filter(Run.id == run_id).first()
        status = run.status if run else "unknown"
        async def already_done():
            yield f"data: Training {status} (stream connected after completion)\n\n"
            yield "event: done\ndata: complete\n\n"
        return StreamingResponse(
            already_done(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def generator():
        try:
            while True:
                try:
                    # Short timeout so we can send heartbeats regularly
                    msg = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    # Send SSE comment heartbeat — keeps nginx & browser alive
                    yield ": heartbeat\n\n"
                    continue

                if msg is None:
                    yield "event: done\ndata: Training complete\n\n"
                    break
                yield f"data: {msg}\n\n"
        except Exception:
            yield "event: done\ndata: Stream ended\n\n"
        finally:
            unregister_log_queue(run_id)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/batch", response_model=list[RunOut], status_code=201)
async def batch_run(payload: BatchRunCreate, db: Session = Depends(get_db)):
    """
    Create and immediately start one run per requested model,
    all using default hyperparams for that model.
    Models run sequentially in a single background thread to avoid
    memory contention from loading multiple large models at once.
    Returns the created Run objects (status='pending', will transition to
    'training' then 'done'/'failed' asynchronously).
    """
    from models import DEFAULT_HYPERPARAMS, REGISTRY

    ds = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")

    valid_models = list(REGISTRY.keys())
    requested = [m for m in payload.models if m in valid_models]
    if not requested:
        raise HTTPException(400, f"No valid models in request. Available: {valid_models}")

    loop = asyncio.get_running_loop()
    created_runs = []

    for model_name in requested:
        label = payload.name_prefix or "Batch"
        run = Run(
            dataset_id=payload.dataset_id,
            model=model_name,
            hyperparams=DEFAULT_HYPERPARAMS.get(model_name, {}),
            horizon_days=payload.horizon_days,
            name=f"{label} — {model_name.upper()}",
            status="pending",
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        created_runs.append(run)

    # Register a queue for each run and launch sequentially in one thread.
    # This means stream_logs works for batch runs exactly like single runs.
    launch_training_batch([r.id for r in created_runs], loop)

    return created_runs


@router.get("/", response_model=list[RunWithResult])
def list_runs(db: Session = Depends(get_db)):
    from schemas import RunWithResult
    runs = db.query(Run).order_by(Run.id.desc()).all()
    return [RunWithResult.from_run(r) for r in runs]


@router.get("/{run_id}", response_model=RunWithResult)
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(404, "Run not found")
    from schemas import RunWithResult
    return RunWithResult.from_run(run)


@router.delete("/{run_id}")
def delete_run(run_id: int, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(404, "Run not found")
    db.delete(run)
    db.commit()
    return {"message": f"Run {run_id} deleted"}


@router.get("/compare/metrics")
def compare_runs(run_ids: str, db: Session = Depends(get_db)):
    """Compare metrics across multiple runs. Pass run_ids as comma-separated string."""
    ids = [int(i) for i in run_ids.split(",") if i.strip().isdigit()]
    runs = db.query(Run).filter(Run.id.in_(ids)).all()
    results = []
    for r in runs:
        entry = {
            "run_id": r.id,
            "model": r.model,
            "name": r.name or f"Run #{r.id}",
            "status": r.status,
            "hyperparams": r.hyperparams,
        }
        if r.result:
            entry.update({"mae": r.result.mae, "rmse": r.result.rmse, "mape": r.result.mape})
        results.append(entry)
    return results
