from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import pandas as pd
import io

from core.database import get_db, Run

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/{run_id}/csv")
def export_csv(run_id: int, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run or not run.result:
        raise HTTPException(404, "Run or result not found")

    forecast_df = pd.DataFrame(run.result.forecast_json)
    actual_df = pd.DataFrame(run.result.actual_json)

    out = io.StringIO()
    pd.DataFrame({
        "timestamp_forecast": forecast_df.get("timestamp", []),
        "predicted_energy": forecast_df.get("predicted", []),
    }).to_csv(out, index=False)

    out.seek(0)
    filename = f"run_{run_id}_{run.model}_forecast.csv"
    return StreamingResponse(io.BytesIO(out.getvalue().encode()),
                             media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/{run_id}/excel")
def export_excel(run_id: int, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run or not run.result:
        raise HTTPException(404, "Run or result not found")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame(run.result.forecast_json).to_excel(writer, sheet_name="Forecast", index=False)
        pd.DataFrame(run.result.actual_json).to_excel(writer, sheet_name="Actual", index=False)

        metrics = pd.DataFrame([{
            "model": run.model,
            "mae": run.result.mae,
            "rmse": run.result.rmse,
            "mape": run.result.mape,
        }])
        metrics.to_excel(writer, sheet_name="Metrics", index=False)

        if run.result.training_history:
            pd.DataFrame(run.result.training_history).to_excel(writer, sheet_name="Training History", index=False)

    buf.seek(0)
    filename = f"run_{run_id}_{run.model}_results.xlsx"
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})
