from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from sqlalchemy.orm import Session
from pathlib import Path
import shutil
import uuid

from core.database import get_db, Dataset
from core.preprocessing import load_and_preprocess
from schemas import DatasetOut
from config import settings

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetOut)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    db: Session = Depends(get_db),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(400, "Only CSV and XLSX files are supported")

    # Save file
    file_id = uuid.uuid4().hex[:8]
    dest = settings.DATA_DIR / f"{file_id}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Parse metadata
    try:
        df, meta = load_and_preprocess(dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"Failed to parse dataset: {e}")

    dataset = Dataset(
        name=name,
        original_filename=file.filename,
        file_path=str(dest),
        file_format=ext.lstrip("."),
        rows=meta["rows"],
        columns=meta["columns"],
        date_range_start=meta.get("date_range_start"),
        date_range_end=meta.get("date_range_end"),
        granularity=meta.get("granularity"),
        energy_column=meta.get("energy_column"),
        description=description or None,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.get("/", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return ds


@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: int, rows: int = 200, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    try:
        df, _ = load_and_preprocess(ds.file_path)
        preview = df.head(rows).reset_index()
        preview.columns = [str(c) for c in preview.columns]
        return {"columns": list(preview.columns), "rows": preview.to_dict("records")}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    # Delete the physical file
    Path(ds.file_path).unlink(missing_ok=True)
    db.delete(ds)
    db.commit()
    return {"message": f"Dataset {dataset_id} deleted"}


@router.put("/{dataset_id}")
def update_dataset(dataset_id: int, name: str = None, description: str = None,
                   db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    if name:
        ds.name = name
    if description is not None:
        ds.description = description
    db.commit()
    db.refresh(ds)
    return ds
