from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker
from datetime import datetime, timezone
from config import settings

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_format = Column(String, nullable=False)  # csv | xlsx
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    rows = Column(Integer)
    columns = Column(JSON)           # list of column names
    date_range_start = Column(String)
    date_range_end = Column(String)
    granularity = Column(String)     # e.g. "5min", "1H"
    energy_column = Column(String)   # detected energy column
    description = Column(Text, nullable=True)

    runs = relationship("Run", back_populates="dataset", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    model = Column(String, nullable=False)          # arima | lstm | bilstm | wavelet_lstm
    hyperparams = Column(JSON, nullable=False)
    horizon_days = Column(Integer, default=7)
    status = Column(String, default="pending")      # pending | training | done | failed
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    name = Column(String, nullable=True)            # optional user-given name

    dataset = relationship("Dataset", back_populates="runs")
    result = relationship("Result", back_populates="run", uselist=False, cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    mae = Column(Float)
    rmse = Column(Float)
    mape = Column(Float)
    forecast_json = Column(JSON)           # [{timestamp, predicted}]  — future horizon
    actual_json = Column(JSON)             # [{timestamp, actual}]      — test set ground truth
    test_predicted_json = Column(JSON, nullable=True)  # [{timestamp, predicted}]  — test set predictions
    training_history = Column(JSON, nullable=True)     # [{epoch, loss, val_loss}]

    run = relationship("Run", back_populates="result")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
