"""
Preprocessing pipeline for energy datasets.
Handles both CSV (5-min, multi-device) and XLSX (hourly, multi-feature) formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re


# ── Public entry point ────────────────────────────────────────────────────────

def load_and_preprocess(file_path: str | Path, resample_freq: str = "1h") -> tuple[pd.DataFrame, dict]:
    """
    Load CSV or XLSX file, clean it, resample to target frequency.
    Returns (processed_df, metadata_dict).
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".csv":
        df, meta = _load_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df, meta = _load_xlsx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    df = _clean_and_resample(df, resample_freq)
    df = _add_time_features(df)

    meta["rows"] = len(df)
    meta["granularity"] = resample_freq
    meta["date_range_start"] = str(df.index.min())
    meta["date_range_end"] = str(df.index.max())

    return df, meta


# ── CSV loader ────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    # Read WITHOUT index_col so ALL columns are visible to _detect_timestamp_col.
    # Using index_col=0 was silently consuming any column named 'timestamp' or
    # any other first column, making it invisible to detection and producing
    # 1970-01-01 epoch timestamps.
    df_raw = pd.read_csv(path)

    # If the very first column is an unnamed integer row index (e.g. from
    # df.to_csv() without index=False), drop it before proceeding.
    first_col = df_raw.columns[0]
    if str(first_col).startswith("Unnamed") or (
        str(first_col) == "" or
        (pd.to_numeric(df_raw[first_col], errors="coerce").notna().all()
         and df_raw[first_col].is_monotonic_increasing
         and df_raw[first_col].iloc[0] in (0, 1))
    ):
        df = df_raw.drop(columns=[first_col])
    else:
        df = df_raw.copy()

    # ── Detect the real timestamp column ─────────────────────────────────────
    ts_col = _detect_timestamp_col(df)

    # Parse timestamps — handle tz-aware strings (e.g. "2023-05-16 16:00:00+0000")
    parsed = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(df[ts_col], errors="coerce")

    df["_ts"] = parsed
    df = df.dropna(subset=["_ts"]).sort_values("_ts")
    df = df.set_index("_ts")

    # Normalise to tz-naive UTC so resampling works cleanly
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    # ── Detect energy column ──────────────────────────────────────────────────
    energy_col = _detect_energy_col(df)

    # Keep energy + voltage (plus any extra numeric sub-meter columns)
    keep = [energy_col]
    if "voltage" in df.columns:
        keep.append("voltage")

    result = df[keep].copy()
    result.rename(columns={energy_col: "energy"}, inplace=True)
    result["energy"] = pd.to_numeric(result["energy"], errors="coerce")

    # Multiple devices → average per timestamp
    result = result.groupby(level=0).mean()

    meta = {
        "columns": [c for c in df.columns if c != "_ts"],
        "energy_column": energy_col,
        "file_format": "csv",
    }
    return result, meta


# ── XLSX loader ───────────────────────────────────────────────────────────────

def _load_xlsx(path: Path) -> tuple[pd.DataFrame, dict]:
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception:
        df = pd.read_excel(path)

    if df.columns[0].startswith("Unnamed"):
        df = pd.read_excel(path, index_col=0, engine="openpyxl")

    # Fix Date/Time column (format: "01/01  01:00:00" — no year)
    if "Date/Time" in df.columns:
        year = datetime.now().year
        df["Date/Time"] = df["Date/Time"].astype(str).str.strip()
        df["Date/Time"] = df["Date/Time"].apply(lambda x: _fix_datetime(x, year))
        df["timestamp"] = pd.to_datetime(df["Date/Time"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp")
    elif isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        df["timestamp"] = pd.to_datetime(df.index, errors="coerce")
        df = df.set_index("timestamp")

    energy_col = _detect_energy_col(df)
    result = df.select_dtypes(include=[np.number]).copy()
    result.rename(columns={energy_col: "energy"}, inplace=True)
    if "energy" not in result.columns:
        result["energy"] = df[energy_col]

    result["energy"] = pd.to_numeric(result["energy"], errors="coerce")

    meta = {
        "columns": list(df.columns),
        "energy_column": energy_col,
        "file_format": "xlsx",
    }
    return result, meta


# ── Cleaning & resampling ─────────────────────────────────────────────────────

def _clean_and_resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols].resample(freq).mean()

    # Fill short gaps by interpolation, longer ones by forward/back fill
    df = df.interpolate(method="time", limit=3)
    df = df.ffill(limit=6).bfill(limit=6)

    # Drop rows where energy is still NaN after filling
    if "energy" in df.columns:
        df = df.dropna(subset=["energy"])
        df["energy"] = df["energy"].clip(lower=0)

    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

# Exact column names that are definitely timestamps, checked before keyword scan
_EXACT_TS_NAMES = {"timestamp", "datetime", "date_time", "time", "date", "ts"}

# Keywords that indicate a timestamp column — ordered from most to least specific
_TS_KEYWORDS = ["timestamp", "datetime", "date_time", " ts", "date", "time"]

# Columns that contain "time" in their name but are NOT timestamps
_TS_EXCLUDE = {"timeslot", "timeofday", "timezone", "uptime", "runtime", "halftime"}


def _detect_timestamp_col(df: pd.DataFrame) -> str:
    cols_lower = {c: c.lower().replace(" ", "").replace("_", "") for c in df.columns}

    # 1. Exact match (case-insensitive, ignoring spaces/underscores)
    for col, cl in cols_lower.items():
        if cl in _EXACT_TS_NAMES:
            return col

    # 2. Keyword scan — skip known non-timestamp columns
    for col, cl in cols_lower.items():
        if cl in _TS_EXCLUDE:
            continue
        if any(kw.replace(" ", "") in cl for kw in _TS_KEYWORDS):
            # Verify it actually parses as datetime on a sample
            try:
                sample = df[col].dropna().iloc[:5]
                pd.to_datetime(sample, errors="raise")
                return col
            except Exception:
                continue

    # 3. Heuristic: find first column whose values parse as datetime
    for col in df.columns:
        try:
            sample = df[col].dropna().iloc[:5]
            pd.to_datetime(sample, errors="raise")
            return col
        except Exception:
            continue

    raise ValueError(
        f"Could not detect a timestamp column. Available columns: {list(df.columns)}"
    )


def _detect_energy_col(df: pd.DataFrame) -> str:
    keywords = ["electricity:facility", "energy", "power", "kwh", "kw", "consumption", "load"]
    for kw in keywords:
        for col in df.columns:
            if kw in col.lower():
                return col
    # Fallback: first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        return numeric_cols[0]
    raise ValueError("Could not detect energy column")


def _fix_datetime(s: str, year: int) -> str:
    """Fix date strings like '01/01  01:00:00' that lack a year."""
    s = re.sub(r"\s+", " ", s.strip())
    if re.match(r"^\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}$", s):
        return f"{year}/{s}"
    if re.match(r"^\d{2}/\d{2}\s+\d{2}:\d{2}$", s):
        return f"{year}/{s}:00"
    return s


def get_dataset_preview(file_path: str | Path, n_rows: int = 100) -> list[dict]:
    """Return first n rows as list of dicts for API preview."""
    df, _ = load_and_preprocess(file_path)
    return df.head(n_rows).reset_index().rename(columns={"index": "timestamp"}).to_dict("records")
