"""
Energy Predictor CLI
Usage: python -m cli.predict --help
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import typer
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich import box
import pandas as pd

app = typer.Typer(
    name="energy-predictor",
    help="⚡ Energy Consumption Prediction CLI",
    rich_markup_mode="rich",
)
console = Console()

# ── Init DB on import ─────────────────────────────────────────────────────────
from core.database import init_db, SessionLocal, Dataset, Run, Result
from core.preprocessing import load_and_preprocess
from models import get_predictor, DEFAULT_HYPERPARAMS, REGISTRY
from config import settings

init_db()


# ── Dataset commands ──────────────────────────────────────────────────────────

dataset_app = typer.Typer(help="Manage datasets")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("upload")
def dataset_upload(
    file: Path = typer.Argument(..., help="Path to CSV or XLSX file"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Dataset name"),
    description: Optional[str] = typer.Option("", "--desc", "-d", help="Description"),
):
    """Upload and register a dataset."""
    if not file.exists():
        console.print(f"[red]✗ File not found: {file}[/red]")
        raise typer.Exit(1)

    ext = file.suffix.lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        console.print("[red]✗ Only CSV and XLSX files are supported[/red]")
        raise typer.Exit(1)

    ds_name = name or file.stem

    console.print(f"\n[bold cyan]Loading dataset:[/bold cyan] {file.name}")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console, transient=True) as progress:
        progress.add_task("Parsing and preprocessing...", total=None)
        try:
            import shutil, uuid
            file_id = uuid.uuid4().hex[:8]
            dest = settings.DATA_DIR / f"{file_id}_{file.name}"
            shutil.copy2(file, dest)
            df, meta = load_and_preprocess(dest)
        except Exception as e:
            console.print(f"[red]✗ Failed to parse: {e}[/red]")
            raise typer.Exit(1)

    db = SessionLocal()
    dataset = Dataset(
        name=ds_name,
        original_filename=file.name,
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
    db.close()

    console.print(Panel(
        f"[bold green]✓ Dataset registered[/bold green]\n\n"
        f"  ID          : [cyan]{dataset.id}[/cyan]\n"
        f"  Name        : {dataset.name}\n"
        f"  Rows        : {dataset.rows:,}\n"
        f"  Granularity : {dataset.granularity}\n"
        f"  Date range  : {dataset.date_range_start} → {dataset.date_range_end}\n"
        f"  Energy col  : {dataset.energy_column}",
        title="Dataset Upload", border_style="green"
    ))


@dataset_app.command("list")
def dataset_list():
    """List all registered datasets."""
    db = SessionLocal()
    datasets = db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()
    db.close()

    if not datasets:
        console.print("[yellow]No datasets registered yet. Use: energy-predictor dataset upload <file>[/yellow]")
        return

    table = Table(title="Registered Datasets", box=box.ROUNDED, border_style="cyan")
    table.add_column("ID", style="cyan", width=5)
    table.add_column("Name", style="bold white", min_width=20)
    table.add_column("Format", width=7)
    table.add_column("Rows", justify="right", width=10)
    table.add_column("Granularity", width=12)
    table.add_column("Date Range", min_width=35)
    table.add_column("Uploaded", min_width=20)

    for ds in datasets:
        table.add_row(
            str(ds.id),
            ds.name,
            ds.file_format.upper(),
            f"{ds.rows:,}" if ds.rows else "?",
            ds.granularity or "?",
            f"{ds.date_range_start[:10] if ds.date_range_start else '?'} → {ds.date_range_end[:10] if ds.date_range_end else '?'}",
            str(ds.uploaded_at)[:19] if ds.uploaded_at else "?",
        )

    console.print(table)


@dataset_app.command("info")
def dataset_info(dataset_id: int = typer.Argument(..., help="Dataset ID")):
    """Show detailed info about a dataset."""
    db = SessionLocal()
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    db.close()

    if not ds:
        console.print(f"[red]✗ Dataset {dataset_id} not found[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]ID[/bold]           : {ds.id}\n"
        f"[bold]Name[/bold]         : {ds.name}\n"
        f"[bold]File[/bold]         : {ds.original_filename}\n"
        f"[bold]Format[/bold]       : {ds.file_format.upper()}\n"
        f"[bold]Rows[/bold]         : {ds.rows:,}\n"
        f"[bold]Granularity[/bold]  : {ds.granularity}\n"
        f"[bold]Date range[/bold]   : {ds.date_range_start} → {ds.date_range_end}\n"
        f"[bold]Energy col[/bold]   : {ds.energy_column}\n"
        f"[bold]Columns[/bold]      : {', '.join(ds.columns or [])}\n"
        f"[bold]Description[/bold]  : {ds.description or '-'}\n"
        f"[bold]Uploaded[/bold]     : {ds.uploaded_at}",
        title=f"Dataset #{ds.id}", border_style="cyan"
    ))


@dataset_app.command("delete")
def dataset_delete(
    dataset_id: int = typer.Argument(..., help="Dataset ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a dataset and its file."""
    db = SessionLocal()
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not ds:
        console.print(f"[red]✗ Dataset {dataset_id} not found[/red]")
        db.close()
        raise typer.Exit(1)

    if not force:
        confirmed = typer.confirm(f"Delete dataset '{ds.name}' (ID={dataset_id}) and all its runs?")
        if not confirmed:
            console.print("[yellow]Cancelled[/yellow]")
            db.close()
            return

    Path(ds.file_path).unlink(missing_ok=True)
    db.delete(ds)
    db.commit()
    db.close()
    console.print(f"[green]✓ Dataset {dataset_id} deleted[/green]")


@dataset_app.command("preview")
def dataset_preview(
    dataset_id: int = typer.Argument(..., help="Dataset ID"),
    rows: int = typer.Option(10, "--rows", "-r", help="Number of rows to show"),
):
    """Preview dataset rows."""
    db = SessionLocal()
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    db.close()

    if not ds:
        console.print(f"[red]✗ Dataset {dataset_id} not found[/red]")
        raise typer.Exit(1)

    df, _ = load_and_preprocess(ds.file_path)
    preview = df.head(rows).reset_index()

    table = Table(title=f"Preview: {ds.name} (first {rows} rows)", box=box.SIMPLE_HEAVY)
    for col in preview.columns[:8]:  # cap columns shown
        table.add_column(str(col), max_width=18)

    for _, row in preview.iterrows():
        table.add_row(*[str(round(v, 4) if isinstance(v, float) else v)[:18] for v in list(row)[:8]])

    console.print(table)


# ── Run / Training commands ───────────────────────────────────────────────────

run_app = typer.Typer(help="Manage prediction runs")
app.add_typer(run_app, name="run")


@run_app.command("create")
def run_create(
    dataset_id: int = typer.Option(..., "--dataset", "-d", help="Dataset ID"),
    model: str = typer.Option(..., "--model", "-m", help="Model: arima | lstm | bilstm | wavelet_lstm"),
    horizon: int = typer.Option(7, "--horizon", help="Forecast horizon in days"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Run name"),
    params: Optional[str] = typer.Option(None, "--params", "-p",
                                          help='JSON hyperparams e.g. \'{"epochs":30,"lookback":48}\''),
    train: bool = typer.Option(True, "--train/--no-train", help="Start training immediately"),
):
    """Create a prediction run (and optionally train immediately)."""
    if model not in REGISTRY:
        console.print(f"[red]✗ Unknown model '{model}'. Choose: {', '.join(REGISTRY)}[/red]")
        raise typer.Exit(1)

    hyperparams = {}
    if params:
        try:
            hyperparams = json.loads(params)
        except json.JSONDecodeError as e:
            console.print(f"[red]✗ Invalid JSON params: {e}[/red]")
            raise typer.Exit(1)

    merged_params = {**DEFAULT_HYPERPARAMS.get(model, {}), **hyperparams}

    db = SessionLocal()
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        console.print(f"[red]✗ Dataset {dataset_id} not found[/red]")
        db.close()
        raise typer.Exit(1)

    run = Run(
        dataset_id=dataset_id,
        model=model,
        hyperparams=merged_params,
        horizon_days=horizon,
        name=name,
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id
    db.close()

    console.print(f"[green]✓ Run #{run_id} created[/green] — model={model}, dataset={ds.name}, horizon={horizon}d")

    if train:
        _do_train(run_id)


def _do_train(run_id: int):
    """Execute training synchronously (CLI context)."""
    db = SessionLocal()
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        console.print(f"[red]✗ Run {run_id} not found[/red]")
        db.close()
        return

    dataset = run.dataset
    file_path = Path(dataset.file_path)
    db.close()

    console.print(f"\n[bold cyan]Training Run #{run_id}[/bold cyan] — {run.model.upper()} on '{dataset.name}'")
    console.print(f"  Horizon : {run.horizon_days} days")
    console.print(f"  Params  : {json.dumps(run.hyperparams, indent=2)}\n")

    # Update status
    db = SessionLocal()
    r = db.query(Run).filter(Run.id == run_id).first()
    from datetime import datetime, timezone
    r.status = "training"
    r.started_at = datetime.now(timezone.utc)
    db.commit()
    db.close()

    logs = []

    def log(msg: str):
        console.print(f"  [dim]{msg}[/dim]")
        logs.append(msg)

    try:
        df, _ = load_and_preprocess(file_path)
        log(f"Loaded {len(df)} rows")

        predictor = get_predictor(run.model, run.hyperparams, log_callback=log)
        result = predictor.fit_predict(df, run.horizon_days)

        db = SessionLocal()
        r = db.query(Run).filter(Run.id == run_id).first()
        db_result = Result(
            run_id=run_id,
            mae=result.mae,
            rmse=result.rmse,
            mape=result.mape,
            forecast_json=result.forecast,
            actual_json=result.actual,
            training_history=result.training_history,
        )
        db.add(db_result)
        r.status = "done"
        r.finished_at = datetime.now(timezone.utc)
        db.commit()
        db.close()

        console.print(Panel(
            f"[bold green]✓ Training complete[/bold green]\n\n"
            f"  MAE  : [cyan]{result.mae:.4f}[/cyan]\n"
            f"  RMSE : [cyan]{result.rmse:.4f}[/cyan]\n"
            f"  MAPE : [cyan]{result.mape:.2f}%[/cyan]\n\n"
            f"  Forecast points : {len(result.forecast)}\n"
            f"  Run ID          : {run_id}",
            title=f"Results — {run.model.upper()}", border_style="green"
        ))

    except Exception as e:
        import traceback
        console.print(f"[red]✗ Training failed: {e}[/red]")
        console.print(traceback.format_exc())
        db = SessionLocal()
        r = db.query(Run).filter(Run.id == run_id).first()
        r.status = "failed"
        r.error_message = str(e)
        r.finished_at = datetime.now(timezone.utc)
        db.commit()
        db.close()


@run_app.command("train")
def run_train(run_id: int = typer.Argument(..., help="Run ID to train")):
    """Start training for an existing run."""
    _do_train(run_id)


@run_app.command("list")
def run_list(
    dataset_id: Optional[int] = typer.Option(None, "--dataset", "-d", help="Filter by dataset"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Filter by model"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
):
    """List all prediction runs."""
    db = SessionLocal()
    q = db.query(Run)
    if dataset_id:
        q = q.filter(Run.dataset_id == dataset_id)
    if model:
        q = q.filter(Run.model == model)
    if status:
        q = q.filter(Run.status == status)
    runs = q.order_by(Run.id.desc()).all()
    db.close()

    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        return

    table = Table(title="Prediction Runs", box=box.ROUNDED, border_style="cyan")
    table.add_column("ID", style="cyan", width=5)
    table.add_column("Name", min_width=15)
    table.add_column("Model", width=14)
    table.add_column("Dataset", min_width=15)
    table.add_column("Horizon", width=8)
    table.add_column("Status", width=10)
    table.add_column("MAE", justify="right", width=10)
    table.add_column("RMSE", justify="right", width=10)
    table.add_column("MAPE%", justify="right", width=8)

    status_colors = {"done": "green", "training": "yellow", "failed": "red", "pending": "dim"}

    for r in runs:
        color = status_colors.get(r.status, "white")
        mae = rmse = mape = "-"
        if r.result:
            mae = f"{r.result.mae:.4f}"
            rmse = f"{r.result.rmse:.4f}"
            mape = f"{r.result.mape:.2f}"

        table.add_row(
            str(r.id),
            r.name or "-",
            r.model,
            r.dataset.name if r.dataset else "?",
            f"{r.horizon_days}d",
            f"[{color}]{r.status}[/{color}]",
            mae, rmse, mape,
        )

    console.print(table)


@run_app.command("results")
def run_results(
    run_id: int = typer.Argument(..., help="Run ID"),
    show_forecast: bool = typer.Option(False, "--forecast", help="Show forecast data"),
    n: int = typer.Option(24, "--n", help="Number of forecast rows to show"),
):
    """Show results for a completed run."""
    db = SessionLocal()
    run = db.query(Run).filter(Run.id == run_id).first()
    db.close()

    if not run:
        console.print(f"[red]✗ Run {run_id} not found[/red]")
        raise typer.Exit(1)

    if not run.result:
        console.print(f"[yellow]Run {run_id} has no results yet (status: {run.status})[/yellow]")
        raise typer.Exit(1)

    console.print(Panel(
        f"  Model   : [bold]{run.model.upper()}[/bold]\n"
        f"  Dataset : {run.dataset.name if run.dataset else '?'}\n"
        f"  Horizon : {run.horizon_days} days\n"
        f"  Status  : [green]{run.status}[/green]\n\n"
        f"  [bold cyan]MAE [/bold cyan] : {run.result.mae:.4f}\n"
        f"  [bold cyan]RMSE[/bold cyan] : {run.result.rmse:.4f}\n"
        f"  [bold cyan]MAPE[/bold cyan] : {run.result.mape:.2f}%",
        title=f"Run #{run_id} Results", border_style="cyan"
    ))

    if show_forecast and run.result.forecast_json:
        table = Table(title=f"Forecast (first {n} steps)", box=box.SIMPLE)
        table.add_column("Timestamp")
        table.add_column("Predicted (kW)", justify="right")

        for row in run.result.forecast_json[:n]:
            table.add_row(str(row.get("timestamp", "")), f"{row.get('predicted', 0):.4f}")
        console.print(table)


@run_app.command("compare")
def run_compare(
    run_ids: str = typer.Argument(..., help="Comma-separated run IDs e.g. '1,2,3'"),
):
    """Compare metrics across multiple runs."""
    ids = [int(i.strip()) for i in run_ids.split(",") if i.strip().isdigit()]
    db = SessionLocal()
    runs = db.query(Run).filter(Run.id.in_(ids)).all()
    db.close()

    if not runs:
        console.print("[red]✗ No runs found[/red]")
        raise typer.Exit(1)

    table = Table(title="Run Comparison", box=box.ROUNDED, border_style="magenta")
    table.add_column("ID", style="cyan", width=5)
    table.add_column("Name / Model", min_width=20)
    table.add_column("Dataset", min_width=15)
    table.add_column("MAE", justify="right", width=12)
    table.add_column("RMSE", justify="right", width=12)
    table.add_column("MAPE%", justify="right", width=10)
    table.add_column("Status", width=10)

    for r in runs:
        name = r.name or r.model.upper()
        mae = rmse = mape = "[dim]-[/dim]"
        if r.result:
            mae = f"[cyan]{r.result.mae:.4f}[/cyan]"
            rmse = f"[cyan]{r.result.rmse:.4f}[/cyan]"
            mape = f"[cyan]{r.result.mape:.2f}[/cyan]"
        table.add_row(
            str(r.id), name,
            r.dataset.name if r.dataset else "?",
            mae, rmse, mape,
            f"[green]{r.status}[/green]" if r.status == "done" else r.status,
        )

    console.print(table)


@run_app.command("delete")
def run_delete(
    run_id: int = typer.Argument(..., help="Run ID"),
    force: bool = typer.Option(False, "--force", "-f"),
):
    """Delete a run and its results."""
    db = SessionLocal()
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        console.print(f"[red]✗ Run {run_id} not found[/red]")
        db.close()
        raise typer.Exit(1)
    if not force:
        confirmed = typer.confirm(f"Delete run #{run_id} ({run.model})?")
        if not confirmed:
            db.close()
            return
    db.delete(run)
    db.commit()
    db.close()
    console.print(f"[green]✓ Run {run_id} deleted[/green]")


# ── Export command ────────────────────────────────────────────────────────────

@app.command("export")
def export(
    run_id: int = typer.Argument(..., help="Run ID to export"),
    fmt: str = typer.Option("csv", "--format", "-f", help="Output format: csv | excel"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output file path"),
):
    """Export prediction results to CSV or Excel."""
    db = SessionLocal()
    run = db.query(Run).filter(Run.id == run_id).first()
    db.close()

    if not run or not run.result:
        console.print(f"[red]✗ Run {run_id} has no results[/red]")
        raise typer.Exit(1)

    forecast_df = pd.DataFrame(run.result.forecast_json)
    actual_df = pd.DataFrame(run.result.actual_json)

    default_name = f"run_{run_id}_{run.model}"

    if fmt == "csv":
        dest = out or Path(f"{default_name}_forecast.csv")
        forecast_df.to_csv(dest, index=False)
        console.print(f"[green]✓ Saved forecast to {dest}[/green]")
    elif fmt == "excel":
        dest = out or Path(f"{default_name}_results.xlsx")
        with pd.ExcelWriter(dest, engine="openpyxl") as writer:
            forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
            actual_df.to_excel(writer, sheet_name="Actual", index=False)
            pd.DataFrame([{
                "model": run.model, "mae": run.result.mae,
                "rmse": run.result.rmse, "mape": run.result.mape,
            }]).to_excel(writer, sheet_name="Metrics", index=False)
            if run.result.training_history:
                pd.DataFrame(run.result.training_history).to_excel(
                    writer, sheet_name="Training History", index=False)
        console.print(f"[green]✓ Saved results to {dest}[/green]")
    else:
        console.print(f"[red]✗ Unknown format '{fmt}'. Use: csv | excel[/red]")
        raise typer.Exit(1)


# ── Quick-train shortcut ──────────────────────────────────────────────────────

@app.command("predict")
def predict_quick(
    file: Path = typer.Argument(..., help="CSV or XLSX file"),
    model: str = typer.Option("lstm", "--model", "-m", help="arima | lstm | bilstm | wavelet_lstm"),
    horizon: int = typer.Option(7, "--horizon", help="Forecast horizon in days"),
    params: Optional[str] = typer.Option(None, "--params", "-p", help='JSON hyperparams'),
    export_fmt: Optional[str] = typer.Option(None, "--export", help="Export results: csv | excel"),
    name: Optional[str] = typer.Option(None, "--name", "-n"),
):
    """One-shot: upload dataset + train + show results."""
    # Upload
    if not file.exists():
        console.print(f"[red]✗ File not found: {file}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]⚡ Energy Predictor[/bold] — quick predict\n")
    console.print(f"  File    : {file.name}")
    console.print(f"  Model   : {model}")
    console.print(f"  Horizon : {horizon} days\n")

    import shutil, uuid
    from datetime import datetime, timezone

    file_id = uuid.uuid4().hex[:8]
    dest = settings.DATA_DIR / f"{file_id}_{file.name}"
    shutil.copy2(file, dest)

    df, meta = load_and_preprocess(dest)

    db = SessionLocal()
    ds = Dataset(
        name=name or file.stem,
        original_filename=file.name,
        file_path=str(dest),
        file_format=file.suffix.lstrip("."),
        rows=meta["rows"],
        columns=meta["columns"],
        date_range_start=meta.get("date_range_start"),
        date_range_end=meta.get("date_range_end"),
        granularity=meta.get("granularity"),
        energy_column=meta.get("energy_column"),
    )
    db.add(ds)
    db.commit()

    hyperparams = json.loads(params) if params else {}
    merged = {**DEFAULT_HYPERPARAMS.get(model, {}), **hyperparams}

    run = Run(dataset_id=ds.id, model=model, hyperparams=merged,
              horizon_days=horizon, name=name, status="training",
              started_at=datetime.now(timezone.utc))
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id
    db.close()

    def log(msg):
        console.print(f"  [dim]{msg}[/dim]")

    predictor = get_predictor(model, merged, log_callback=log)
    result = predictor.fit_predict(df, horizon)

    db = SessionLocal()
    r = db.query(Run).filter(Run.id == run_id).first()
    db.add(Result(run_id=run_id, mae=result.mae, rmse=result.rmse, mape=result.mape,
                  forecast_json=result.forecast, actual_json=result.actual,
                  test_predicted_json=result.test_predicted,
                  training_history=result.training_history))
    r.status = "done"
    r.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.close()

    console.print(Panel(
        f"  [bold cyan]MAE [/bold cyan]: {result.mae:.4f}\n"
        f"  [bold cyan]RMSE[/bold cyan]: {result.rmse:.4f}\n"
        f"  [bold cyan]MAPE[/bold cyan]: {result.mape:.2f}%\n\n"
        f"  Forecast steps : {len(result.forecast)}\n"
        f"  Run ID         : {run_id}",
        title=f"✓ {model.upper()} Results", border_style="green"
    ))

    if export_fmt:
        ctx = typer.Context(export)
        export(run_id, export_fmt, None)


# ── Models info ───────────────────────────────────────────────────────────────

@app.command("models")
def list_models():
    """Show available models and their default hyperparameters."""
    for model_name, params in DEFAULT_HYPERPARAMS.items():
        console.print(Panel(
            Syntax(json.dumps(params, indent=2), "json", theme="monokai"),
            title=f"[bold cyan]{model_name.upper()}[/bold cyan]",
            border_style="cyan",
        ))


if __name__ == "__main__":
    app()
