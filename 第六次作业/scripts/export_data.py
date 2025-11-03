from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_datasets, summarise_by_institution


app = typer.Typer(help="Export cleaned discipline records and institution summary to JSON.")


@app.command()
def main(
    data_dir: Path = typer.Option(Path("download"), help="Directory containing raw CSV files."),
    cleaned_path: Path = typer.Option(Path("data/cleaned_records.json"), help="Path to store row-level cleaned data."),
    summary_path: Path = typer.Option(Path("data/institution_summary.json"), help="Path to store institution-level summary."),
) -> None:
    df = load_datasets(data_dir)
    summary = summarise_by_institution(df)

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_json(cleaned_path, orient="records", indent=2, force_ascii=False)
    summary.to_json(summary_path, orient="records", indent=2, force_ascii=False)

    metadata = {
        "record_count": int(len(df)),
        "discipline_count": int(df["Discipline"].nunique()),
        "institution_count": int(df["Institution"].nunique()),
        "summary_count": int(len(summary)),
        "fields": df.columns.tolist(),
    }
    meta_path = cleaned_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    typer.echo(f"Cleaned data exported to {cleaned_path}")
    typer.echo(f"Institution summary exported to {summary_path}")
    typer.echo(f"Metadata exported to {meta_path}")


if __name__ == "__main__":
    app()
