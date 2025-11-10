from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import sys

import numpy as np
import pandas as pd
from pytorch_tabular import TabularModel
from rich.console import Console
from rich.table import Table
import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_datasets_from_json
from src.metrics import compute_rank_metrics


app = typer.Typer(help="Predict ranks with a trained PyTorch-Tabular FT-Transformer model.")
console = Console()

TARGET_COL = "Rank"
TARGET_LOG_COL = "LogRank"
BASE_NUMERIC_COLS = ["Documents", "Citations", "CitationsPerPaper", "TopPapers"]
DERIVED_NUMERIC_COLS = [
    "LogDocuments",
    "LogCitations",
    "LogCitationsPerPaper",
    "LogTopPapers",
    "DocsPerTopPaper",
    "CitationsPerDocument",
    "TopPapersRatio",
]


def _load_training_summary(model_dir: Path) -> dict:
    summary_path = model_dir / "training_summary.json"
    if not summary_path.exists():
        raise typer.BadParameter(f"未找到训练摘要文件：{summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _select_rows(df: pd.DataFrame, institutions: List[str], discipline: Optional[str], sample_rows: int) -> pd.DataFrame:
    subset = df
    if institutions:
        mask = subset["Institution"].str.upper().isin({inst.upper() for inst in institutions})
        subset = subset[mask]
    if discipline:
        subset = subset[subset["Discipline"].str.upper() == discipline.upper()]
    if subset.empty:
        raise typer.BadParameter("筛选条件下没有匹配的记录，请调整机构或学科过滤条件。")
    if sample_rows > 0 and sample_rows < len(subset):
        subset = subset.sample(sample_rows, random_state=42).reset_index(drop=True)
    else:
        subset = subset.reset_index(drop=True)
    return subset


def _augment_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Documents" in df.columns:
        df["LogDocuments"] = np.log1p(df["Documents"].clip(lower=0))
        if "Citations" in df.columns:
            df["CitationsPerDocument"] = df["Citations"] / df["Documents"].clip(lower=1)
        else:
            df["CitationsPerDocument"] = 0.0
        if "TopPapers" in df.columns:
            df["TopPapersRatio"] = df["TopPapers"] / df["Documents"].clip(lower=1)
            df["DocsPerTopPaper"] = df["Documents"] / df["TopPapers"].clip(lower=1)
        else:
            df["TopPapersRatio"] = 0.0
            df["DocsPerTopPaper"] = 0.0
    if "Citations" in df.columns:
        df["LogCitations"] = np.log1p(df["Citations"].clip(lower=0))
    else:
        df["LogCitations"] = 0.0
    if "CitationsPerPaper" in df.columns:
        df["LogCitationsPerPaper"] = np.log1p(df["CitationsPerPaper"].clip(lower=0))
    else:
        df["LogCitationsPerPaper"] = 0.0
    if "TopPapers" in df.columns:
        df["LogTopPapers"] = np.log1p(df["TopPapers"].clip(lower=0))
    else:
        df["LogTopPapers"] = 0.0
    for col in DERIVED_NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def _prepare_features(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    prepared = df.copy()
    base_numeric = [col for col in BASE_NUMERIC_COLS if col in prepared.columns]
    for col in base_numeric:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    prepared = prepared.dropna(subset=base_numeric)
    prepared = _augment_numeric_features(prepared)
    for col in categorical_cols:
        prepared[col] = prepared[col].fillna("UNKNOWN").astype(str)
    if TARGET_LOG_COL not in prepared.columns:
        prepared[TARGET_LOG_COL] = np.log1p(prepared.get(TARGET_COL, pd.Series(data=np.nan, index=prepared.index))).fillna(0.0)
    return prepared.reset_index(drop=True)


def _predict(model: TabularModel, df: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(df)
    prediction_col = None
    for col in predictions.columns:
        if col.endswith("_prediction") or col == "prediction":
            prediction_col = col
            break
    if prediction_col is None:
        raise RuntimeError("Model output missing *_prediction column")
    return np.expm1(predictions[prediction_col].to_numpy())


def _format_table(records: pd.DataFrame) -> None:
    table = Table(title="Predicted Ranks")
    for col in records.columns:
        table.add_column(col)
    for _, row in records.iterrows():
        table.add_row(*[
            f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ])
    console.print(table)


@app.command()
def main(
    model_dir: Path = typer.Option(Path("models/ft_transformer"), help="Directory containing the saved model."),
    data_json: Path = typer.Option(Path("data/cleaned_records.json"), help="Cleaned dataset JSON for sampling."),
    output_path: Path = typer.Option(Path("logs/predicted_ranks.json"), help="Where to save prediction JSON."),
    institutions: List[str] = typer.Option([], "--institution", "-i", help="Institution names to filter."),
    discipline: Optional[str] = typer.Option(None, help="Discipline filter."),
    record_json: Optional[Path] = typer.Option(None, help="Custom JSON file containing records to score."),
    sample_rows: int = typer.Option(10, min=1, help="Number of rows to sample when filters are broad."),
) -> None:
    if not model_dir.exists():
        raise typer.BadParameter(f"模型目录不存在：{model_dir}")

    summary = _load_training_summary(model_dir)
    numeric_cols = summary.get("numeric_cols", [])
    categorical_cols = summary.get("categorical_cols", [])

    model = TabularModel.load_model(dir=str(model_dir))
    if hasattr(model, 'model'):
        model.model.eval()
    else:
        model.eval()

    df = load_datasets_from_json(data_json)

    if record_json is not None:
        custom_df = pd.read_json(record_json)
        missing_numeric = [col for col in numeric_cols if col not in custom_df.columns]
        missing_cat = [col for col in categorical_cols if col not in custom_df.columns]
        if missing_numeric or missing_cat:
            raise typer.BadParameter(
                f"自定义文件缺少列：numeric={missing_numeric}, categorical={missing_cat}"
            )
        selected = custom_df.copy()
        if TARGET_COL not in selected.columns:
            selected[TARGET_COL] = np.nan
    else:
        selected = _select_rows(df, institutions=institutions, discipline=discipline, sample_rows=sample_rows)

    features = _prepare_features(selected, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    predictions = _predict(model, features)

    result_df = selected.reset_index(drop=True)
    result_df["PredictedRank"] = predictions
    if TARGET_COL in result_df.columns and result_df[TARGET_COL].notna().any():
        result_df["AbsoluteError"] = np.abs(result_df["PredictedRank"] - result_df[TARGET_COL])
        stats = compute_rank_metrics(result_df[TARGET_COL].fillna(0), result_df["PredictedRank"])
    else:
        result_df["AbsoluteError"] = np.nan
        stats = {}

    display_cols = [
        "Institution",
        "Discipline",
        "Country",
        *numeric_cols,
        "PredictedRank",
    ]
    if TARGET_COL in result_df.columns:
        display_cols.append(TARGET_COL)
    display_cols.append("AbsoluteError")
    records = result_df[[col for col in display_cols if col in result_df.columns]].copy()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "data_json": str(data_json),
        "filters": {
            "institutions": institutions,
            "discipline": discipline,
            "record_json": str(record_json) if record_json else None,
            "sample_rows": sample_rows,
        },
        "stats": stats,
        "records": records.to_dict(orient="records"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    _format_table(records)
    console.print(f"Prediction results saved to {output_path}")


if __name__ == "__main__":
    app()
