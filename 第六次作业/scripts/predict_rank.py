from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import sys

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_datasets_from_json
from src.ranking_data import RankingEncoder, RankingEncoderState
from src.ranking_model import RankingNet


app = typer.Typer(help="Predict discipline-level ranks using the trained model.")
console = Console()


def _select_rows(
    df: pd.DataFrame,
    institutions: List[str],
    discipline: Optional[str],
    sample_rows: int,
) -> pd.DataFrame:
    subset = df
    if institutions:
        mask = subset["Institution"].str.upper().isin({inst.upper() for inst in institutions})
        subset = subset[mask]
    if discipline:
        subset = subset[subset["Discipline"].str.upper() == discipline.upper()]
    if subset.empty:
        raise typer.BadParameter("筛选条件下没有匹配的记录，请检查输入的院校或学科名称。")
    if sample_rows > 0 and sample_rows < len(subset):
        subset = subset.sample(sample_rows, random_state=42).reset_index(drop=True)
    else:
        subset = subset.reset_index(drop=True)
    return subset


@app.command()
def main(
    model_path: Path = typer.Option(Path("models/ranking_model.pt"), help="Trained model checkpoint produced by train_ranking.py."),
    data_json: Path = typer.Option(Path("data/cleaned_records.json"), help="Cleaned dataset JSON exported via export_data.py."),
    output_path: Path = typer.Option(Path("logs/predicted_ranks.json"), help="Path to save prediction results."),
    institutions: List[str] = typer.Option([], "--institution", "-i", help="Institution name(s) to evaluate."),
    discipline: Optional[str] = typer.Option(None, help="Optional discipline filter."),
    record_json: Optional[Path] = typer.Option(None, help="Optional custom JSON file containing records to score."),
    sample_rows: int = typer.Option(10, min=1, help="Number of rows to sample when no explicit record JSON is provided."),
) -> None:
    if not model_path.exists():
        raise typer.BadParameter(f"模型文件不存在：{model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")

    if "data_state" not in checkpoint:
        raise typer.BadParameter("模型 checkpoint 缺少 data_state，请重新训练模型以包含编码信息。")

    data_state = RankingEncoderState.from_dict(checkpoint["data_state"])
    encoder = RankingEncoder(data_state)

    model_config = checkpoint.get(
        "model_config",
        {"hidden_dim": 192, "num_residual_blocks": 4, "cross_layers": 2, "dropout": 0.2},
    )
    model = RankingNet(numeric_dim=len(data_state.numeric_cols), embedding_info=checkpoint["embedding_info"], **model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = load_datasets_from_json(data_json)

    if record_json is not None:
        custom_df = pd.read_json(record_json)
        missing_numeric = [col for col in data_state.numeric_cols if col not in custom_df.columns]
        missing_cat = [col for col in data_state.categorical_cols if col not in custom_df.columns]
        if missing_numeric or missing_cat:
            raise typer.BadParameter(
                f"自定义记录缺失列：numeric={missing_numeric}, categorical={missing_cat}"
            )
        selected = custom_df.copy()
        selected["Country"] = selected.get("Country", "UNKNOWN")
    else:
        selected = _select_rows(df, institutions=institutions, discipline=discipline, sample_rows=sample_rows)

    encoded = encoder.encode(selected)

    with torch.no_grad():
        preds_log = model(encoded["numeric"], encoded["categorical"])
        preds = torch.expm1(preds_log).cpu().numpy().reshape(-1)

    result_df = selected.copy()
    result_df["PredictedRank"] = preds
    if "Rank" in result_df.columns:
        result_df["AbsoluteError"] = np.abs(result_df["PredictedRank"] - result_df["Rank"])
    else:
        result_df["AbsoluteError"] = np.nan

    result_records = result_df[
        [
            "Institution",
            "Discipline",
            "Country",
            "Documents",
            "Citations",
            "CitationsPerPaper",
            "TopPapers",
            "PredictedRank",
            "Rank" if "Rank" in result_df.columns else None,
            "AbsoluteError",
        ]
    ]
    result_records = result_records[[col for col in result_records.columns if col is not None]]

    stats = {}
    if "Rank" in result_df.columns:
        errors = result_df["PredictedRank"] - result_df["Rank"]
        stats = {
            "count": int(len(errors)),
            "mse": float(np.mean(errors ** 2)),
            "mae": float(np.mean(np.abs(errors))),
            "mape": float(np.mean(np.abs(errors) / np.clip(result_df["Rank"], a_min=1, a_max=None)) * 100),
        }

    output_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "data_json": str(data_json),
        "filters": {
            "institutions": institutions,
            "discipline": discipline,
            "record_json": str(record_json) if record_json else None,
            "sample_rows": sample_rows,
        },
        "stats": stats,
        "records": result_records.to_dict(orient="records"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    table = Table(title="Predicted Ranks")
    for col in result_records.columns:
        table.add_column(col)
    for _, row in result_records.iterrows():
        formatted = []
        for col in result_records.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.2f}")
            else:
                formatted.append(str(value))
        table.add_row(*formatted)
    console.print(table)
    console.print(f"Prediction results saved to {output_path}")


if __name__ == "__main__":
    app()
