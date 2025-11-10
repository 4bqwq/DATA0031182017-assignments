from __future__ import annotations

import json
import os
os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
import random
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Optional
import sys

import numpy as np
import pandas as pd
import torch
import torch.serialization as torch_serialization
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.category_embedding import CategoryEmbeddingModelConfig
from pytorch_tabular.models.ft_transformer import FTTransformerConfig
from pytorch_tabular.models.tabnet import TabNetModelConfig
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split
import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_datasets, load_datasets_from_json
from src.metrics import compute_rank_metrics


app = typer.Typer(help="Train an FT-Transformer ranking model using PyTorch-Tabular.")
console = Console()

try:
    torch_serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, Any, dict, list, tuple, defaultdict])
except Exception:
    pass

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
NUMERIC_COLS = BASE_NUMERIC_COLS + DERIVED_NUMERIC_COLS
CATEGORICAL_COLS = ["Discipline", "Country", "Institution"]
TARGET_COL = "Rank"
TARGET_LOG_COL = "LogRank"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_dataframe(data_dir: Path, data_json: Optional[Path]) -> tuple[pd.DataFrame, str]:
    default_json = Path("data/cleaned_records.json")
    if data_json is not None:
        df = load_datasets_from_json(data_json)
        return df, str(data_json)
    if default_json.exists():
        df = load_datasets_from_json(default_json)
        return df, str(default_json)
    df = load_datasets(data_dir)
    return df, str(data_dir)


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in BASE_NUMERIC_COLS + [TARGET_COL]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.dropna(subset=[TARGET_COL])
    for col in BASE_NUMERIC_COLS:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    cleaned["LogDocuments"] = np.log1p(cleaned["Documents"].clip(lower=0))
    cleaned["LogCitations"] = np.log1p(cleaned["Citations"].clip(lower=0))
    cleaned["LogCitationsPerPaper"] = np.log1p(cleaned["CitationsPerPaper"].clip(lower=0))
    cleaned["LogTopPapers"] = np.log1p(cleaned["TopPapers"].clip(lower=0))
    cleaned["DocsPerTopPaper"] = cleaned["Documents"] / cleaned["TopPapers"].clip(lower=1)
    cleaned["CitationsPerDocument"] = cleaned["Citations"] / cleaned["Documents"].clip(lower=1)
    cleaned["TopPapersRatio"] = cleaned["TopPapers"] / cleaned["Documents"].clip(lower=1)
    for col in DERIVED_NUMERIC_COLS:
        cleaned[col] = cleaned[col].replace([np.inf, -np.inf], np.nan).fillna(cleaned[col].median())
    for col in CATEGORICAL_COLS:
        cleaned[col] = cleaned[col].fillna("UNKNOWN").astype(str)
    cleaned[TARGET_LOG_COL] = np.log1p(cleaned[TARGET_COL].clip(lower=1))
    return cleaned.reset_index(drop=True)


def _split_dataframe(df: pd.DataFrame, seed: int, train_ratio: float = 0.6, val_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temp_ratio = 1 - train_ratio
    val_share = val_ratio / temp_ratio

    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        shuffle=True,
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_share,
        shuffle=True,
        random_state=seed + 1,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _build_model(model_type: str, epochs: int, batch_size: int, lr: float, target_range: list[float] | None = None) -> TabularModel:
    data_config = DataConfig(
        target=[TARGET_LOG_COL],
        continuous_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        normalize_continuous_features=True,
        handle_unknown_categories=True,
        handle_missing_values=True,
    )

    if model_type == "ft":
        model_config = FTTransformerConfig(
            task="regression",
            learning_rate=lr,
            num_heads=8,
            num_attn_blocks=4,
            input_embed_dim=64,
            attn_dropout=0.2,
            add_norm_dropout=0.1,
            ff_dropout=0.2,
            embedding_dropout=0.05,
            target_range=target_range,
        )
    elif model_type == "tabnet":
        model_config = TabNetModelConfig(
            task="regression",
            learning_rate=lr,
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            virtual_batch_size=256,
            target_range=target_range,
        )
    elif model_type == "embed":
        model_config = CategoryEmbeddingModelConfig(
            task="regression",
            learning_rate=lr,
            layers="1024-512-256-128",
            activation="GELU",
            dropout=0.1,
            target_range=target_range,
        )
    else:
        raise typer.BadParameter("model_type must be 'ft', 'tabnet', or 'embed'")

    trainer_config = TrainerConfig(
        batch_size=batch_size,
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        progress_bar="none",
        checkpoints=None,
        early_stopping=None,
        load_best=False,
    )

    optimizer_config = OptimizerConfig(
        optimizer="AdamW",
        optimizer_params={"weight_decay": 1e-4},
    )

    return TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )


def _evaluate_split(model: TabularModel, df: pd.DataFrame) -> Dict[str, float]:
    predictions = model.predict(df)
    prediction_col = None
    for col in predictions.columns:
        if col.endswith("_prediction") or col == "prediction":
            prediction_col = col
            break
    if prediction_col is None:
        raise RuntimeError("Model prediction output missing *_prediction column")
    pred_log = predictions[prediction_col].to_numpy()
    pred_rank = np.expm1(pred_log)
    true_rank = df[TARGET_COL].to_numpy()
    return compute_rank_metrics(true_rank, pred_rank)


def _print_metrics_table(metrics: Dict[str, Dict[str, float]]) -> None:
    table = Table(title="FT-Transformer Ranking Metrics")
    table.add_column("Split")
    for key in ["mse", "rmse", "mae", "mape"]:
        table.add_column(key.upper())
    for split_name, split_metrics in metrics.items():
        table.add_row(
            split_name.title(),
            f"{split_metrics['mse']:.2f}",
            f"{split_metrics['rmse']:.2f}",
            f"{split_metrics['mae']:.2f}",
            f"{split_metrics['mape']:.2f}%",
        )
    console.print(table)


@app.command()
def main(
    data_dir: Path = typer.Option(Path("download"), help="Directory with raw CSV files."),
    data_json: Optional[Path] = typer.Option(None, help="Optional cleaned JSON dataset."),
    epochs: int = typer.Option(40, min=1),
    batch_size: int = typer.Option(512, min=32),
    lr: float = typer.Option(3e-4, help="Learning rate."),
    model_type: str = typer.Option("ft", help="Model backbone to use: 'ft', 'tabnet', or 'embed'."),
    seed: int = typer.Option(42, help="Random seed."),
    model_dir: Path = typer.Option(Path("models/ft_transformer"), help="Directory to store the trained model."),
    metrics_path: Path = typer.Option(Path("logs/ranking_metrics.json"), help="Path to save evaluation metrics."),
) -> None:
    set_seed(seed)

    raw_df, dataset_source = _prepare_dataframe(data_dir=data_dir, data_json=data_json)
    df = _clean_features(raw_df)
    train_df, val_df, test_df = _split_dataframe(df, seed=seed)

    max_rank = float(df[TARGET_COL].max())
    log_max = float(np.log1p(max_rank))
    target_range = [[0.0, max(0.0, log_max)]]

    model = _build_model(
        model_type=model_type.lower(),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        target_range=target_range,
    )
    console.rule("Training FT-Transformer Model")
    model.fit(train=train_df, validation=val_df)

    metrics = {
        "train": _evaluate_split(model, train_df),
        "validation": _evaluate_split(model, val_df),
        "test": _evaluate_split(model, test_df),
    }
    _print_metrics_table(metrics)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir))

    summary = {
        "dataset_source": dataset_source,
        "model_type": model_type.lower(),
        "target_range": target_range,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "target": TARGET_COL,
        "splits": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "metrics": metrics,
    }
    with (model_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    console.print(f"Model saved to {model_dir}")
    console.print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    app()
