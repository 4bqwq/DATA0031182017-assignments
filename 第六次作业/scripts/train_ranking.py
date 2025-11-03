from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Optional
import sys

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_datasets, load_datasets_from_json
from src.ranking_data import RankingDataModule, RankingDataset, collate_batch
from src.ranking_model import RankingNet


app = typer.Typer()
console = Console()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) for key, tensor in batch.items()}


def compute_metrics(pred_log: torch.Tensor, target_log: torch.Tensor) -> Dict[str, float]:
    preds = torch.expm1(pred_log)
    targets = torch.expm1(target_log)
    mse = torch.mean((preds - targets) ** 2)
    rmse = torch.sqrt(mse)
    mape = torch.mean(torch.abs((preds - targets) / torch.clamp(targets, min=1.0))) * 100
    mae = torch.mean(torch.abs(preds - targets))
    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mape": mape.item(),
        "mae": mae.item(),
    }


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        preds = model(batch["numeric"], batch["categorical"])
        loss = loss_fn(preds, batch["target_log_rank"])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * batch["numeric"].size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            preds = model(batch["numeric"], batch["categorical"])
            preds_list.append(preds.cpu())
            targets_list.append(batch["target_log_rank"].cpu())
    preds_full = torch.cat(preds_list, dim=0)
    targets_full = torch.cat(targets_list, dim=0)
    return compute_metrics(preds_full, targets_full)


@app.command()
def main(
    data_dir: Path = typer.Option(Path("download"), help="Directory with raw discipline CSV files."),
    data_json: Optional[Path] = typer.Option(None, help="Optional path to JSON dataset produced by export_data.py."),
    epochs: int = typer.Option(40, min=1),
    batch_size: int = typer.Option(512, min=16),
    lr: float = typer.Option(3e-4, help="Learning rate."),
    seed: int = typer.Option(42, help="Random seed."),
    model_path: Path = typer.Option(Path("models/ranking_model.pt"), help="Path to store trained model."),
    metrics_path: Path = typer.Option(Path("logs/ranking_metrics.json"), help="Where to save evaluation metrics."),
) -> None:
    set_seed(seed)

    default_json = Path("data/cleaned_records.json")
    dataset_source: str
    if data_json is not None:
        df = load_datasets_from_json(data_json)
        dataset_source = str(data_json)
    elif default_json.exists():
        df = load_datasets_from_json(default_json)
        dataset_source = str(default_json)
    else:
        df = load_datasets(data_dir)
        dataset_source = str(data_dir)

    numeric_cols = ["Documents", "Citations", "CitationsPerPaper", "TopPapers"]
    categorical_cols = ["Discipline", "Country", "Institution"]

    data_module = RankingDataModule(
        df=df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col="Rank",
        seed=seed,
    )
    data_module.setup(train_ratio=0.6, val_ratio=0.2)

    train_dataset = RankingDataset(data_module.train)
    val_dataset = RankingDataset(data_module.val)
    test_dataset = RankingDataset(data_module.test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    embedding_info = data_module.get_embedding_sizes()
    model_config = {
        "hidden_dim": 192,
        "num_residual_blocks": 4,
        "cross_layers": 2,
        "dropout": 0.2,
    }
    model = RankingNet(
        numeric_dim=len(numeric_cols),
        embedding_info=embedding_info,
        **model_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mse = math.inf
    best_state = None

    console.rule("Training Ranking Model")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        console.print(
            f"Epoch {epoch:02d} | TrainLoss(log-MSE): {train_loss:.4f} | "
            f"Val MSE: {val_metrics['mse']:.2f} | Val MAPE: {val_metrics['mape']:.2f}%"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_eval_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    data_state = data_module.to_state_dict()
    split_sizes = {
        "train": len(train_dataset),
        "validation": len(val_dataset),
        "test": len(test_dataset),
    }

    table = Table(title="Ranking Model Metrics")
    table.add_column("Split")
    for key in ["mse", "rmse", "mae", "mape"]:
        table.add_column(key.upper())

    for split_name, metrics in [
        ("Train", train_metrics),
        ("Validation", val_metrics),
        ("Test", test_metrics),
    ]:
        table.add_row(
            split_name,
            f"{metrics['mse']:.2f}",
            f"{metrics['rmse']:.2f}",
            f"{metrics['mae']:.2f}",
            f"{metrics['mape']:.2f}%",
        )
    console.print(table)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "embedding_info": embedding_info,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "seed": seed,
        "data_state": data_state,
        "dataset_source": dataset_source,
        "split_sizes": split_sizes,
        "model_config": model_config,
    }, model_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics,
                "best_val_mse": best_val_mse,
                "epochs": epochs,
                "dataset_source": dataset_source,
                "split_sizes": split_sizes,
                "batch_size": batch_size,
                "learning_rate": lr,
                "model_config": model_config,
            },
            f,
            indent=2,
        )

    console.print(f"Saved model to {model_path}")
    console.print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    app()
