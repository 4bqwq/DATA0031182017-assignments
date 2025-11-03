from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_processing import load_datasets, summarise_by_institution


app = typer.Typer()
console = Console()


def build_feature_matrix(summary: pd.DataFrame, features: List[str]) -> np.ndarray:
    scaler = StandardScaler()
    matrix = scaler.fit_transform(summary[features])
    return matrix, scaler


@app.command()
def main(
    data_dir: Path = typer.Option(Path("download"), help="目录包含原始学科 CSV 数据"),
    target_institution: str = typer.Option("EAST CHINA NORMAL UNIVERSITY", help="目标高校名称"),
    n_clusters: int = typer.Option(12, min=2, help="聚类簇数量"),
    min_disciplines: int = typer.Option(5, min=1, help="纳入聚类的最少学科数"),
    top_k: int = typer.Option(10, min=1, help="展示与目标高校最接近的同行数量"),
    output_path: Path = typer.Option(Path("logs/cluster_similar_ecnu.json"), help="输出结果 JSON 路径"),
    assignments_path: Path = typer.Option(Path("logs/cluster_assignments.json"), help="输出全量聚类分配 JSON"),
    figure_path: Path = typer.Option(Path("pictures/cluster-scatter.png"), help="聚类可视化保存路径"),
    seed: int = typer.Option(42, help="随机种子"),
) -> None:
    df = load_datasets(data_dir)
    summary = summarise_by_institution(df)

    filtered = summary[summary["DisciplineCount"] >= min_disciplines].reset_index(drop=True)
    if filtered.empty:
        raise typer.BadParameter("筛选条件过严，缺少可聚类样本。")

    features = [
        "Rank",
        "BestRank",
        "MedianRank",
        "Documents",
        "Citations",
        "CitationsPerPaper",
        "TopPapers",
        "DisciplineCount",
    ]

    feature_matrix, scaler = build_feature_matrix(filtered, features)

    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    labels = model.fit_predict(feature_matrix)
    filtered["Cluster"] = labels
    center_distances = np.linalg.norm(feature_matrix - model.cluster_centers_[labels], axis=1)
    filtered["CenterDistance"] = center_distances

    target_mask = filtered["Institution"].str.upper() == target_institution.upper()
    if not target_mask.any():
        raise typer.BadParameter(f"未在数据中找到高校：{target_institution}")

    target_index = filtered[target_mask].index[0]
    target_cluster = filtered.loc[target_index, "Cluster"]

    distances = pairwise_distances(
        feature_matrix[target_index].reshape(1, -1),
        feature_matrix,
        metric="euclidean",
    )[0]
    filtered["DistanceToTarget"] = distances

    peers = filtered[filtered["Cluster"] == target_cluster].sort_values("DistanceToTarget")
    peers = peers.reset_index(drop=True)

    top_peers = peers.head(top_k)

    table = Table(title=f"与 {target_institution.title()} 类似的高校（簇 {target_cluster}）")
    table.add_column("Rank", justify="right")
    table.add_column("Institution")
    table.add_column("Country")
    table.add_column("Avg Rank", justify="right")
    table.add_column("Best Rank", justify="right")
    table.add_column("Disciplines", justify="right")
    table.add_column("Distance", justify="right")

    for _, row in top_peers.iterrows():
        table.add_row(
            f"{row.name + 1}",
            row["Institution"],
            row["Country"],
            f"{row['Rank']:.1f}",
            f"{row['BestRank']:.0f}",
            str(int(row["DisciplineCount"])),
            f"{row['DistanceToTarget']:.3f}",
        )
    console.print(table)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "target_institution": target_institution,
        "cluster_id": int(target_cluster),
        "parameters": {
            "n_clusters": n_clusters,
            "min_disciplines": min_disciplines,
            "features": features,
        },
        "peers": [
            {
                "institution": row["Institution"],
                "country": row["Country"],
                "avg_rank": float(row["Rank"]),
                "best_rank": float(row["BestRank"]),
                "median_rank": float(row["MedianRank"]),
                "discipline_count": int(row["DisciplineCount"]),
                "distance": float(row["DistanceToTarget"]),
            }
            for _, row in peers.iterrows()
        ],
        "top_{0}".format(top_k): [
            {
                "institution": row["Institution"],
                "country": row["Country"],
                "avg_rank": float(row["Rank"]),
                "best_rank": float(row["BestRank"]),
                "median_rank": float(row["MedianRank"]),
                "discipline_count": int(row["DisciplineCount"]),
                "distance": float(row["DistanceToTarget"]),
            }
            for _, row in top_peers.iterrows()
        ],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    clusters_summary = []
    for cluster_id in range(n_clusters):
        mask = filtered["Cluster"] == cluster_id
        clusters_summary.append(
            {
                "cluster_id": cluster_id,
                "member_count": int(mask.sum()),
                "center": model.cluster_centers_[cluster_id].tolist(),
            }
        )

    assignments_payload = {
        "parameters": {
            "n_clusters": n_clusters,
            "min_disciplines": min_disciplines,
            "features": features,
            "seed": seed,
        },
        "clusters": clusters_summary,
        "records": [
            {
                "institution": row["Institution"],
                "country": row["Country"],
                "avg_rank": float(row["Rank"]),
                "best_rank": float(row["BestRank"]),
                "median_rank": float(row["MedianRank"]),
                "discipline_count": int(row["DisciplineCount"]),
                "cluster": int(row["Cluster"]),
                "center_distance": float(row["CenterDistance"]),
            }
            for _, row in filtered.iterrows()
        ],
    }
    with assignments_path.open("w", encoding="utf-8") as f:
        json.dump(assignments_payload, f, indent=2, ensure_ascii=False)

    # 计算 PCA 以便二维可视化
    pca = PCA(n_components=2, random_state=seed)
    embedding_2d = pca.fit_transform(feature_matrix)
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap="tab20",
        s=25,
        alpha=0.8,
        linewidths=0,
    )
    target_idx = target_index
    ax.scatter(
        embedding_2d[target_idx, 0],
        embedding_2d[target_idx, 1],
        c="red",
        s=120,
        edgecolors="black",
        label=target_institution,
    )
    ax.set_title("ESI Institution Clusters (PCA-2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="Cluster ID")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    console.print(f"结果已保存到 {output_path}")
    console.print(f"全量聚类分配写入 {assignments_path}")
    console.print(f"聚类散点图保存为 {figure_path}")


if __name__ == "__main__":
    app()
