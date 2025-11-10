from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATA_COLUMNS = {
    "Rank": "Rank",
    "Institutions": "Institution",
    "Countries/Regions": "Country",
    "Web of Science Documents": "Documents",
    "Cites": "Citations",
    "Cites/Paper": "CitationsPerPaper",
    "Top Papers": "TopPapers",
}


@dataclass(frozen=True)
class DisciplineDataset:
    discipline: str
    path: Path


def _load_single_csv(entry: DisciplineDataset) -> pd.DataFrame:
    """Load one discipline CSV, normalise column names and attach metadata."""

    df = pd.read_csv(
        entry.path,
        skiprows=1,
        encoding="latin1",
        dtype=str,
    )

    df = df.rename(columns={src: dst for src, dst in DATA_COLUMNS.items() if src in df.columns})

    # Some files keep the rank column unnamed; account for that scenario explicitly.
    if "Rank" not in df.columns and "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Rank"})

    df["Discipline"] = entry.discipline

    numeric_columns = [
        "Rank",
        "Documents",
        "Citations",
        "CitationsPerPaper",
        "TopPapers",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("\"", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Institution" in df.columns:
        df["Institution"] = (
            df["Institution"].astype(str).str.strip().str.replace('"', "", regex=False)
        )
    if "Country" in df.columns:
        df["Country"] = df["Country"].astype(str).str.strip().str.replace('"', "", regex=False)

    filtered = df[df["Rank"].notna()].copy()
    filtered = filtered.reset_index(drop=True)
    return filtered


def load_datasets(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all discipline CSV files found in the data directory."""

    entries: list[DisciplineDataset] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        if csv_path.name.lower().startswith(".~"):
            continue
        discipline = csv_path.stem
        entries.append(DisciplineDataset(discipline=discipline, path=csv_path))

    frames = [_load_single_csv(entry) for entry in entries]
    if not frames:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    desired_columns = [
        "Discipline",
        "Institution",
        "Country",
        "Rank",
        "Documents",
        "Citations",
        "CitationsPerPaper",
        "TopPapers",
    ]
    combined = combined[[col for col in desired_columns if col in combined.columns]]
    combined = combined.dropna(subset=["Rank"]).reset_index(drop=True)
    return combined


def load_datasets_from_json(path: Path) -> pd.DataFrame:
    """Load the cleaned dataset previously exported to JSON."""

    df = pd.read_json(path)
    expected = [
        "Discipline",
        "Institution",
        "Country",
        "Rank",
        "Documents",
        "Citations",
        "CitationsPerPaper",
        "TopPapers",
    ]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"JSON dataset missing required columns: {missing}")

    numeric_cols = ["Rank", "Documents", "Citations", "CitationsPerPaper", "TopPapers"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["Rank"]).reset_index(drop=True)


def summarise_by_institution(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per institution across disciplines for downstream analysis."""

    numeric_columns = [
        col
        for col in ["Rank", "Documents", "Citations", "CitationsPerPaper", "TopPapers"]
        if col in df.columns
    ]

    grouped = df.groupby(["Institution", "Country"], dropna=False)
    agg_mean = grouped[numeric_columns].mean()
    agg_min = grouped["Rank"].min().rename("BestRank")
    agg_median = grouped["Rank"].median().rename("MedianRank")
    counts = grouped.size().rename("DisciplineCount")

    summary = (
        pd.concat([agg_mean, agg_min, agg_median, counts], axis=1)
        .reset_index()
        .sort_values("BestRank")
        .reset_index(drop=True)
    )
    return summary


def list_disciplines(df: pd.DataFrame) -> list[str]:
    return sorted(df["Discipline"].dropna().unique())


def filter_institution(df: pd.DataFrame, name: str) -> pd.DataFrame:
    mask = df["Institution"].str.upper() == name.upper()
    return df[mask].reset_index(drop=True)
