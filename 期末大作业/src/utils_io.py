import pandas as pd
from pathlib import Path
from .config import DATA_PROCESSED

def load_csv(path: Path | str) -> pd.DataFrame:
    """Load CSV file into a DataFrame."""
    return pd.read_csv(path)

def save_parquet(df: pd.DataFrame, filename: str, directory: Path = DATA_PROCESSED) -> None:
    """Save DataFrame to Parquet format in the specified directory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    df.to_parquet(path, index=False)
    print(f"Saved parquet to {path}")

def load_parquet(filename: str, directory: Path = DATA_PROCESSED) -> pd.DataFrame:
    """Load Parquet file from the specified directory."""
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)
