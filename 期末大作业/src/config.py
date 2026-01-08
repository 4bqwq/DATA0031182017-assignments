import os
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_RAW = PROJECT_ROOT / "data" / "tweets-4k.csv"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"

# Output Paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Create directories if they don't exist
for path in [DATA_PROCESSED, FIGURES_DIR, TABLES_DIR]:
    path.mkdir(parents=True, exist_ok=True)
