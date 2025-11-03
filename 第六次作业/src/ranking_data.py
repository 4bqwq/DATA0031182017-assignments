from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


@dataclass
class EncodedSplit:
    features_num: torch.Tensor
    features_cat: torch.Tensor
    targets: torch.Tensor


@dataclass
class RankingEncoderState:
    numeric_cols: List[str]
    categorical_cols: List[str]
    target_col: str
    scaler_mean: List[float]
    scaler_scale: List[float]
    category_maps: Dict[str, Dict[str, int]]
    category_cardinalities: Dict[str, int]
    seed: int | None = None

    @staticmethod
    def from_dict(payload: Dict) -> RankingEncoderState:
        return RankingEncoderState(
            numeric_cols=list(payload["numeric_cols"]),
            categorical_cols=list(payload["categorical_cols"]),
            target_col=payload.get("target_col", "Rank"),
            scaler_mean=list(payload["scaler_mean"]),
            scaler_scale=list(payload["scaler_scale"]),
            category_maps={col: {k: int(v) for k, v in mapping.items()} for col, mapping in payload["category_maps"].items()},
            category_cardinalities={col: int(val) for col, val in payload.get("category_cardinalities", {}).items()},
            seed=payload.get("seed"),
        )


class RankingDataset(Dataset):
    def __init__(self, split: EncodedSplit):
        self.split = split

    def __len__(self) -> int:
        return self.split.features_num.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "numeric": self.split.features_num[idx],
            "categorical": self.split.features_cat[idx],
            "target_log_rank": self.split.targets[idx],
        }


class RankingDataModule:
    def __init__(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
        target_col: str = "Rank",
        seed: int = 42,
    ) -> None:
        self.df = df.copy()
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.seed = seed

        self.scaler = StandardScaler()
        self.category_maps: Dict[str, Dict[str, int]] = {}
        self.category_cardinalities: Dict[str, int] = {}

    def setup(self, train_ratio: float = 0.6, val_ratio: float = 0.2) -> None:
        temp_ratio = 1 - train_ratio
        val_share = val_ratio / temp_ratio

        train_df, temp_df = train_test_split(
            self.df,
            train_size=train_ratio,
            shuffle=True,
            random_state=self.seed,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_share,
            shuffle=True,
            random_state=self.seed + 1,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        self._fit_encoders(train_df)

        self.train = self._encode_split(train_df)
        self.val = self._encode_split(val_df)
        self.test = self._encode_split(test_df)

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        self.scaler.fit(df[self.numeric_cols])

        for col in self.categorical_cols:
            values = df[col].fillna("<UNK>").astype(str)
            uniques = sorted(values.unique())
            mapping = {"<UNK>": 0}
            next_index = 1
            for value in uniques:
                if value == "<UNK>":
                    continue
                mapping[value] = next_index
                next_index += 1
            self.category_maps[col] = mapping
            self.category_cardinalities[col] = max(mapping.values()) + 1

    def _encode_split(self, df: pd.DataFrame) -> EncodedSplit:
        enc = self.encode_features(df)

        targets = torch.tensor(np.log1p(df[self.target_col].to_numpy()), dtype=torch.float32).unsqueeze(1)

        return EncodedSplit(
            features_num=enc["numeric"],
            features_cat=enc["categorical"],
            targets=targets,
        )

    def get_embedding_sizes(self, min_dim: int = 8, max_dim: int = 64) -> List[Tuple[int, int]]:
        sizes = []
        for col in self.categorical_cols:
            cardinality = self.category_cardinalities[col]
            dim = min(max_dim, max(min_dim, int(round(cardinality ** 0.25 * 8))))
            sizes.append((cardinality, dim))
        return sizes

    def encode_features(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        numeric_array = self.scaler.transform(df[self.numeric_cols])
        numeric_tensor = torch.tensor(numeric_array, dtype=torch.float32)

        categorical_arrays = []
        for col in self.categorical_cols:
            mapping = self.category_maps[col]
            series = df[col].fillna("<UNK>").astype(str)
            encoded = series.map(lambda v: mapping.get(v, 0)).astype(np.int64).to_numpy()
            categorical_arrays.append(encoded)
        categorical_tensor = torch.tensor(np.stack(categorical_arrays, axis=1), dtype=torch.long)

        return {
            "numeric": numeric_tensor,
            "categorical": categorical_tensor,
        }

    def get_state(self) -> RankingEncoderState:
        if not hasattr(self.scaler, "mean_"):
            raise RuntimeError("Scaler has not been fitted; call setup() first.")
        return RankingEncoderState(
            numeric_cols=list(self.numeric_cols),
            categorical_cols=list(self.categorical_cols),
            target_col=self.target_col,
            scaler_mean=self.scaler.mean_.tolist(),
            scaler_scale=self.scaler.scale_.tolist(),
            category_maps={col: dict(mapping) for col, mapping in self.category_maps.items()},
            category_cardinalities=dict(self.category_cardinalities),
            seed=self.seed,
        )

    @classmethod
    def from_state(cls, state: RankingEncoderState) -> RankingDataModule:
        module = cls(
            df=pd.DataFrame(),
            numeric_cols=state.numeric_cols,
            categorical_cols=state.categorical_cols,
            target_col=state.target_col,
            seed=state.seed or 42,
        )
        module.scaler.mean_ = np.array(state.scaler_mean, dtype=np.float64)
        module.scaler.scale_ = np.array(state.scaler_scale, dtype=np.float64)
        module.scaler.var_ = module.scaler.scale_ ** 2
        module.scaler.n_features_in_ = len(state.numeric_cols)
        module.scaler.feature_names_in_ = np.array(state.numeric_cols, dtype=object)
        module.category_maps = {col: dict(mapping) for col, mapping in state.category_maps.items()}
        module.category_cardinalities = dict(state.category_cardinalities)
        return module

    def to_state_dict(self) -> Dict:
        return asdict(self.get_state())


class RankingEncoder:
    def __init__(self, state: RankingEncoderState) -> None:
        self.state = state
        self.module = RankingDataModule.from_state(state)

    def encode(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        missing_numeric = [col for col in self.state.numeric_cols if col not in df.columns]
        missing_cat = [col for col in self.state.categorical_cols if col not in df.columns]
        if missing_numeric or missing_cat:
            raise ValueError(
                f"Input frame missing columns: numeric={missing_numeric}, categorical={missing_cat}"
            )
        return self.module.encode_features(df)


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    numeric = torch.stack([item["numeric"] for item in batch])
    categorical = torch.stack([item["categorical"] for item in batch])
    targets = torch.stack([item["target_log_rank"] for item in batch])
    return {
        "numeric": numeric,
        "categorical": categorical,
        "target_log_rank": targets,
    }
