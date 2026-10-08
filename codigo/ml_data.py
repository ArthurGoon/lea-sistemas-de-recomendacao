"""MovieLens 100k data loading utilities.

Reference: Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent Systems.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# MovieLens 100k was collected between September 1997 and April 1998.
# Unix epoch timestamps identify instants independently of a local timezone.
EPOCH_OFFSET = 0  # seconds; kept explicit for reproducibility

GENRES: List[str] = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

GENRES_PT: Dict[str, str] = {
    "unknown": "Desconhecido", "Action": "Ação", "Adventure": "Aventura",
    "Animation": "Animação", "Children's": "Infantil", "Comedy": "Comédia",
    "Crime": "Crime", "Documentary": "Documentário", "Drama": "Drama",
    "Fantasy": "Fantasia", "Film-Noir": "Film-Noir", "Horror": "Terror",
    "Musical": "Musical", "Mystery": "Mistério", "Romance": "Romance",
    "Sci-Fi": "Ficção Científica", "Thriller": "Suspense", "War": "Guerra",
    "Western": "Faroeste",
}

OCCUPATIONS: List[str] = [
    "other", "academic/educator", "artist", "clerical/admin",
    "college/grad student", "customer service", "doctor/health care",
    "executive/managerial", "farmer", "homemaker", "K-12 student", "lawyer",
    "programmer", "retired", "sales/marketing", "scientist", "self-employed",
    "technician/engineer", "tradesman/craftsman", "unemployed", "writer",
]


@dataclass
class Dataset:
    """Container for the parsed MovieLens 100k data."""
    ratings: pd.DataFrame      # user, item, rating, timestamp
    users: pd.DataFrame        # user, age, gender, occupation, zip
    items: pd.DataFrame        # item, title, release_date, genres...
    n_users: int
    n_items: int
    global_mean: float
    sparsity: float

    @property
    def genre_matrix(self) -> np.ndarray:
        return self.items[GENRES].to_numpy(dtype=np.float64)

    def train_test_indices(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return train/test DataFrames for the official u1..u5 split."""
        base = os.path.join(DATA_DIR, f"u{fold}.base")
        test = os.path.join(DATA_DIR, f"u{fold}.test")
        tr = _read_ratings(base)
        te = _read_ratings(test)
        return tr, te


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dados", "ml-100k")


def _read_ratings(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None, names=["user", "item", "rating", "timestamp"],
        encoding="latin-1",
    )
    df["user"] = df["user"].astype(int)
    df["item"] = df["item"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    return df


def _read_items(path: str) -> pd.DataFrame:
    # u.item has a trailing empty field after the 19 genre flags
    df = pd.read_csv(
        path, sep="|", header=None, encoding="latin-1",
        names=["item", "title", "release_date", "video_date", "imdb_url"] + GENRES,
    )
    df["item"] = df["item"].astype(int)
    df["release_year"] = pd.to_datetime(
        df["release_date"], format="%d-%b-%Y", errors="coerce"
    ).dt.year
    df["n_genres"] = df[GENRES].sum(axis=1).astype(int)
    return df


def _read_users(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="|", header=None, names=["user", "age", "gender", "occupation", "zip"],
        encoding="latin-1",
    )
    df["user"] = df["user"].astype(int)
    return df


def load_dataset(data_dir: str | None = None) -> Dataset:
    global DATA_DIR
    if data_dir:
        DATA_DIR = data_dir

    ratings = _read_ratings(os.path.join(DATA_DIR, "u.data"))
    users = _read_users(os.path.join(DATA_DIR, "u.user"))
    items = _read_items(os.path.join(DATA_DIR, "u.item"))

    n_users = int(ratings["user"].max())
    n_items = int(ratings["item"].max())
    gmean = float(ratings["rating"].mean())
    sparsity = 1.0 - len(ratings) / (n_users * n_items)

    return Dataset(
        ratings=ratings, users=users, items=items,
        n_users=n_users, n_items=n_items, global_mean=gmean, sparsity=sparsity,
    )


def to_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> np.ndarray:
    """Dense user x item matrix filled with NaN for missing entries."""
    M = np.full((n_users, n_items), np.nan, dtype=np.float64)
    M[df["user"].to_numpy() - 1, df["item"].to_numpy() - 1] = df["rating"].to_numpy()
    return M


def load_official_fold(fold: int, data_dir: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Official MovieLens 100k 5-fold split (u1..u5)."""
    d = data_dir or DATA_DIR
    tr = _read_ratings(os.path.join(d, f"u{fold}.base"))
    te = _read_ratings(os.path.join(d, f"u{fold}.test"))
    return tr, te
