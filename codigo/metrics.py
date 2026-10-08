"""Evaluation metrics for recommender systems.

Rating-prediction metrics
    RMSE, MAE  (regression-style error on held-out ratings)

Top-N ranking metrics
    Precision@k, Recall@k, nDCG@k, Hit Rate@k, MRR

Beyond-accuracy metrics
    Catalog coverage, novelty (self-information), intra-list diversity (ILD),
    serendipity
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Rating prediction
# ---------------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------
def _topk_from_scores(scores: np.ndarray, seen_mask: np.ndarray, k: int) -> np.ndarray:
    """Top-k item ids (0-based) after masking out items already seen in training."""
    s = scores.copy()
    s[seen_mask] = -np.inf
    idx = np.argsort(-s, kind="stable")
    return idx[np.isfinite(s[idx])][:k]


def expected_random_precision(train_matrix, test_matrix, user_ids, threshold=4.0):
    """Exact mean R_u/C_u, excluding users with no relevant test item.

    This is expected precision, not expected hits: no factor k is required.
    """
    values = []
    for u in user_ids:
        relevant = int(np.sum(test_matrix[u] >= threshold))
        if relevant:
            candidates = int(np.isnan(train_matrix[u]).sum())
            values.append(relevant / candidates)
    return float(np.mean(values)) if values else 0.0


def precision_at_k(recs: Sequence[int], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for r in recs[:k] if r in relevant)
    return hits / k


def recall_at_k(recs: Sequence[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in recs[:k] if r in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recs: Sequence[int], relevant: set, k: int) -> float:
    return 1.0 if any(r in relevant for r in recs[:k]) else 0.0


def ndcg_at_k(recs: Sequence[int], relevant: set, k: int) -> float:
    dcg = 0.0
    for pos, r in enumerate(recs[:k], start=1):
        if r in relevant:
            dcg += 1.0 / np.log2(pos + 1)
    m = min(k, len(relevant))
    if m == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(p + 1) for p in range(1, m + 1))
    return dcg / idcg


def reciprocal_rank(recs: Sequence[int], relevant: set) -> float:
    for pos, r in enumerate(recs, start=1):
        if r in relevant:
            return 1.0 / pos
    return 0.0


# ---------------------------------------------------------------------------
# Full ranking evaluation over a set of users
# ---------------------------------------------------------------------------
def evaluate_ranking(
    score_fn,
    user_ids: np.ndarray,
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    ks: Iterable[int] = (5, 10, 20),
    threshold: float = 4.0,
    item_genres: np.ndarray | None = None,
    item_counts: np.ndarray | None = None,
) -> Dict[str, float]:
    """Evaluate a scoring function with top-N ranking metrics.

    Parameters
    ----------
    score_fn : callable(user_ids) -> np.ndarray (len(user_ids) x n_items)
        Raw (unclipped) preference scores; higher means more recommended.
    user_ids : 0-based user indices to evaluate.
    train_matrix, test_matrix : dense NaN-filled rating matrices.
    threshold : ratings >= threshold are considered relevant.

    Notes
    -----
    The scoring function is called **once** for the whole set of users; the
    top-k lists are then extracted per user from the resulting matrix, which
    avoids recomputing the full score matrix once per user.
    """
    ks = list(ks)
    kmax = max(ks)
    user_ids = np.asarray(user_ids)

    # score every evaluated user once
    S = np.asarray(score_fn(user_ids), dtype=np.float64)

    acc = {f"Precision@{k}": 0.0 for k in ks}
    acc.update({f"Recall@{k}": 0.0 for k in ks})
    acc.update({f"nDCG@{k}": 0.0 for k in ks})
    acc.update({f"HitRate@{k}": 0.0 for k in ks})
    mrr = 0.0
    considered = 0

    rec_lists: List[np.ndarray] = []

    for r, u in enumerate(user_ids):
        seen = ~np.isnan(train_matrix[u])
        relevant = set(np.where(test_matrix[u] >= threshold)[0].tolist())
        recs = _topk_from_scores(S[r], seen, kmax)
        rec_lists.append(recs[:10])
        if not relevant:
            # user has no relevant held-out item: only coverage-type metrics apply
            continue
        considered += 1
        for k in ks:
            acc[f"Precision@{k}"] += precision_at_k(recs, relevant, k)
            acc[f"Recall@{k}"] += recall_at_k(recs, relevant, k)
            acc[f"nDCG@{k}"] += ndcg_at_k(recs, relevant, k)
            acc[f"HitRate@{k}"] += hit_rate_at_k(recs, relevant, k)
        mrr += reciprocal_rank(recs, relevant)

    out: Dict[str, float] = {}
    denom = max(considered, 1)
    for k in acc:
        out[k] = acc[k] / denom
    out["MRR"] = mrr / denom
    out["n_users_evaluated"] = float(considered)

    # ---- beyond-accuracy ----
    if item_counts is not None:
        p = item_counts / item_counts.sum()
        with np.errstate(divide="ignore"):
            selfinfo = -np.log2(np.maximum(p, 1e-12))
        nov = [float(np.mean([selfinfo[j] for j in recs])) for recs in rec_lists]
        out["Novelty"] = float(np.mean(nov))
        covered = set()
        for recs in rec_lists:
            covered.update(recs.tolist())
        out["Coverage@10"] = len(covered) / train_matrix.shape[1]

    if item_genres is not None:
        ild = []
        for recs in rec_lists:
            if len(recs) < 2:
                continue
            G = item_genres[recs]
            sims = []
            for a in range(len(recs)):
                for b in range(a + 1, len(recs)):
                    inter = np.dot(G[a], G[b])
                    union = np.sum((G[a] + G[b]) > 0)
                    sims.append(inter / union if union > 0 else 0.0)
            ild.append(1.0 - float(np.mean(sims)))
        out["ILD@10"] = float(np.mean(ild))

    return out


def precision_at_k_direct(topk_items: np.ndarray, relevant_sets: List[set], k: int) -> float:
    """Precision@k when the recommendation lists are already materialised."""
    vals = []
    for recs, rel in zip(topk_items, relevant_sets):
        if not rel:
            continue
        vals.append(precision_at_k(list(recs[:k]), rel, k))
    return float(np.mean(vals)) if vals else 0.0
