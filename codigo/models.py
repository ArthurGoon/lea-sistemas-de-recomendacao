"""Recommender models implemented from first principles.

The formulations follow the notation used in the course slides of
Laboratorio de Estatistica Aplicada (Victor Coscrato):

  * Memory-based collaborative filtering (user- and item-based KNN)
  * Matrix factorization (FunkSVD with biases, trained by SGD)
  * Content-based filtering on movie genres

Models expose predict() for rating estimates and score_matrix() for ranking.
User KNN additionally exposes rating_score_matrix(), preserving normalization
when ranking by predicted rating instead of weighted agreement.

The SGD loop of the matrix factorization is optionally JIT-compiled with numba;
when numba is unavailable the identical pure-NumPy loop is used instead, so the
the reference experiment uses the serial JIT kernel and fixed seeds.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

try:  # optional accelerator
    from numba import njit, prange  # type: ignore
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*a, **k):  # type: ignore
        def deco(f):
            return f
        return a[0] if a and callable(a[0]) else deco

    def prange(*a):  # type: ignore
        return range(*a)

CLIP_MIN, CLIP_MAX = 1.0, 5.0


# ---------------------------------------------------------------------------
# JIT kernel: one SGD epoch over the observed ratings
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True, parallel=True)
def _sgd_epoch_par(order, indptr, indices, item_ids, ratings, b_u, b_i, U, V, mu,
                   lr, reg, reg_bias):
    """One epoch of stochastic gradient descent, parallelised over users.

    Ratings are grouped by user (CSR layout) so that threads never touch the
    same ``b_u``/``U`` row simultaneously. Item parameters *are* shared, so the
    updates to ``V`` and ``b_i`` are unsynchronized, following the
    Hogwild! scheme of Recht et al. (2011). Because floating-point addition is
    not associative, the exact result depends on the thread schedule; pass
    ``parallel=False`` for bitwise reproducibility.
    """
    n_users = indptr.shape[0] - 1
    d = U.shape[1]
    for pos in prange(n_users):
        u = order[pos]
        for k in range(indptr[u], indptr[u + 1]):
            i = item_ids[indices[k]]
            pred = mu + b_u[u] + b_i[i]
            for f in range(d):
                pred += U[u, f] * V[i, f]
            e = ratings[indices[k]] - pred
            b_u[u] += lr * (e - reg_bias * b_u[u])
            for f in range(d):
                Uuf = U[u, f]
                Vif = V[i, f]
                U[u, f] += lr * (e * Vif - reg * Uuf)
                V[i, f] += lr * (e * Uuf - reg * Vif)
            b_i[i] += lr * (e - reg_bias * b_i[i])


@njit(cache=True, fastmath=True, parallel=False)
def _sgd_epoch_serial(order, indptr, indices, item_ids, ratings, b_u, b_i, U, V,
                      mu, lr, reg, reg_bias):
    """Serial (deterministic) reference implementation of one SGD epoch."""
    n_users = indptr.shape[0] - 1
    d = U.shape[1]
    for pos in range(n_users):
        u = order[pos]
        for k in range(indptr[u], indptr[u + 1]):
            i = item_ids[indices[k]]
            pred = mu + b_u[u] + b_i[i]
            for f in range(d):
                pred += U[u, f] * V[i, f]
            e = ratings[indices[k]] - pred
            b_u[u] += lr * (e - reg_bias * b_u[u])
            for f in range(d):
                Uuf = U[u, f]
                Vif = V[i, f]
                U[u, f] += lr * (e * Vif - reg * Uuf)
                V[i, f] += lr * (e * Uuf - reg * Vif)
            b_i[i] += lr * (e - reg_bias * b_i[i])


def _csr_by_user(users: np.ndarray, ratings: np.ndarray, n_users: int):
    """Group ratings by user in CSR layout.

    ``indices`` stores, for each user slot, the original rating position ``k``;
    the kernel reads the item id from ``item_ids[k]`` and the value from
    ``ratings[k]`` so that the SGD order follows a permutation of users.
    """
    counts = np.bincount(users, minlength=n_users)
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.empty(len(users), dtype=np.int64)
    cursor = indptr[:-1].copy()
    for k in range(len(users)):
        u = users[k]
        c = cursor[u]
        indices[c] = k
        cursor[u] = c + 1
    return indptr, np.ascontiguousarray(indices)


def build_matrix(users: np.ndarray, items: np.ndarray, values: np.ndarray,
                 n_users: int, n_items: int) -> np.ndarray:
    M = np.full((n_users, n_items), np.nan, dtype=np.float64)
    M[users - 1, items - 1] = values
    return M


def user_means(M: np.ndarray, gmean: float, shrink: float = 0.0) -> np.ndarray:
    """User means. With shrink>0 applies shrinkage towards the global mean.

    b_u = sum(r_ui - mu) / (n_u + shrink)
    """
    n = np.sum(~np.isnan(M), axis=1)
    s = np.nansum(M, axis=1)
    with np.errstate(invalid="ignore"):
        means = np.where(n > 0, (s + shrink * gmean) / (n + shrink), gmean)
    return means, n


# ---------------------------------------------------------------------------
# Memory-based collaborative filtering
# ---------------------------------------------------------------------------
class KNNRecommender:
    """Memory-based CF as KNN imputation of the rating matrix.

    Parameters
    ----------
    kind : {'user', 'item'}
        User-based imputes rows of R; item-based imputes columns.
    k : int
        Number of neighbours N(u) / N(i) considered.
    similarity : {'cosine', 'pearson'}
    shrink : float
        Shrinkage applied to means (mean-centering step).
    sim_shrink : float
        Encolhimento por suporte (*significance weighting* de Herlocker et al.,
        1999). A similaridade e multiplicada por
        ``min(n_co, sim_shrink) / sim_shrink``, em que ``n_co`` e o numero de
        itens avaliados em comum. Atenua similaridades com pouco suporte.
        O cosseno usa as normas das linhas completas: um unico item em comum
        nao implica similaridade 1. Com 0, nada muda.
    """

    def __init__(self, kind: str = "user", k: int = 40, similarity: str = "cosine",
                 shrink: float = 0.0, sim_shrink: float = 0.0):
        assert kind in ("user", "item")
        self.kind = kind
        self.k = k
        self.similarity = similarity
        self.shrink = shrink
        self.sim_shrink = sim_shrink
        self.S: Optional[np.ndarray] = None
        self.mu_u: Optional[np.ndarray] = None
        self.mu_i: Optional[np.ndarray] = None
        self.gmean: float = 0.0
        self.M: Optional[np.ndarray] = None
        self.n_users = 0
        self.n_items = 0
        self._cached_idx: Optional[np.ndarray] = None
        self._cached_w: Optional[np.ndarray] = None

    # -- fitting -----------------------------------------------------------
    def fit(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray,
            n_users: int, n_items: int) -> "KNNRecommender":
        M = build_matrix(users, items, ratings, n_users, n_items)
        self.M = M
        self.n_users, self.n_items = n_users, n_items
        self.gmean = float(np.nanmean(M))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mu_u, self.n_u = user_means(M, self.gmean, self.shrink)
            mu_i_all = np.nanmean(M, axis=0)
            self.n_i = (~np.isnan(M)).sum(axis=0)
        self.mu_i = np.where(np.isnan(mu_i_all), self.gmean, mu_i_all)

        if self.kind == "user":
            # Centre each row by its user mean: R~_ui = r_ui - mu - b_u
            centred = M - self.mu_u[:, None]
        else:
            # Adjusted cosine: centre each column of R (i.e. each item's
            # ratings) by the corresponding user mean.
            centred = M - self.mu_u[:, None]

        X = np.nan_to_num(centred, nan=0.0)
        mask = ~np.isnan(centred)
        if self.kind == "item":
            X = X.T                 # items x users
            mask = mask.T

        if self.similarity == "cosine":
            norms = np.linalg.norm(X, axis=1)
            norms[norms == 0] = np.inf
            S = (X / norms[:, None]) @ (X / norms[:, None]).T
        elif self.similarity == "pearson":
            S = self._pearson(X, mask=mask)
        else:
            raise ValueError(self.similarity)

        S = np.nan_to_num(S, nan=0.0)
        np.fill_diagonal(S, 0.0)

        if self.sim_shrink > 0:
            # peso por suporte: quantos itens os dois avaliaram em comum
            co = (mask.astype(np.float64) @ mask.astype(np.float64).T)
            peso = np.minimum(co, self.sim_shrink) / self.sim_shrink
            S = S * peso

        # only strictly positive similarities are meaningful neighbours
        S[S <= 0] = 0.0
        self.S = S

        # keep only the top-k neighbours per row
        kk = min(self.k, S.shape[1] - 1)
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        w = np.take_along_axis(S, idx, axis=1)
        order = np.argsort(-w, axis=1)
        self._cached_idx = np.take_along_axis(idx, order, axis=1)
        self._cached_w = np.take_along_axis(w, order, axis=1)
        return self

    @staticmethod
    def _pearson(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        S = np.zeros((n, n))
        for i in range(n):
            mi = mask[i]
            Xi = np.where(mi, X[i], 0.0)
            for j in range(i + 1, n):
                mj = mask[j]
                both = mi & mj
                c = int(both.sum())
                if c < 2:
                    continue
                a = X[i][both]
                b = X[j][both]
                a = a - a.mean()
                b = b - b.mean()
                den = np.linalg.norm(a) * np.linalg.norm(b)
                if den > 0:
                    S[i, j] = S[j, i] = float(a @ b / den)
        return S

    # -- prediction --------------------------------------------------------
    def _neighbour_terms(self, row: int):
        idx = self._cached_idx[row]
        w = self._cached_w[row]
        valid = np.isfinite(w) & (w > 0)
        return idx[valid], w[valid]

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Vectorised predictions for arrays of (user, item) pairs (1-based)."""
        assert self.M is not None
        u = users - 1
        i = items - 1
        out = np.empty(len(u), dtype=np.float64)

        if self.kind == "user":
            for n, (uu, ii) in enumerate(zip(u, i)):
                nb, w = self._neighbour_terms(uu)
                if nb.size == 0:
                    out[n] = self.mu_u[uu]
                    continue
                # only neighbours that actually rated item ii
                r = self.M[nb, ii] - self.mu_u[nb] - 0.0
                m = ~np.isnan(r)
                if not m.any():
                    out[n] = self.mu_u[uu]
                    continue
                ww = w[m]
                den = np.abs(ww).sum()
                out[n] = self.mu_u[uu] + (0.0 if den == 0 else float(ww @ r[m]) / den)
        else:
            for n, (uu, ii) in enumerate(zip(u, i)):
                nb, w = self._neighbour_terms(ii)
                if nb.size == 0:
                    out[n] = self.mu_u[uu]
                    continue
                # centring is by the *user* mean (adjusted cosine), so the
                # same scalar shift applies to every neighbour
                r = self.M[uu, nb] - self.mu_u[uu]
                m = ~np.isnan(r)
                if not m.any():
                    out[n] = self.mu_u[uu]
                    continue
                # weighted average of the user's *centred* ratings on the
                # neighbours; the denominator uses every neighbour so that
                # items the user has not rated simply dilute the estimate
                est = float((w[m] @ r[m]) / w.sum())
                out[n] = self.mu_u[uu] + est

        return np.clip(out, CLIP_MIN, CLIP_MAX)

    def rating_score_matrix(self, user_ids: np.ndarray) -> np.ndarray:
        """All user-based KNN rating estimates, for ranking by predicted rating.

        Exactly the same normalization and fallback as predict(). Ranking
        retains raw scores to avoid ties caused by clipping at the scale limits.
        """
        if self.kind != "user":
            raise ValueError("rating_score_matrix requires user-based KNN")
        scores = np.empty((len(user_ids), self.n_items), dtype=np.float64)
        for row, uu in enumerate(user_ids):
            nb, w = self._neighbour_terms(uu)
            sub = self.M[nb] - self.mu_u[nb, None]
            observed = np.isfinite(sub)
            num = np.nansum(sub * w[:, None], axis=0)
            den = np.sum(observed * w[:, None], axis=0)
            scores[row] = self.mu_u[uu] + np.divide(
                num, den, out=np.zeros_like(num), where=den > 0)
        return scores

    def score_matrix(self, user_ids: np.ndarray) -> np.ndarray:
        """Full recommendation scores for a subset of users (0-based ids).

        Returns an array of shape (len(user_ids), n_items).

        The score implemented here is the **unnormalised agreement**

        .. math::
            \\text{score}(u,i) = \\sum_{v \\in N_k(u,i)}
            \\mathrm{sim}(u,v)\\,\\tilde r_{vi}

        rather than the normalised ratio :math:`\\sum w\\tilde r / \\sum |w|`
        used by :meth:`predict`. Normalization changes the ordering: a large
        positive deviation from one neighbour can produce a high predicted
        rating, whereas the sum also reflects accumulated support. Neither
        rule is universally better. Both are evaluated in the final report.
        """
        assert self.M is not None
        scores = np.empty((len(user_ids), self.n_items), dtype=np.float64)

        if self.kind == "user":
            for r, uu in enumerate(user_ids):
                nb, w = self._neighbour_terms(uu)
                if nb.size == 0:
                    scores[r] = 0.0
                    continue
                sub = self.M[nb, :] - self.mu_u[nb, None]
                valid = ~np.isnan(sub)
                scores[r] = np.nansum(np.where(valid, sub * w[:, None], 0.0), axis=0)
        else:
            # item-based: score(u, i) = sum_j S[i,j] r~_uj, where S holds the k
            # neighbours per row exactly as used by ``predict``.
            S = np.zeros((self.n_items, self.n_items), dtype=np.float64)
            np.put_along_axis(S, self._cached_idx, self._cached_w, axis=1)
            for r, uu in enumerate(user_ids):
                ru = self.M[uu, :] - self.mu_u[uu]
                obs = np.isfinite(ru)
                if not obs.any():
                    scores[r] = 0.0
                    continue
                scores[r] = S[:, obs] @ ru[obs]

        return scores


# ---------------------------------------------------------------------------
# Matrix factorization: FunkSVD with biases
# ---------------------------------------------------------------------------
class FunkSVD:
    r"""Bias-aware matrix factorization.

    .. math::
        \hat r_{ui} = \mu + b_u + b_i + U_u^T V_i

    Minimises squared error with per-observation regularization by SGD:

    .. math::
        \min_{U,V,b}\ \sum_{(u,i)\in K}\left[(r_{ui}-\hat r_{ui})^2
        + \lambda(\|U_u\|^2+\|V_i\|^2+b_u^2+b_i^2)\right]
    """

    def __init__(self, n_factors: int = 50, n_epochs: int = 40, lr: float = 0.005,
                 reg: float = 0.05, reg_bias: float | None = None,
                 random_state: int = 42, verbose: bool = False,
                 early_stopping: bool = True, patience: int = 5,
                 min_delta: float = 1e-5,
                 init_std: float = 0.05, parallel: bool = False):
        """FunkSVD com vieses, treinado por SGD.

        Parameters
        ----------
        parallel : bool, default False
            Se True, usa o kernel paralelizado com atualizacoes atomicas
            (*Hogwild!*). E mais rapido, mas como a soma em ponto flutuante nao
            e associativa o resultado depende do escalonamento das threads e
            varia na quarta casa decimal entre execucoes. O padrao e False,
            para que os resultados sejam reproduziveis bit a bit; True e
            indicado apenas para varreduras exploratorias de hiperparametros.
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.reg_bias = reg if reg_bias is None else reg_bias
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.init_std = init_std
        self.parallel = parallel and _HAVE_NUMBA
        self.debug_hook = None
        self.history_: list[tuple[int, float, float | None]] = []
        self.best_epoch_: int | None = None

    def fit(self, users, items, ratings, n_users, n_items,
            val: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None):
        rng = np.random.default_rng(self.random_state)
        self.mu = float(ratings.mean())
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.U = rng.normal(0, self.init_std, (n_users, self.n_factors))
        self.V = rng.normal(0, self.init_std, (n_items, self.n_factors))
        self.n_users, self.n_items = n_users, n_items

        u = users - 1
        i = items - 1
        r = ratings.astype(np.float64)
        n = len(r)
        # No learned interaction is available for an unseen entity. Its random
        # initialization must not be mistaken for evidence at prediction time.
        self.U[np.bincount(u, minlength=n_users) == 0] = 0.0
        self.V[np.bincount(i, minlength=n_items) == 0] = 0.0
        self.history_ = []

        # CSR-by-user layout for the JIT kernel; ``order`` keeps the SGD order
        # stochastic across epochs.
        order = np.arange(n_users, dtype=np.int64)
        indptr, k_idx = _csr_by_user(u, r, n_users)

        best_rmse = np.inf
        best_state = None
        bad = 0

        # O early stopping exige um conjunto de validacao: monitorar o erro de
        # TREINO nao funciona, porque ele decresce (ou fica achatado pelo
        # recorte em [1,5]) e a paciencia dispara antes de o modelo aprender.
        # Sem validacao, treinamos o numero de epocas pedido.
        use_early_stopping = self.early_stopping and val is not None

        for epoch in range(1, self.n_epochs + 1):
            rng.shuffle(order)
            kernel = _sgd_epoch_par if self.parallel else _sgd_epoch_serial
            kernel(order, indptr, k_idx, i.astype(np.int64), r,
                   self.b_u, self.b_i, self.U, self.V, self.mu,
                   self.lr, self.reg, self.reg_bias)

            tr_rmse = self._rmse_raw(u, i, r)
            va_rmse = None
            if val is not None:
                vu, vi, vr = val
                # val chega com identificadores 1-based (como nos arquivos do
                # MovieLens); converte-se uma unica vez aqui
                va_rmse = self._rmse(np.asarray(vu) - 1, np.asarray(vi) - 1,
                                     vr.astype(np.float64))
            self.history_.append((epoch, tr_rmse, va_rmse))
            if self.verbose:
                print(f"  epoch {epoch:3d}  train RMSE={tr_rmse:.4f}"
                      + (f"  val RMSE={va_rmse:.4f}" if va_rmse is not None else ""))

            if use_early_stopping:
                if va_rmse < best_rmse - self.min_delta:
                    best_rmse = va_rmse
                    self.best_epoch_ = epoch
                    best_state = (self.b_u.copy(), self.b_i.copy(),
                                  self.U.copy(), self.V.copy())
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            self.b_u, self.b_i, self.U, self.V = best_state
        return self

    # -- metricas internas -------------------------------------------------
    # Convencao: ``_rmse`` e ``_rmse_raw`` recebem indices **0-based**, iguais
    # aos usados pelo kernel SGD e por ``raw_predict``. A conversao a partir dos
    # identificadores 1-based do MovieLens e feita UMA vez, em ``fit``.
    def _rmse(self, u, i, r):
        """RMSE com as predicoes projetadas em [1, 5].

        E o valor efetivamente reportado como desempenho, e o criterio de
        \\textit{early stopping}.
        """
        u = np.asarray(u, dtype=np.int64)
        i = np.asarray(i, dtype=np.int64)
        pred = np.clip(self.mu + self.b_u[u] + self.b_i[i]
                       + np.einsum("ij,ij->i", self.U[u], self.V[i]),
                       CLIP_MIN, CLIP_MAX)
        return float(np.sqrt(np.mean((r - pred) ** 2)))

    def _rmse_raw(self, u, i, r):
        """RMSE sem a projecao em [1, 5].

        Serve para acompanhar o progresso do treino sem que o recorte altere
        os erros das previsoes que ultrapassam os limites da escala. No inicio,
        os vieses sao nulos e as predicoes ficam proximas da media do treino.
        """
        u = np.asarray(u, dtype=np.int64)
        i = np.asarray(i, dtype=np.int64)
        pred = (self.mu + self.b_u[u] + self.b_i[i]
                + np.einsum("ij,ij->i", self.U[u], self.V[i]))
        return float(np.sqrt(np.mean((r - pred) ** 2)))

    def predict(self, users, items) -> np.ndarray:
        u = np.asarray(users) - 1
        i = np.asarray(items) - 1
        out = self.raw_predict(u, i)
        return np.clip(out, CLIP_MIN, CLIP_MAX)

    def raw_predict(self, users_0based, items_0based) -> np.ndarray:
        """Unclipped estimate, i.e. without projecting onto the [1, 5] range."""
        u = np.asarray(users_0based)
        i = np.asarray(items_0based)
        return (self.mu + self.b_u[u] + self.b_i[i]
                + np.einsum("ij,ij->i", self.U[u], self.V[i]))

    def score_matrix(self, user_ids: np.ndarray) -> np.ndarray:
        """Raw scores for every item, for the given 0-based user ids.

        The scores are deliberately **not** clipped to $[1,5]$. Clipping is
        correct when the output is interpreted as a rating (it is what
        :meth:`predict` does, and it improves RMSE), but it is harmful for
        ranking: several items saturate at 5, the ties destroy the ordering of
        the top of the list, and the resulting recommendation quality collapses.
        """
        u = np.asarray(user_ids)
        return (self.U[u] @ self.V.T) + self.b_u[u][:, None] + self.b_i[None, :] + self.mu


# ---------------------------------------------------------------------------
# Content-based filtering on genres
# ---------------------------------------------------------------------------
class ContentBasedGenre:
    r"""Item-item filtragem por conteudo usando os generos dos filmes.

    Cada item e descrito pelo seu vetor binario de generos ``g_i``. A
    similaridade entre dois filmes e o cosseno (ou Jaccard) entre esses
    vetores, com ponderacao IDF opcional para atenuar generos muito frequentes
    (ex.: Drama aparece em 43% do catalogo). A nota prevista segue o mesmo
    principio da filtragem colaborativa baseada em itens,

    .. math::
        \\hat r_{ui} = \\mu_u + \\frac{\\sum_j s(i,j)\\,\\tilde r_{uj}}
        {\\sum_j s(i,j)}, \\qquad \\tilde r_{uj} = r_{uj} - \\mu_u,

    em que ``j`` percorre os itens que o usuario ``u`` ja avaliou e ``s(i,j)``
    depende apenas dos metadados dos itens. O modelo nao usa nenhuma
    informacao de outros usuarios, ilustrando a propriedade de independencia
    entre usuarios da filtragem por conteudo.
    """

    def __init__(self, metric: str = "cosine", idf: bool = True,
                 min_sim: float = 1e-9):
        assert metric in ("cosine", "jaccard")
        self.metric = metric
        self.idf = idf
        self.min_sim = min_sim

    def fit(self, users, items, ratings, n_users, n_items, G: np.ndarray):
        M = build_matrix(users, items, ratings, n_users, n_items)
        self.M = M
        self.G = G.copy().astype(np.float64)
        self.n_users, self.n_items = n_users, n_items
        self.gmean = float(np.nanmean(M))
        self.mu_u, self.n_u = user_means(M, self.gmean)

        X = self.G.copy()
        if self.idf:
            df = np.maximum(self.G.sum(axis=0), 1.0)
            idf = np.log(self.n_items / df) + 1.0
            X = X * idf[None, :]

        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1)
            norms[norms == 0] = np.inf
            Xn = X / norms[:, None]
            S = Xn @ Xn.T
        else:  # Jaccard
            inter = self.G @ self.G.T
            card = self.G.sum(axis=1)
            union = card[:, None] + card[None, :] - inter
            with np.errstate(invalid="ignore", divide="ignore"):
                S = np.where(union > 0, inter / union, 0.0)

        np.fill_diagonal(S, 0.0)
        S[S < self.min_sim] = 0.0
        self.S = S
        return self

    def region_predict(self, uu: int, ii: np.ndarray) -> np.ndarray:
        """Notas previstas para o usuario ``uu`` nos itens ``ii`` (0-based)."""
        S = self.S[ii, :]                       # n x n_items
        ru = self.M[uu, :] - self.mu_u[uu]
        obs = np.isfinite(ru)
        if not obs.any():
            return np.full(len(ii), self.mu_u[uu])
        So = S[:, obs]
        num = So @ ru[obs]
        den = So.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            est = np.where(den > 0, num / den, 0.0)
        return self.mu_u[uu] + est

    def predict(self, users, items) -> np.ndarray:
        u = np.asarray(users) - 1
        i = np.asarray(items) - 1
        out = np.empty(len(u), dtype=np.float64)
        for n, (uu, ii) in enumerate(zip(u, i)):
            out[n] = self.region_predict(uu, np.array([ii]))[0]
        return np.clip(out, CLIP_MIN, CLIP_MAX)

    def score_matrix(self, user_ids: np.ndarray) -> np.ndarray:
        u = np.asarray(user_ids)
        scores = np.empty((len(u), self.n_items), dtype=np.float64)
        allitems = np.arange(self.n_items)
        for r, uu in enumerate(u):
            scores[r] = self.region_predict(uu, allitems)
        return np.clip(scores, CLIP_MIN, CLIP_MAX)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
class GlobalMeanBaseline:
    def fit(self, users, items, ratings, n_users, n_items):
        self.mu = float(ratings.mean())
        self.n_items = n_items
        return self

    def predict(self, users, items):
        return np.full(len(np.asarray(users)), self.mu)

    def score_matrix(self, user_ids):
        return np.full((len(user_ids), self.n_items), self.mu)


class BiasBaseline:
    """mu + b_u + b_i (no interaction terms): a strong non-personalised floor."""

    def __init__(self, reg: float = 10.0):
        self.reg = reg

    def fit(self, users, items, ratings, n_users, n_items):
        self.mu = float(ratings.mean())
        self.n_users, self.n_items = n_users, n_items
        u = users - 1
        i = items - 1
        r = ratings.astype(float)
        b_u = np.zeros(n_users)
        b_i = np.zeros(n_items)
        for _ in range(15):
            resid = r - self.mu - b_i[i]
            num = np.bincount(u, weights=resid, minlength=n_users)
            cnt = np.bincount(u, minlength=n_users)
            b_u = num / (cnt + self.reg)
            resid = r - self.mu - b_u[u]
            num = np.bincount(i, weights=resid, minlength=n_items)
            cnt = np.bincount(i, minlength=n_items)
            b_i = num / (cnt + self.reg)
        self.b_u, self.b_i = b_u, b_i
        return self

    def predict(self, users, items):
        u = np.asarray(users) - 1
        i = np.asarray(items) - 1
        return np.clip(self.mu + self.b_u[u] + self.b_i[i], CLIP_MIN, CLIP_MAX)

    def score_matrix(self, user_ids):
        u = np.asarray(user_ids)
        S = self.mu + self.b_u[u][:, None] + self.b_i[None, :]
        return np.clip(S, CLIP_MIN, CLIP_MAX)


class PopularityBaseline:
    """Recomendador nao personalizado de POPULARIDADE.

    Ordena os itens pelo numero de avaliacoes no treino. E o modelo
    "recomende o que todo mundo assistiu": simples, barato e surpreendentemente
    forte como referencia, porque itens populares tem mais chance de agradar.

    Para predizer uma nota, usa a media do item encolhida em direcao a media
    global (``(n * media + lambda * media_global) / (n + lambda)``),
    evitando que itens com poucas
    avaliacoes recebam notas extremas.
    """

    def __init__(self, shrink: float = 10.0):
        self.shrink = shrink

    def fit(self, users, items, ratings, n_users, n_items):
        self.mu = float(ratings.mean())
        self.n_items = n_items
        i = items - 1
        r = ratings.astype(float)
        self.counts = np.bincount(i, minlength=n_items).astype(float)
        # media do item encolhida para a media global; itens sem avaliacao
        # recebem a propria media global (evita NaN)
        soma = np.bincount(i, weights=r, minlength=n_items)
        if self.shrink > 0:
            self.item_mean = (soma + self.shrink * self.mu) / (self.counts + self.shrink)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                self.item_mean = np.where(self.counts > 0,
                                          soma / np.maximum(self.counts, 1), self.mu)
        return self

    def predict(self, users, items):
        i = np.asarray(items) - 1
        return np.clip(self.item_mean[i], CLIP_MIN, CLIP_MAX)

    def score_matrix(self, user_ids):
        """Pontuacao de ranking: numero de avaliacoes (popularidade)."""
        return np.repeat(self.counts[None, :], len(user_ids), axis=0)


class TopRatedBaseline:
    """Recomendador nao personalizado de MELHORES AVALIADOS.

    Ordena os itens pela nota media no treino, com encolhimento bayesiano em
    direcao a media global:

    .. math::
        \\bar r_i = \\frac{n_i \\bar r_i + \\lambda \\mu}{n_i + \\lambda}

    Sem encolhimento (``lambda = 0``), um filme com uma unica avaliacao 5.0
    lideraria o ranking -- um artefato de amostra pequena, nao um sinal de
    qualidade. O encolhimento e, na pratica, a versao suave do filtro
    "so recomende com pelo menos N avaliacoes": o item so aparece no topo se
    tiver evidencia suficiente para superar a media global.

    Parameters
    ----------
    shrink : float
        Peso da media global (em numero de avaliacoes equivalentes).
    min_count : int
        Numero minimo de avaliacoes para o item ser elegivel. Itens abaixo do
        corte sao excluidos do ranking e recebem a media global na predicao.
    """

    def __init__(self, shrink: float = 25.0, min_count: int = 0):
        self.shrink = shrink
        self.min_count = min_count

    def fit(self, users, items, ratings, n_users, n_items):
        self.mu = float(ratings.mean())
        self.n_items = n_items
        i = items - 1
        r = ratings.astype(float)
        self.counts = np.bincount(i, minlength=n_items).astype(float)
        soma = np.bincount(i, weights=r, minlength=n_items)
        if self.shrink > 0:
            media = (soma + self.shrink * self.mu) / (self.counts + self.shrink)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                media = np.where(self.counts > 0, soma / np.maximum(self.counts, 1),
                                 self.mu)
        # itens com suporte insuficiente nao sao elegiveis para recomendacao
        if self.min_count > 0:
            media = np.where(self.counts >= self.min_count, media, self.mu)
        self.item_mean = media
        return self

    def predict(self, users, items):
        i = np.asarray(items) - 1
        return np.clip(self.item_mean[i], CLIP_MIN, CLIP_MAX)

    def score_matrix(self, user_ids):
        """Pontuacao de ranking: nota media encolhida."""
        scores = self.item_mean.copy()
        scores[self.counts < max(1, self.min_count)] = -np.inf
        return np.repeat(scores[None, :], len(user_ids), axis=0)


class RandomBaseline:
    """Recomendador ALEATORIO -- referencia sem informacao de preferencias.

    Sorteia uma pontuacao independente e uniforme em [1, 5] para cada par
    (usuario, item). Serve como referencia nula para cada metrica avaliada.

    O gerador e semeado (``seed``), de modo que o "sorteio" e reproduzivel e a
    comparacao entre modelos permanece justa.
    """

    def __init__(self, seed: int = 42, low: float = CLIP_MIN, high: float = CLIP_MAX):
        self.seed = seed
        self.low, self.high = low, high

    def fit(self, users, items, ratings, n_users, n_items):
        self.mu = float(ratings.mean())
        self.n_items = n_items
        self._rng = np.random.default_rng(self.seed)
        return self

    def predict(self, users, items):
        return self._rng.uniform(self.low, self.high, size=len(np.asarray(users)))

    def score_matrix(self, user_ids):
        return self._rng.uniform(self.low, self.high,
                                 size=(len(np.asarray(user_ids)), self.n_items))

