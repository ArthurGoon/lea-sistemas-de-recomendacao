"""Testes de corretude dos modelos e das metricas.

Sao testes de propriedade (nao apenas de "roda sem erro"): cada um verifica uma
identidade matematica que a implementacao deve satisfazer. Rodar com:

    python codigo/testes.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import (hit_rate_at_k, mae, ndcg_at_k, precision_at_k, expected_random_precision, _topk_from_scores,  # noqa: E402
                     recall_at_k, reciprocal_rank, rmse)
from models import (BiasBaseline, ContentBasedGenre, FunkSVD,  # noqa: E402
                    GlobalMeanBaseline, KNNRecommender, PopularityBaseline,
                    build_matrix, user_means, TopRatedBaseline)

FALHAS = []
TOTAL = [0]


def checar(nome, cond, detalhe=""):
    TOTAL[0] += 1
    status = "OK  " if cond else "FALHA"
    print(f"  [{status}] {nome}" + (f"  -> {detalhe}" if detalhe and not cond else ""))
    if not cond:
        FALHAS.append(nome)


# ---------------------------------------------------------------------------
print("\n== metricas ==")
checar("rmse de predicao perfeita = 0", abs(rmse(np.array([1., 5.]), np.array([1., 5.]))) < 1e-12)
checar("rmse constante c = c",
       abs(rmse(np.array([1., 2., 3.]), np.array([2., 3., 4.])) - 1.0) < 1e-12)
checar("mae constante c = c",
       abs(mae(np.array([1., 2., 3.]), np.array([2., 3., 4.])) - 1.0) < 1e-12)
checar("precision@3 com 2 acertos = 2/3",
       abs(precision_at_k([0, 1, 5], {0, 1}, 3) - 2 / 3) < 1e-12)
checar("recall@3 com 2 de 4 relevantes = 0.5",
       abs(recall_at_k([0, 1, 5], {0, 1, 7, 8}, 3) - 0.5) < 1e-12)
checar("hit_rate@1 = 1 quando o topo acerta", hit_rate_at_k([3, 1], {3}, 1) == 1.0)
checar("ndcg@k = 1 com ordenacao ideal",
       abs(ndcg_at_k([0, 1], {0, 1}, 2) - 1.0) < 1e-12)
checar("ndcg@2 penaliza acerto na 2a posicao",
       ndcg_at_k([9, 0], {0}, 2) < ndcg_at_k([0, 9], {0}, 2))
checar("reciprocal_rank = 1/2 na 2a posicao",
       abs(reciprocal_rank([9, 3], {3}) - 0.5) < 1e-12)
checar("reciprocal_rank = 0 sem acerto", reciprocal_rank([9, 8], {3}) == 0.0)

# ---------------------------------------------------------------------------
print("\n== media global ==")
u = np.array([1, 1, 2, 2])
i = np.array([1, 2, 1, 2])
r = np.array([1.0, 3.0, 5.0, 3.0])
gm = GlobalMeanBaseline().fit(u, i, r, 2, 2)
checar("media global prediz a media", abs(gm.predict(u, i)[0] - 3.0) < 1e-12)
checar("media global e o minimizador do MSE",
       abs(rmse(r, gm.predict(u, i)) - r.std()) < 1e-12)

# ---------------------------------------------------------------------------
print("\n== baseline de vieses ==")
# Dados construidos para serem EXATAMENTE aditivos: r_ui = 3 + b_u + b_i.
# Nesse caso o modelo mu + b_u + b_i e corretamente especificado e deve
# reconstruir o treino sem erro residual.
b_u_true = np.array([1.0, -1.0, 0.0, 0.5])
b_i_true = np.array([0.5, -0.5, 0.0])
U4 = np.repeat(np.arange(1, 5), 3)
I4 = np.tile(np.arange(1, 4), 4)
R4 = 3.0 + b_u_true[U4 - 1] + b_i_true[I4 - 1]
bb = BiasBaseline(reg=0.0).fit(U4, I4, R4, 4, 3)
p_bb = bb.predict(U4, I4)
checar("baseline reproduz dados aditivos exatamente (reg=0)",
       rmse(R4, p_bb) < 1e-6, f"RMSE={rmse(R4, p_bb):.2e}")
checar("baseline recupera os vieses verdadeiros (a menos de constante)",
       np.allclose(bb.b_u - bb.b_u[2], b_u_true - b_u_true[2], atol=1e-6),
       f"b_u={np.round(bb.b_u, 3)} vs {b_u_true}")
checar("baseline e melhor que a media global nesse caso",
       rmse(R4, p_bb) < rmse(R4, np.full_like(R4, R4.mean())))
# Dados com interacao: o modelo aditivo NAO pode ajustar tudo.
R4i = R4.copy()
R4i[0] += 1.0
bb_i = BiasBaseline(reg=0.0).fit(U4, I4, R4i, 4, 3)
checar("baseline NAO ajusta efeitos de interacao",
       rmse(R4i, bb_i.predict(U4, I4)) > 0.1)

# ---------------------------------------------------------------------------
print("\n== FunkSVD: fatoracao exata de uma matriz de posto baixo ==")
# Construimos uma matriz de notas EXATAMENTE de posto 2 e dentro da escala
# valida [1, 5], e verificamos que o FunkSVD com d=2 a reconstroi. Sem vieses
# (reg_bias=0) e sem regularizacao, o SGD converge para os fatores verdadeiros.
#
# A trajetória de treino registra o RMSE recalculado em cada época,
# após a atualização, de modo que history[i] reflete o estado após a
# (i+1)-esima epoca. A comparacao de convergencia abaixo usa o ultimo valor.
rng = np.random.default_rng(0)
A = rng.normal(0, 1.0, (12, 2))
B = rng.normal(0, 1.0, (15, 2))
P = A @ B.T
P = P - P.mean()
P = 0.45 * P / P.std()          # mantem a matriz dentro de [1, 5]
M2 = 3.5 + P
uu, ii = np.meshgrid(np.arange(12), np.arange(15), indexing="ij")
u_flat, i_flat = uu.ravel() + 1, ii.ravel() + 1
r_flat = M2.ravel()
checar("matriz de teste esta na escala [1,5]",
       bool(r_flat.min() >= 1.0 and r_flat.max() <= 5.0),
       f"range=[{r_flat.min():.2f}, {r_flat.max():.2f}]")

m = FunkSVD(n_factors=2, n_epochs=1500, lr=0.05, reg=0.0, reg_bias=0.0,
            random_state=0, early_stopping=False, init_std=0.01)
m.fit(u_flat, i_flat, r_flat, 12, 15)
err = rmse(r_flat, m.predict(u_flat, i_flat))
checar("SVD reconstroi matriz de posto 2 (d=2)", err < 1e-6, f"RMSE={err:.3e}")
checar("SVD converge: erro final proximo de zero",
       err < 1e-6 and m.history_[0][1] > 0.1,
       f"erro inicial {m.history_[0][1]:.4f} -> final {err:.3e}")
checar("historico de treino tem tendencia decrescente",
       m.history_[-1][1] < m.history_[0][1]
       and np.mean([h[1] for h in m.history_[:50]])
       > np.mean([h[1] for h in m.history_[-50:]]),
       f"media inicial {np.mean([h[1] for h in m.history_[:50]]):.4f} -> "
       f"final {np.mean([h[1] for h in m.history_[-50:]]):.6f}")
checar("oscilacoes do SGD sao pequenas (sem divergencia)",
       max(m.history_[k + 1][1] - m.history_[k][1]
           for k in range(len(m.history_) - 1)) < 0.01,
       f"maior aumento={max(m.history_[k + 1][1] - m.history_[k][1] for k in range(len(m.history_) - 1)):.3e}")
checar("erro de treino cai pelo menos uma ordem de grandeza",
       m.history_[-1][1] < m.history_[0][1] / 10,
       f"{m.history_[0][1]:.4f} -> {m.history_[-1][1]:.6f}")
checar("estado final restaurado ajusta exatamente (sem projecao)",
       abs(float(np.sqrt(np.mean(
           (r_flat - m.raw_predict(u_flat - 1, i_flat - 1)) ** 2)))) < 1e-6)

# _rmse deve ser identico ao RMSE calculado por predict (mesma convencao).
# Convencao: _rmse / _rmse_raw recebem indices 0-based.
m_check = FunkSVD(n_factors=3, n_epochs=3, lr=0.01, reg=0.05, random_state=1,
                  early_stopping=False)
m_check.fit(u_flat, i_flat, r_flat, 12, 15)
checar("_rmse coincide com o RMSE de predict()",
       abs(m_check._rmse(u_flat - 1, i_flat - 1, r_flat)
           - rmse(r_flat, m_check.predict(u_flat, i_flat))) < 1e-12)
checar("_rmse_raw e >= _rmse (o recorte em [1,5] so pode ajudar)",
       m_check._rmse_raw(u_flat - 1, i_flat - 1, r_flat)
       >= m_check._rmse(u_flat - 1, i_flat - 1, r_flat) - 1e-12)
checar("historico de treino e globalmente decrescente",
       m_check.history_[-1][1] < m_check.history_[0][1],
       f"{m_check.history_[0][1]:.6f} -> {m_check.history_[-1][1]:.6f}")
checar("historico converge para o valor final recalculado",
       abs(m.history_[-1][1]
           - m._rmse_raw(u_flat - 1, i_flat - 1, r_flat)) < 1e-9)

m1f = FunkSVD(n_factors=1, n_epochs=1500, lr=0.05, reg=0.0, reg_bias=0.0,
              random_state=0, early_stopping=False, init_std=0.01)
m1f.fit(u_flat, i_flat, r_flat, 12, 15)
err1 = rmse(r_flat, m1f.predict(u_flat, i_flat))
checar("SVD com d insuficiente erra mais que com d suficiente",
       err1 > err, f"d=1: {err1:.4f} vs d=2: {err:.3e}")

# ---------------------------------------------------------------------------
print("\n== KNN: caso construido a mao ==")
# matriz 3 usuarios x 3 itens, sem ausencias, para conferir a formula
U3 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
I3 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
R3 = np.array([5.0, 4.0, 1.0,
               5.0, 4.0, 1.0,
               1.0, 2.0, 5.0])
M3 = build_matrix(U3, I3, R3, 3, 3)
mu3, _ = user_means(M3, 3.3333)
checar("medias de usuario corretas",
       np.allclose(mu3, [10 / 3, 10 / 3, 8 / 3]), f"{mu3}")

knn = KNNRecommender("user", k=1).fit(U3, I3, R3, 3, 3)
# usuarios 1 e 2 tem perfis identicos -> sim = 1
checar("similaridade entre perfis identicos = 1",
       abs(knn.S[0, 1] - 1.0) < 1e-9, f"S[0,1]={knn.S[0,1]:.6f}")
checar("similaridade entre perfis opostos = -1 (descartada)",
       knn.S[0, 2] <= 0.0, f"S[0,2]={knn.S[0,2]:.6f}")

knn_all = KNNRecommender("item", k=2).fit(U3, I3, R3, 3, 3)
checar("previsao do item-KNN fica no intervalo [1,5]",
       np.all((knn_all.predict(U3, I3) >= 1) & (knn_all.predict(U3, I3) <= 5)))
checar("score_matrix tem forma (n_usuarios, n_itens)",
       knn_all.score_matrix(np.arange(3)).shape == (3, 3))

# ---------------------------------------------------------------------------
print("\n== filtragem por conteudo ==")
G = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
cb = ContentBasedGenre(metric="cosine", idf=False).fit(U3, I3, R3, 3, 3, G)
checar("itens de mesmo genero tem similaridade 1",
       abs(cb.S[0, 1] - 1.0) < 1e-9)
checar("itens de generos disjuntos tem similaridade 0",
       abs(cb.S[0, 2]) < 1e-9)
cb_j = ContentBasedGenre(metric="jaccard", idf=False).fit(U3, I3, R3, 3, 3, G)
checar("Jaccard entre identicos = 1", abs(cb_j.S[0, 1] - 1.0) < 1e-9)
checar("previsao de conteudo no intervalo [1,5]",
       np.all((cb.predict(U3, I3) >= 1) & (cb.predict(U3, I3) <= 5)))

# ---------------------------------------------------------------------------
print("\n== determinismo ==")
m1 = FunkSVD(n_factors=5, n_epochs=5, lr=0.01, reg=0.05, random_state=42,
             parallel=False, early_stopping=False)
m2 = FunkSVD(n_factors=5, n_epochs=5, lr=0.01, reg=0.05, random_state=42,
             parallel=False, early_stopping=False)
big_u = np.repeat(np.arange(1, 21), 5)
big_i = np.tile(np.arange(1, 6), 20)
big_r = np.random.default_rng(1).uniform(1, 5, size=big_u.size)
m1.fit(big_u, big_i, big_r, 20, 5)
m2.fit(big_u, big_i, big_r, 20, 5)
checar("modo serial e bit a bit reproduzivel",
       np.array_equal(m1.predict(big_u, big_i), m2.predict(big_u, big_i)))

pop = PopularityBaseline().fit(U3, I3, R3, 3, 3)
checar("baseline de popularidade nao personaliza",
       np.allclose(pop.score_matrix(np.arange(3))[0], pop.score_matrix(np.arange(3))[1]))

# ---------------------------------------------------------------------------
print("\n== regressões do protocolo final ==")
from itertools import combinations
train_toy = np.array([[5., np.nan, np.nan, np.nan, np.nan]])
test_toy = np.array([[np.nan, 4., 5., 1., np.nan]])
all_lists = list(combinations([1, 2, 3, 4], 2))
enumerated = np.mean([precision_at_k(x, {1,2}, 2) for x in all_lists])
checar("precisao esperada coincide com enumeracao de todos os sorteios",
       abs(expected_random_precision(train_toy,test_toy,[0])-enumerated)<1e-12)
checar("empates ordenados por id; vistos e inelegiveis excluidos",
       np.array_equal(_topk_from_scores(np.array([5.,4.,4.,-np.inf]),
                      np.array([True,False,False,False]),10),[1,2]))
ku=np.repeat(np.arange(1,4),3)
ki=np.tile(np.arange(1,4),3)
kr=np.array([5.,4.,1.,5.,4.,1.,1.,2.,5.])
km=KNNRecommender("user",k=2).fit(ku,ki,kr,3,4)
grid_u=np.repeat(np.arange(1,4),4)
grid_i=np.tile(np.arange(1,5),3)
checar("KNN matricial normalizado coincide com predict em todos os pares",
       np.allclose(np.clip(km.rating_score_matrix(np.arange(3)),1,5).ravel(),
                   km.predict(grid_u,grid_i)))
tm=TopRatedBaseline(shrink=0,min_count=4).fit(ku,ki,kr,3,4)
checar("suporte minimo exclui itens no ranking mesmo com notas baixas",
       not np.isfinite(tm.score_matrix(np.array([0]))).any())
sm=FunkSVD(n_factors=3,n_epochs=3,early_stopping=False).fit(ku,ki,kr,4,4)
checar("SVD sem historico usa apenas termos estimaveis",
       np.all(sm.U[3]==0) and np.all(sm.V[3]==0) and
       abs(sm.predict(np.array([4]),np.array([4]))[0]-sm.mu)<1e-12)

print(f"\n{'=' * 60}")
print(f"testes executados: {TOTAL[0]}   falhas: {len(FALHAS)}")
if FALHAS:
    for f in FALHAS:
        print("  X", f)
    sys.exit(1)
print("TODOS OS TESTES PASSARAM")
