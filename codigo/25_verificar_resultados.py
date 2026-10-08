"""Independent aggregation and ranking audit, including complete refitting.

Recomputes metrics without calling metrics.evaluate_ranking or its helpers.
Verifies raw fold complements, internal split hashes and generated report values.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ml_data import load_dataset,load_official_fold,to_matrix
from models import FunkSVD,KNNRecommender,PopularityBaseline,TopRatedBaseline,RandomBaseline

ROOT=Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((ROOT/"resultados"/name).read_text(encoding="utf-8"))
def pairs(df):
    return set(zip(df.user,df.item,df.rating,df.timestamp))
def fingerprint(df):
    return hashlib.sha256(df[["user","item"]].to_numpy(dtype="int64").tobytes()).hexdigest()

ds=load_dataset()
whole=pairs(ds.ratings)
assert len(whole)==100000 and not ds.ratings.isna().any().any()
assert ds.ratings.rating.between(1,5).all()
assert not ds.ratings.duplicated(["user","item"]).any()
seen_test=set()
records=read("atividade_rmse.json")
protocol=read("atividade_protocolo.json")
choices=read("atividade_escolhas.json")
summary=read("atividade_sintese.json")
checked=0
for fold in range(1,6):
    tr,te=load_official_fold(fold)
    pt,pe=pairs(tr),pairs(te)
    assert not pt&pe and pt|pe==whole and not seen_test&pe
    seen_test|=pe
    assert set(te.user)<=set(tr.user)
    order=np.random.default_rng(42+fold).permutation(80000)
    fit,val=tr.iloc[order[16000:]],tr.iloc[order[:16000]]
    pr=protocol[fold-1]
    assert fingerprint(fit)==pr["inner_fit_sha256"]
    assert fingerprint(val)==pr["inner_val_sha256"]
    assert not (pairs(fit)&pairs(val) or (pairs(fit)|pairs(val))&pe)
    a,b=to_matrix(tr,ds.n_users,ds.n_items),to_matrix(te,ds.n_users,ds.n_items)
    users=np.where(np.sum(b>=4,axis=1)>0)[0]
    expectation=np.mean([np.sum(b[u]>=4)/np.sum(np.isnan(a[u])) for u in users])
    assert abs(expectation-pr["random_precision"])<1e-14 and len(users)==pr["ranking_users"]
    c=choices[fold-1]
    factories={"Aleatório":lambda:RandomBaseline(seed=42),"Mais populares":lambda:PopularityBaseline(shrink=10),
        "Melhores avaliados":lambda:TopRatedBaseline(shrink=0,min_count=c["top_rated"]["min_count"]),
        "KNN":lambda:KNNRecommender("user",k=c["knn"]["k"],sim_shrink=c["knn"]["sim_shrink"]),
        "SVD":lambda:FunkSVD(n_factors=c["svd"]["n_factors"],lr=c["svd"]["lr"],reg=c["svd"]["reg"],
                            n_epochs=40,early_stopping=False,random_state=42)}
    for name,factory in factories.items():
        m=factory().fit(*(tr[x].to_numpy() for x in ("user","item","rating")),ds.n_users,ds.n_items)
        pred=m.predict(te.user.to_numpy(),te.item.to_numpy())
        error=te.rating.to_numpy()-pred
        modes=[("KNN por nota",m.rating_score_matrix),("KNN por soma",m.score_matrix)] if name=="KNN" else [(name,m.score_matrix)]
        for label,score in modes:
            s=score(users)
            precisions,ndcgs=[],[]
            for row,u in enumerate(users):
                candidates=np.where(np.isnan(a[u])&np.isfinite(s[row]))[0]
                order=np.lexsort((candidates,-s[row,candidates]))
                rec=candidates[order[:10]]
                assert len(rec)==10 and not np.isfinite(a[u,rec]).any()
                relevant=b[u]>=4
                hits=relevant[rec].astype(float)
                precisions.append(float(hits.sum()/10))
                discount=1/np.log2(np.arange(2,12))
                ideal=discount[:min(10,int(relevant.sum()))].sum()
                ndcgs.append(float((hits*discount).sum()/ideal))
            old=next(x for x in records if x["fold"]==fold and x["model"]==label)
            for key,value in (("rmse",np.sqrt(np.mean(error**2))),("mae",np.mean(abs(error))),
                              ("precision10",np.mean(precisions)),("ndcg10",np.mean(ndcgs))):
                assert abs(old[key]-value)<1e-12,(fold,label,key,old[key],value)
                checked+=1
    global_row=next(x for x in records if x["fold"]==fold and x["model"]=="Média global")
    assert abs(global_row["rmse"]-np.sqrt(np.mean((te.rating-tr.rating.mean())**2)))<1e-12
    print(f"u{fold}: integridade, separação e métricas independentes conferidas.",flush=True)
assert seen_test==whole
for row in summary:
    for key in ("rmse","mae","precision10","ndcg10"):
        values=[x[key] for x in records if x["model"]==row["model"] and x[key] is not None]
        if values:
            assert abs(row[key]-np.mean(values))<1e-14
            assert abs(row[key+"_sd"]-np.std(values,ddof=1))<1e-14
diag=read("atividade_diagnostico.json")
user_results=pd.read_csv(ROOT/"resultados/atividade_usuarios.csv")
assert not user_results.duplicated(["fold","user","model"]).any()
assert user_results.hits10.between(0,10).all()
assert (user_results.hits10==user_results.hits10.astype(int)).all()
for fold in range(1,6):
    tr,te=load_official_fold(fold)
    matrix=to_matrix(tr,ds.n_users,ds.n_items)
    test=to_matrix(te,ds.n_users,ds.n_items)
    eligible=set(np.flatnonzero(np.any(test>=4,axis=1))+1)
    part=user_results[user_results.fold==fold]
    assert part.model.nunique()==4
    for model,g in part.groupby("model"):
        assert set(g.user)==eligible
        record=next(x for x in records if x["fold"]==fold and x["model"]==model)
        assert abs(g.hits10.mean()/10-record["precision10"])<1e-12
        support=next(x for x in diag["support"] if x["fold"]==fold and x["model"]==model)
        for key,column in (("mean_item_count","mean_item_count"),("mean_neighbor_count","mean_neighbor_count")):
            if support[key] is not None:
                assert abs(g[column].mean()-support[key])<1e-10
    hits=part.pivot(index="user",columns="model",values="hits10")
    for comparison in (x for x in diag["user_comparisons"] if x["fold"]==fold):
        delta=hits[comparison["A"]]-hits[comparison["B"]]
        assert comparison["n_users"]==len(eligible)
        for key,value in (("win_pct",(delta>0).mean()*100),("tie_pct",(delta==0).mean()*100),
                          ("loss_pct",(delta<0).mean()*100),("mean_extra_hits",delta.mean())):
            assert abs(comparison[key]-value)<1e-12
    counts=np.isfinite(matrix).sum(axis=0)[te.item.to_numpy()-1]
    for model,original in (("SVD","SVD"),("KNN","KNN por nota")):
        bins=[x for x in diag["error_by_item_support"] if x["fold"]==fold and x["model"]==model]
        assert len(bins)==4 and sum(x["n"] for x in bins)==len(te)
        for x in bins:
            assert x["n"]==int(((counts>=x["lo"])&(counts<=x["hi"])).sum())
        pooled=np.sqrt(sum(x["n"]*x["rmse"]**2 for x in bins)/len(te))
        record=next(x for x in records if x["fold"]==fold and x["model"]==original)
        assert abs(pooled-record["rmse"])<1e-12
print("Diagnósticos conferidos: 20 grupos de usuários, 10 comparações e 10 decomposições do RMSE.")
report={"status":"passed","outer_folds":5,"metric_values_recomputed":checked,
    "diagnostic_groups_checked":20,"user_comparisons_checked":10,"rmse_decompositions_checked":10,
    "method":"Full refit with selected parameters; independent lexsort, precision and nDCG; fold complement and split hash checks",
    "tolerance":1e-12}
(ROOT/"resultados/atividade_verificacao.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8",newline="\n")
print(f"Auditoria independente aprovada: {checked} valores de métricas recalculados.")
