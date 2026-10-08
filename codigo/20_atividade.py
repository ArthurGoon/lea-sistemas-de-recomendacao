"""Final activity: official outer CV with independent inner holdout tuning.

Run: python codigo/20_atividade.py. Outer test ratings never select parameters.
"""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ml_data import load_dataset, load_official_fold, to_matrix
from metrics import evaluate_ranking, expected_random_precision, mae, rmse, _topk_from_scores
from models import FunkSVD, KNNRecommender, PopularityBaseline, RandomBaseline, TopRatedBaseline

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "resultados", ROOT / "figuras"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
SEED, EPOCHS = 42, 40
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.spines.top":False,
    "axes.spines.right":False,"legend.frameon":False,"figure.dpi":140,"savefig.dpi":250})

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8",newline="\n")

def arrays(df):
    return tuple(df[c].to_numpy() for c in ("user","item","rating"))

def pairs(df):
    return set(zip(df.user,df.item))

def inner_split(outer_train,fold):
    order=np.random.default_rng(SEED+fold).permutation(len(outer_train))
    nval=len(order)//5
    return outer_train.iloc[order[nval:]].copy(),outer_train.iloc[order[:nval]].copy()

def fingerprint(df):
    return hashlib.sha256(df[["user","item"]].to_numpy(dtype="int64").tobytes()).hexdigest()

def ranking(model,tr,te,mode="standard"):
    a,b=to_matrix(tr,NU,NI),to_matrix(te,NU,NI)
    users=np.flatnonzero(np.any(b>=4,axis=1))
    score=model.rating_score_matrix if mode=="rating" else model.score_matrix
    return evaluate_ranking(score,users,a,b,ks=(10,),threshold=4.0)

def savefig(fig,name):
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(FIG/f"{name}.{ext}",bbox_inches="tight")
    plt.close(fig)

ds=load_dataset()
NU,NI=ds.n_users,ds.n_items
R=ds.ratings
user=R.groupby("user").rating.agg(["count","mean"])
item=R.groupby("item").rating.agg(["count","mean"])
eda={"n_ratings":len(R),"n_users":NU,"n_items":NI,"mean":float(R.rating.mean()),
    "sd":float(R.rating.std(ddof=0)),"density":len(R)/(NU*NI),
    "rating_counts":{str(i):int((R.rating==i).sum()) for i in range(1,6)},
    "user_counts":{k:float(v) for k,v in user["count"].describe().items()},
    "item_counts":{k:float(v) for k,v in item["count"].describe().items()},
    "few_items":int((item["count"]<=5).sum()),"one_rating":int((item["count"]==1).sum()),
    "perfect_mean":int((item["mean"]==5).sum()),
    "male_pct":float((ds.users.gender=="M").mean()*100),"median_age":float(ds.users.age.median())}
cum=np.cumsum(np.sort(item["count"])[::-1])/len(R)
eda["catalog_half_pct"]=100*(np.searchsorted(cum,.5)+1)/NI
titles=ds.items.set_index("item").title
top=item.assign(title=titles).reset_index()
eda["popular"]=top.sort_values(["count","item"],ascending=[False,True]).head(5).to_dict("records")
eda["rated"]=top[top["count"]>=100].sort_values(["mean","item"],ascending=[False,True]).head(5).to_dict("records")
eda["support"]=[]
for lo,hi in ((1,2),(3,5),(6,20),(21,50),(51,100),(101,100000)):
    x=item.loc[item["count"].between(lo,hi),"mean"]
    eda["support"].append({"lo":lo,"hi":hi,"n":len(x),"mean":float(x.mean()),"sd":float(x.std(ddof=0))})
assert sum(x["n"] for x in eda["support"])==NI
dump("atividade_eda.json",eda)
fig,axes=plt.subplots(1,3,figsize=(9,2.7))
axes[0].bar(range(1,6),[eda["rating_counts"][str(i)] for i in range(1,6)],color="0.25")
axes[0].set(xlabel="Nota",ylabel="Avaliações",title="(a) Notas",xticks=range(1,6))
axes[1].hist(user["count"],bins=35,color="0.25")
axes[1].set(xlabel="Avaliações por usuário",ylabel="Usuários",title="(b) Atividade dos usuários")
axes[2].hist(item["count"],bins=35,color="0.25")
axes[2].set(xlabel="Avaliações por filme",ylabel="Filmes",title="(c) Popularidade dos filmes")
savefig(fig,"ativ_eda")

records,choices,tuning,protocol,qualitative=[],[],[],[],[]
for fold in range(1,6):
    print(f"Fold u{fold}: seleção interna",flush=True)
    tr,te=load_official_fold(fold)
    fit,val=inner_split(tr,fold)
    assert not (pairs(fit)&pairs(val) or pairs(tr)&pairs(te))
    assert pairs(fit)|pairs(val)==pairs(tr)
    uf,itf,rf=arrays(fit)
    uv,iv,rv=arrays(val)
    trials={"fold":fold,"knn":[],"top_rated":[],"svd":[]}
    for k in (10,20,40,80,160,300):
        for shrink in (0,50):
            m=KNNRecommender("user",k=k,sim_shrink=shrink).fit(uf,itf,rf,NU,NI)
            trials["knn"].append({"k":k,"sim_shrink":shrink,"rmse":rmse(rv,m.predict(uv,iv))})
    bk=min(trials["knn"],key=lambda x:x["rmse"])
    for mc in (1,5,10,20,40,80,150):
        m=TopRatedBaseline(shrink=0,min_count=mc).fit(uf,itf,rf,NU,NI)
        rk=ranking(m,fit,val)
        trials["top_rated"].append({"min_count":mc,"precision10":rk["Precision@10"],
            "ndcg10":rk["nDCG@10"],"rmse":rmse(rv,m.predict(uv,iv)),
            "eligible":int((m.counts>=mc).sum())})
    bt=max(trials["top_rated"],key=lambda x:(x["precision10"],x["ndcg10"]))
    for d in (20,50,100):
        for lr in (.005,.01):
            for reg in (.05,.1):
                m=FunkSVD(n_factors=d,n_epochs=EPOCHS,lr=lr,reg=reg,
                    random_state=SEED,early_stopping=False).fit(uf,itf,rf,NU,NI)
                trials["svd"].append({"n_factors":d,"lr":lr,"reg":reg,"rmse":rmse(rv,m.predict(uv,iv))})
    bs=min(trials["svd"],key=lambda x:x["rmse"])
    choices.append({"fold":fold,"knn":bk,"top_rated":bt,"svd":bs})
    tuning.append(trials)
    print(f"  KNN {bk}; SVD {bs}; suporte {bt['min_count']}",flush=True)
    ut,it,rt=arrays(tr)
    ue,ie,re=arrays(te)
    a,b=to_matrix(tr,NU,NI),to_matrix(te,NU,NI)
    ev=np.flatnonzero(np.any(b>=4,axis=1))
    test_users=np.flatnonzero(np.any(np.isfinite(b),axis=1))
    expected=expected_random_precision(a,b,ev)
    protocol.append({"fold":fold,"n_train":len(tr),"n_test":len(te),"n_inner_fit":len(fit),
        "n_inner_val":len(val),"test_users":len(test_users),"ranking_users":len(ev),
        "unseen_items":len(set(te.item)-set(tr.item)),"random_precision":expected,
        "inner_fit_sha256":fingerprint(fit),"inner_val_sha256":fingerprint(val)})
    factories={"Aleatório":lambda:RandomBaseline(seed=SEED),
        "Mais populares":lambda:PopularityBaseline(shrink=10),
        "Melhores avaliados":lambda:TopRatedBaseline(shrink=0,min_count=bt["min_count"]),
        "KNN":lambda:KNNRecommender("user",k=bk["k"],sim_shrink=bk["sim_shrink"]),
        "SVD":lambda:FunkSVD(n_factors=bs["n_factors"],lr=bs["lr"],reg=bs["reg"],
            n_epochs=EPOCHS,random_state=SEED,early_stopping=False)}
    for name,factory in factories.items():
        m=factory().fit(ut,it,rt,NU,NI)
        pred=m.predict(ue,ie)
        modes=[("KNN por nota","rating"),("KNN por soma","standard")] if name=="KNN" else [(name,"standard")]
        for label,mode in modes:
            rk=ranking(m,tr,te,mode)
            row={"fold":fold,"model":label,"rmse":rmse(re,pred),"mae":mae(re,pred),
                "precision10":rk["Precision@10"],"ndcg10":rk["nDCG@10"],"n_users":int(rk["n_users_evaluated"])}
            records.append(row)
            print(f"  {label:20s} RMSE={row['rmse']:.4f} P@10={row['precision10']:.4f}",flush=True)
            if fold==5 and label in ("Mais populares","KNN por nota","KNN por soma","SVD"):
                score=m.rating_score_matrix if mode=="rating" else m.score_matrix
                ids=_topk_from_scores(score(np.array([12]))[0],np.isfinite(a[12]),5)
                qualitative.append({"model":label,"user":13,"train_count":int(np.isfinite(a[12]).sum()),
                    "recommendations":[{"item":int(i+1),"title":titles.loc[i+1],
                        "test_rating":None if np.isnan(b[12,i]) else float(b[12,i]),"hit":bool(b[12,i]>=4)} for i in ids]})
    baseline=np.full(len(re),rt.mean())
    records.append({"fold":fold,"model":"Média global","rmse":rmse(re,baseline),"mae":mae(re,baseline),
        "precision10":None,"ndcg10":None,"n_users":None})
    for name,obj in (("atividade_rmse.json",records),("atividade_escolhas.json",choices),
        ("atividade_tuning.json",tuning),("atividade_protocolo.json",protocol)):
        dump(name,obj)

df=pd.DataFrame(records)
summary=[]
for name,g in df.groupby("model",sort=False):
    row={"model":name}
    for metric in ("rmse","mae","precision10","ndcg10"):
        x=g[metric].dropna()
        row[metric]=float(x.mean()) if len(x) else None
        row[metric+"_sd"]=float(x.std(ddof=1)) if len(x)>1 else None
    summary.append(row)
dump("atividade_sintese.json",summary)
dump("atividade_ranking.json",records)
dump("atividade_exemplo.json",qualitative)
comparisons=[]
for metric,x,y in (("rmse","SVD","KNN por nota"),("precision10","SVD","KNN por nota"),
    ("precision10","KNN por soma","SVD"),("precision10","KNN por soma","Mais populares")):
    xa=df[df.model==x].sort_values("fold")[metric].to_numpy()
    xb=df[df.model==y].sort_values("fold")[metric].to_numpy()
    delta=xa-xb
    comparisons.append({"metric":metric,"A":x,"B":y,"mean_difference":float(delta.mean()),
        "min_difference":float(delta.min()),"max_difference":float(delta.max()),
        "wins":int(np.sum(delta<0 if metric=="rmse" else delta>0))})
dump("atividade_testes.json",{"interpretation":"Descriptive paired differences; overlapping training sets; no iid t-test", "comparisons":comparisons})
dump("atividade_ambiente.json",{"python":platform.python_version(),"numpy":np.__version__,
    "pandas":pd.__version__,"matplotlib":matplotlib.__version__,"seed":SEED,"epochs":EPOCHS})
fig,axes=plt.subplots(1,2,figsize=(9,3.5))
names=["Aleatório","Mais populares","Melhores avaliados","KNN por nota","KNN por soma","SVD"]
for ax,metric in zip(axes,("rmse","precision10")):
    means=[next(s[metric] for s in summary if s["model"]==n) for n in names]
    sd=[next(s[metric+"_sd"] for s in summary if s["model"]==n) for n in names]
    ax.barh(names,means,xerr=sd,color="0.35",error_kw={"elinewidth":1,"capsize":2})
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (menor é melhor)" if metric=="rmse" else "Precisão@10 (maior é melhor)")
axes[1].axvline(np.mean([p["random_precision"] for p in protocol]),color="black",ls="--",lw=1)
savefig(fig,"ativ_modelos")
fig,axes=plt.subplots(1,2,figsize=(9,3))
for name,style in (("SVD","o-"),("KNN por nota","s--"),("KNN por soma","^:"),("Mais populares","D-.")):
    g=df[df.model==name].sort_values("fold")
    for ax,metric in zip(axes,("rmse","precision10")):
        if metric=="rmse" and name=="KNN por soma": continue
        ax.plot(g.fold,g[metric],style,color="black",label=name,ms=4,lw=1)
for ax,metric in zip(axes,("RMSE","Precisão@10")):
    ax.set(xlabel="Partição oficial",ylabel=metric,xticks=range(1,6))
    ax.legend(fontsize=8)
savefig(fig,"ativ_folds")
print("Concluído: resultados, protocolo e figuras atualizados.",flush=True)
