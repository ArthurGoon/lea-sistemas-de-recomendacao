"""Descriptive diagnostics of the selected models; no further tuning.

Adds paired gains, user-level comparisons and support of recommended items.
Every diagnostic retains the five external folds and their eligibility rule.
"""
import csv
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from ml_data import load_dataset,load_official_fold,to_matrix
from models import FunkSVD,KNNRecommender,PopularityBaseline

ROOT=Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((ROOT/"resultados"/name).read_text(encoding="utf-8"))
def write(name,data):
    (ROOT/"resultados"/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8",newline="\n")
def save(fig,name):
    fig.tight_layout(pad=1.1)
    for ext in ("pdf","png"):
        fig.savefig(ROOT/"figuras"/f"{name}.{ext}",bbox_inches="tight",dpi=260)
    plt.close(fig)
def dec(x,pos):
    return f"{x:g}".replace(".",",")
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.spines.top":False,
    "axes.spines.right":False,"legend.frameon":False,"axes.titleweight":"bold"})
choices=read("atividade_escolhas.json")
records=read("atividade_rmse.json")
summary=read("atividade_sintese.json")
def calculate():
    ds=load_dataset()
    NU,NI=ds.n_users,ds.n_items
    users_out=[]
    support=[]
    wins=[]
    errors=[]
    for f in range(1,6):
        tr,te=load_official_fold(f)
        a,b=to_matrix(tr,NU,NI),to_matrix(te,NU,NI)
        ev=np.flatnonzero(np.any(b>=4,axis=1))
        counts=np.isfinite(a).sum(axis=0)
        c=choices[f-1]
        u,i,r=(tr[x].to_numpy() for x in ("user","item","rating"))
        knn=KNNRecommender("user",k=c["knn"]["k"],sim_shrink=c["knn"]["sim_shrink"]).fit(u,i,r,NU,NI)
        svd=FunkSVD(n_factors=c["svd"]["n_factors"],lr=c["svd"]["lr"],reg=c["svd"]["reg"],
            n_epochs=40,early_stopping=False,random_state=42).fit(u,i,r,NU,NI)
        pop=PopularityBaseline().fit(u,i,r,NU,NI)
        by_model={}
        for label,score in (("KNN por nota",knn.rating_score_matrix),("KNN por soma",knn.score_matrix),
                            ("SVD",svd.score_matrix),("Mais populares",pop.score_matrix)):
            s=score(ev)
            hitcounts=[]
            per_user=[]
            neighbor_support=[]
            for row,uu in enumerate(ev):
                candidates=np.flatnonzero(np.isnan(a[uu])&np.isfinite(s[row]))
                rec=candidates[np.lexsort((candidates,-s[row,candidates]))[:10]]
                hits=int(np.sum(b[uu,rec]>=4))
                item_support=float(np.mean(counts[rec]))
                nbs=None
                if label.startswith("KNN"):
                    nb,_=knn._neighbour_terms(uu)
                    nbs=float(np.mean(np.isfinite(a[nb[:,None],rec]).sum(axis=0)))
                    neighbor_support.append(nbs)
                per_user.append(item_support)
                hitcounts.append(hits)
                users_out.append({"fold":f,"user":int(uu+1),"model":label,"hits10":hits,
                    "mean_item_count":item_support,"mean_neighbor_count":nbs})
            observed=next(x["precision10"] for x in records if x["fold"]==f and x["model"]==label)
            assert abs(np.mean(hitcounts)/10-observed)<1e-12
            by_model[label]=np.array(hitcounts)
            support.append({"fold":f,"model":label,"mean_item_count":float(np.mean(per_user)),
                "median_user_mean_item_count":float(np.median(per_user)),
                "mean_neighbor_count":float(np.mean(neighbor_support)) if neighbor_support else None})
        for first,second in (("KNN por soma","Mais populares"),("SVD","KNN por nota")):
            delta=by_model[first]-by_model[second]
            wins.append({"fold":f,"A":first,"B":second,"n_users":len(ev),
                "win_pct":float(np.mean(delta>0)*100),"tie_pct":float(np.mean(delta==0)*100),
                "loss_pct":float(np.mean(delta<0)*100),"mean_extra_hits":float(np.mean(delta)),
                "mean_extra_hits_winners":float(delta[delta>0].mean()),
                "mean_extra_hits_losers":float(delta[delta<0].mean()) if np.any(delta<0) else None})
        for label,m in (("SVD",svd),("KNN",knn)):
            err=te.rating.to_numpy()-m.predict(te.user.to_numpy(),te.item.to_numpy())
            ct=counts[te.item.to_numpy()-1]
            for lo,hi in ((0,5),(6,20),(21,100),(101,100000)):
                mask=(ct>=lo)&(ct<=hi)
                errors.append({"fold":f,"model":label,"lo":lo,"hi":hi,"n":int(mask.sum()),
                    "rmse":float(np.sqrt(np.mean(err[mask]**2)))})
        print(f"u{f}: diagnóstico de listas e comparações por usuário conferidos.",flush=True)
    write("atividade_diagnostico.json",{"support":support,"user_comparisons":wins,"error_by_item_support":errors,
        "interpretation":"Descriptive diagnostics of fixed selected models; fold means have equal weight; no new tuning or causal inference"})
    with (ROOT/"resultados/atividade_usuarios.csv").open("w",encoding="utf-8",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(users_out[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(users_out)


if "--plots-only" not in sys.argv:
    calculate()
diag=read("atividade_diagnostico.json")
support=diag["support"]
wins=diag["user_comparisons"]
errors=diag["error_by_item_support"]

# Two views of performance: ranking of the methods and same-fold gains.
fig,axes=plt.subplots(1,2,figsize=(9.2,3.5),gridspec_kw={"width_ratios":[1,1.2]})
names=["SVD","KNN por nota","KNN por soma","Mais populares","Melhores avaliados","Aleatório"]
for n,marker in zip(names,("o","v","^","s","D","x")):
    row=next(x for x in summary if x["model"]==n)
    axes[0].scatter(row["rmse"],10*row["precision10"],marker=marker,color="black",s=45)
    short={"KNN por nota":"KNN nota","KNN por soma":"KNN soma","Mais populares":"Populares","Melhores avaliados":"Nota média"}.get(n,n)
    offset={"SVD":(5,2),"KNN por nota":(5,3),"KNN por soma":(5,2),"Mais populares":(5,-11),"Melhores avaliados":(5,3),"Aleatório":(-48,7)}[n]
    axes[0].annotate(short,(row["rmse"],10*row["precision10"]),xytext=offset,textcoords="offset points",fontsize=9)
axes[0].set(xlabel="RMSE  ←  menor é melhor",ylabel="Acertos por lista de dez  →",title="(a) Dois objetivos",xlim=(.87,1.79),ylim=(-.1,2.35))
comparisons=[("SVD","KNN por nota","SVD − KNN nota"),("KNN por soma","SVD","KNN soma − SVD"),
    ("KNN por soma","Mais populares","KNN soma − populares")]
for j,(first,second,label) in enumerate(comparisons):
    dif=[]
    for f in range(1,6):
        x=next(x["precision10"] for x in records if x["fold"]==f and x["model"]==first)
        y=next(x["precision10"] for x in records if x["fold"]==f and x["model"]==second)
        dif.append(10*(x-y))
    axes[1].plot([min(dif),max(dif)],[j,j],color="0.55",lw=1)
    axes[1].scatter(dif,j+np.linspace(-.08,.08,5),facecolors="white",edgecolors="black",s=28)
    axes[1].scatter(np.mean(dif),j,marker="D",color="black",s=32,zorder=4)
axes[1].axvline(0,color="0.3",ls="--",lw=1)
axes[1].set(yticks=range(3),yticklabels=[x[2] for x in comparisons],xlabel="Acertos adicionais por lista de dez",title="(b) Ganhos na mesma partição",xlim=(-.08,1.75))
axes[1].invert_yaxis()
for ax in axes:
    ax.xaxis.set_major_formatter(FuncFormatter(dec))
    ax.yaxis.set_major_formatter(FuncFormatter(dec)) if ax==axes[0] else None
save(fig,"ativ_comparacoes")

# Mechanism diagnostic: support among selected neighbours and item frequency.
fig,axes=plt.subplots(1,2,figsize=(9.2,2.6))
for f in range(1,6):
    x=[next(x["mean_neighbor_count"] for x in support if x["fold"]==f and x["model"]==n) for n in ("KNN por nota","KNN por soma")]
    axes[0].plot([0,1],x,"o-",color="0.25",lw=1,ms=4)
axes[0].set(xticks=[0,1],xticklabels=["KNN por nota","KNN por soma"],ylabel="Vizinhos por\nfilme recomendado",title="(a) Suporte das recomendações",xlim=(-.2,1.25),ylim=(0,None))
labels=["0 a 5","6 a 20","21 a 100","> 100"]
for n,style in (("SVD","o-"),("KNN","s--")):
    values=[]
    for lo in (0,6,21,101):
        vals=[x["rmse"] for x in errors if x["model"]==n and x["lo"]==lo]
        values.append(np.mean(vals))
        axes[1].plot([len(values)-1]*2,[min(vals),max(vals)],color="0.6",lw=1)
    axes[1].plot(range(4),values,style,color="black",ms=5,label=n)
axes[1].set(xticks=range(4),xticklabels=labels,xlabel="Avaliações do filme no treino",ylabel="RMSE no teste",title="(b) Erro por suporte do filme")
for ax in axes:
    ax.tick_params(axis="both",labelsize=10,colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
axes[1].legend()
axes[0].yaxis.set_major_formatter(FuncFormatter(dec))
axes[1].yaxis.set_major_formatter(FuncFormatter(dec))
save(fig,"ativ_diagnostico")

# A mean advantage does not imply an improvement for every user.
fig,ax=plt.subplots(figsize=(8.6,2.2))
selected=[x for x in wins if x["A"]=="KNN por soma"]
left=np.zeros(5)
for key,color,hatch,label in (("loss_pct","0.2",None,"Menos acertos"),("tie_pct","0.88",None,"Mesmos acertos"),("win_pct","white","///","Mais acertos")):
    values=np.array([x[key] for x in selected])
    ax.barh(range(5),values,left=left,color=color,edgecolor="black",lw=.6,hatch=hatch,label=label,height=.65)
    for j,(v,l) in enumerate(zip(values,left)):
        if v>7: ax.text(l+v/2,j,f"{v:.1f}%".replace(".",","),ha="center",va="center",color="white" if key=="loss_pct" else "black",fontsize=9,bbox=dict(facecolor="white",edgecolor="none",pad=1.1) if key=="win_pct" else None)
    left+=values
ax.set(yticks=range(5),yticklabels=[f"u{f}" for f in range(1,6)],xlim=(0,100),xlabel="Usuários elegíveis em cada partição (%)")
ax.invert_yaxis()
ax.legend(ncol=3,loc="lower center",bbox_to_anchor=(.5,1.01),fontsize=9)
save(fig,"ativ_usuarios")
print("Visões comparativas e diagnósticos finalizados.")
