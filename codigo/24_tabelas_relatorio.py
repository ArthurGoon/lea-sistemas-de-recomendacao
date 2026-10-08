"""Generate every reported result directly from the final JSON files."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TABLE=ROOT/"relatorio/tabelas"
TABLE.mkdir(exist_ok=True)
def read(name):
    return json.loads((ROOT/"resultados"/name).read_text(encoding="utf-8"))
def number(x,d=4):
    return rf"\num{{{x:.{d}f}}}"
def rows(name,items):
    content="\n".join(" & ".join(map(str,row))+r" \\" for row in items)+"\n"
    (TABLE/name).write_text(content,encoding="utf-8",newline="\n")
    command="tbl"+"".join(x.capitalize() for x in Path(name).stem.split("_"))
    with (ROOT/"relatorio/numeros_atividade.tex").open("a",encoding="utf-8",newline="\n") as f:
        f.write(rf"\newcommand{{\{command}}}{{"+"\n"+content+"}\n")
def escape(s):
    for old,new in (("&",r"\&"),("%",r"\%"),("_",r"\_"),("#",r"\#")):
        s=s.replace(old,new)
    return s

eda=read("atividade_eda.json")
summary=read("atividade_sintese.json")
protocol=read("atividade_protocolo.json")
choices=read("atividade_escolhas.json")
cv=read("atividade_rmse.json")
paired=read("atividade_testes.json")["comparisons"]
macros=[]
for model,prefix in (("SVD","svd"),("KNN por nota","knn"),("KNN por soma","soma"),
                    ("Mais populares","pop"),("Melhores avaliados","top"),("Aleatório","random"),("Média global","global")):
    data=next(s for s in summary if s["model"]==model)
    for key,suffix in (("rmse","RMSE"),("mae","MAE"),("precision10","P"),("ndcg10","N"),
                       ("rmse_sd","RMSEsd"),("precision10_sd","Psd")):
        if data[key] is not None:
            macros.append(rf"\newcommand{{\{prefix}{suffix}}}{{{number(data[key])}}}")
expected=np.mean([p["random_precision"] for p in protocol])
macros.append(rf"\newcommand{{\acasoP}}{{{number(expected)}}}")
diag=read("atividade_diagnostico.json")
by_model={s["model"]:s for s in summary}
extra={
    "ganhoRmse":100*(1-by_model["SVD"]["rmse"]/by_model["KNN por nota"]["rmse"]),
    "ganhoPop":100*(by_model["KNN por soma"]["precision10"]/by_model["Mais populares"]["precision10"]-1),
}
for label,suffix in (("KNN por nota","Nota"),("KNN por soma","Soma")):
    extra["viz"+suffix]=np.mean([s["mean_neighbor_count"] for s in diag["support"] if s["model"]==label])
    extra["aval"+suffix]=np.mean([s["mean_item_count"] for s in diag["support"] if s["model"]==label])
for key,name in (("win_pct","vence"),("tie_pct","empata"),("loss_pct","perde")):
    extra[name]=np.mean([s[key] for s in diag["user_comparisons"] if s["A"]=="KNN por soma"])
for name,value in extra.items():
    macros.append(rf"\newcommand{{\{name}}}{{{number(value,1)}}}")
for model,name in (("SVD","acertosSvd"),("KNN por soma","acertosSoma"),("Mais populares","acertosPop")):
    macros.append(rf"\newcommand{{\{name}}}{{{number(10*by_model[model]['precision10'],2)}}}")
for other,name in (("SVD","difSomaSvd"),("Mais populares","difSomaPop")):
    difference=10*(by_model["KNN por soma"]["precision10"]-by_model[other]["precision10"])
    macros.append(rf"\newcommand{{\{name}}}{{{number(difference,2)}}}")
for lo,name in ((0,"nPoucoSuporte"),(101,"nMuitoSuporte")):
    count=sum(s["n"] for s in diag["error_by_item_support"] if s["model"]=="SVD" and s["lo"]==lo)
    macros.append(rf"\newcommand{{\{name}}}{{{number(count,0)}}}")
(ROOT/"relatorio/numeros_atividade.tex").write_text("\n".join(macros)+"\n",encoding="utf-8",newline="\n")
for key,file in (("popular","ativ_populares.tex"),("rated","ativ_bem_avaliados.tex")):
    rows(file,[(escape(x["title"]),x["count"],number(x["mean"],3)) for x in eda[key][:3]])
rows("ativ_suporte.tex",[(f"{x['lo']} a {x['hi']}" if x["hi"]<100000 else "Mais de 100",x["n"],
    number(x["mean"],3),number(x["sd"],3)) for x in eda["support"]])
rows("ativ_protocolo.tex",[(f"u{x['fold']}",x["test_users"],x["ranking_users"],x["unseen_items"],
    number(x["random_precision"])) for x in protocol])
rows("ativ_escolhas.tex",[(f"u{x['fold']}",x["knn"]["k"],x["knn"]["sim_shrink"],x["svd"]["n_factors"],
    number(x["svd"]["lr"],3),number(x["svd"]["reg"],2),x["top_rated"]["min_count"]) for x in choices])
out=[]
for s in summary:
    def with_sd(key):
        return "--" if s[key] is None else number(s[key])+r" $\pm$ "+number(s[key+"_sd"])
    out.append((s["model"],with_sd("rmse"),number(s["mae"]),with_sd("precision10"),
                "--" if s["ndcg10"] is None else number(s["ndcg10"])))
rows("ativ_resultados.tex",out)
rows("ativ_folds.tex",[(f"u{f}",*[number(next(x[metric] for x in cv if x["fold"]==f and x["model"]==model))
    for model,metric in (("SVD","rmse"),("KNN por nota","rmse"),("SVD","precision10"),
                        ("KNN por nota","precision10"),("KNN por soma","precision10"),("Mais populares","precision10"))]) for f in range(1,6)])
rows("ativ_diferencas.tex",[(x["A"]+" / "+x["B"],"RMSE" if x["metric"]=="rmse" else "$P$@10",
    number(x["mean_difference"]),f"{x['wins']}/5") for x in paired])
trials=read("atividade_tuning.json")[0]["top_rated"]
rows("ativ_suporte_validacao.tex",[(x["min_count"],x["eligible"],number(x["rmse"]),number(x["precision10"]))
    for x in trials if x["min_count"] in (1,5,20,80,150)])
print("Macros e tabelas sincronizadas com os resultados finais.")
