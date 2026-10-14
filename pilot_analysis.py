import pandas as pd, numpy as np, sys, json
# usage: python pilot_analysis.py outputs.csv rater1.xlsx rater2.xlsx
from recal.stats_human import wilcoxon_rb, holm, weighted_kappa
from scipy import stats
import statsmodels.formula.api as smf
D=['Concept Alignment (1-5)','Cognitive Alignment (1-5)','Knowledge-Type Alignment (1-5)','Pedagogical Appropriateness (1-5)','Correctness (1-5)']
o=pd.read_csv(sys.argv[1]); key=o[["case_id","method","prompt","generated_output"]]
E={}
for k,f in [("R1",sys.argv[2]),("R2",sys.argv[3])]:
    d=pd.read_excel(f); m=d.merge(key,left_on=["Case ID","Prompt / Learner Context"],right_on=["case_id","prompt"],how="left")
    d["method"]=m.method.values; d["out"]=m.generated_output.values; d["Q"]=d[D].astype(float).mean(axis=1); d["rater"]=k; E[k]=d
order=["Generic-Concept","Bloom-Aware","Context-Aware","RePatch","RePatch+Validator"]
res={}
res["kappa"]={dd:round(weighted_kappa(E["R1"][dd].astype(int),E["R2"][dd].astype(int)),3) for dd in D}
res["kappa_pooled"]=round(weighted_kappa(E["R1"][D].astype(int).values.ravel(),E["R2"][D].astype(int).values.ravel()),3)
Q=((E["R1"].set_index(["Case ID","method"]).Q+E["R2"].set_index(["Case ID","method"]).Q)/2).unstack()[order]
rng=np.random.default_rng(42); desc=[]
for c in order:
    v=Q[c].values; bs=[rng.choice(v,len(v)).mean() for _ in range(5000)]
    desc.append([c,round(v.mean(),3),round(v.std(ddof=1),3),round(np.percentile(bs,2.5),3),round(np.percentile(bs,97.5),3)])
res["desc"]=desc
fr=stats.friedmanchisquare(*[Q[c] for c in order]); n,k=Q.shape; res["friedman"]=[round(fr.statistic,2),fr.pvalue,round(fr.statistic/(n*(k-1)),3)]
rows=[]
for x,y in zip(order[:-1],order[1:]): r=wilcoxon_rb(Q[x],Q[y]); rows.append([f"{x}->{y}",round(r["mean_diff"],3),r["n_nonzero"],r["p"],round(r["r_rb"],3)])
ph=holm([r[3] for r in rows]); 
for r,p in zip(rows,ph): r.append(p)
res["contrasts"]=rows
long=pd.concat(E.values()); long=long.rename(columns={"Case ID":"case"})
mm=smf.mixedlm("Q ~ C(method, Treatment('Context-Aware')) + C(rater)",long,groups=long["case"]).fit()
res["lmm_RePatch"]=[round(mm.params["C(method, Treatment('Context-Aware'))[T.RePatch]"],3), [round(x,3) for x in mm.conf_int().loc["C(method, Treatment('Context-Aware'))[T.RePatch]"]]]
p=o[o.method=="RePatch"].set_index("case_id").prompt; reg=p.str.extract(r"support intensity: (\w+)")[0]
g=(Q["RePatch"]-Q["Context-Aware"]); gr=[]
for rr in ["HIGH_SUPPORT","MEDIUM_SUPPORT","LOW_SUPPORT"]:
    v=g[reg[reg==rr].index]; w=wilcoxon_rb(Q.loc[v.index,"Context-Aware"],Q.loc[v.index,"RePatch"]); gr.append([rr,len(v),round(v.mean(),3),round(w["p"],3)])
res["region"]=gr; res["kruskal"]=stats.kruskal(*[g[reg[reg==rr].index] for rr in ["HIGH_SUPPORT","MEDIUM_SUPPORT","LOW_SUPPORT"]]).pvalue
chk=o[o.method=="RePatch+Validator"].set_index("case_id").generated_output.astype(str).str.contains("Concept Alignment:|Bloom Alignment:")
res["validator_checklists"]=int(chk.sum())
res["validator_Q_checklist_vs_not"]=[round(Q.loc[chk[chk].index,"RePatch+Validator"].mean(),3), round(Q.loc[chk[~chk].index,"RePatch+Validator"].mean(),3)]
# validator vs RePatch excluding checklists
ok=chk[~chk].index; w=wilcoxon_rb(Q.loc[ok,"RePatch"],Q.loc[ok,"RePatch+Validator"]); res["validator_valid_only"]=[len(ok),round(w["mean_diff"],3),w["p"],round(w["r_rb"],3)]
res["leakage"]=[int(E["R1"]["Answer Leakage (0/1)"].sum()),int(E["R2"]["Answer Leakage (0/1)"].sum())]
res["len_words"]=o.assign(n=o.generated_output.astype(str).str.split().str.len()).groupby("method").n.median().to_dict()
res["n_cases"]=int(n); res["pt_range"]=p.str.extract(r"probability=([\d.]+?)\.?(?:\s|;|$)")[0].astype(float).agg(["min","max"]).round(3).tolist()
print(json.dumps(res,indent=1,default=str))
json.dump(res,open("pilot_results.json","w"),indent=1,default=str)
