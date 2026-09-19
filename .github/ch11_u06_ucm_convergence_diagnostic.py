from __future__ import annotations
import json, os, platform
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
import statsmodels

PROTOCOL=Path(".github/ch11_stage5_sequence_protocol_frozen.json")
OUT=Path("u06_ucm_diag"); OUT.mkdir(exist_ok=True)
p=json.loads(PROTOCOL.read_text())

def dgp(seed):
    d=p["dgp"]; n=int(d["generated_n"]); burn=int(d["burn_in"])
    a,b,c=np.random.SeedSequence(seed).spawn(3)
    eta=np.random.default_rng(a).normal(0,float(d["sigma_eta"]),n)
    eps=np.random.default_rng(b).normal(0,float(d["sigma_epsilon"]),n)
    z=np.random.default_rng(c).normal(0,1,n)
    s=np.empty(n); s[0]=float(d["mu"]); mu=float(d["mu"]); phi=float(d["phi"])
    for t in range(1,n): s[t]=mu+phi*(s[t-1]-mu)+eta[t]
    v=s+eps; r=np.exp(s/2)*z
    return s[burn:],v[burn:],r[burn:]

def fit(v,maxiter=None):
    mean=float(np.mean(v[:4800])); std=float(np.std(v[:4800],ddof=0))
    zz=(v[:4800]-mean)/std
    kw={"disp":False}
    if maxiter is not None: kw["maxiter"]=maxiter
    res=UnobservedComponents(zz,level=False,trend=False,irregular=True,autoregressive=1).fit(**kw)
    mp=dict(zip(res.param_names,np.asarray(res.params,float)))
    return {
      "sigma2.irregular":float(mp["sigma2.irregular"]),
      "sigma2.ar":float(mp["sigma2.ar"]),
      "ar.L1":float(mp["ar.L1"]),
      "stationary":bool(abs(float(mp["ar.L1"]))<1),
      "converged":bool(res.mle_retvals.get("converged",False)),
      "iterations":res.mle_retvals.get("iterations"),
      "llf":float(res.llf),
      "mean":mean,"std":std,
    }

rows=[]
for rep,seed in enumerate(p["dgp"]["dgp_seeds"],1):
    s,v,r=dgp(seed)
    a=fit(v,None); b=fit(v,1000)
    row={"replicate":rep,"seed":seed}
    for k,val in a.items(): row[f"default_{k}"]=val
    for k,val in b.items(): row[f"maxiter1000_{k}"]=val
    for k in ["sigma2.irregular","sigma2.ar","ar.L1","llf"]:
        row[f"absdiff_{k}"]=abs(float(a[k])-float(b[k]))
    rows.append(row)

df=pd.DataFrame(rows)
df.to_csv(OUT/"CH11_U06_UCM_CONVERGENCE_DIAGNOSTIC.csv",index=False)
summary={
 "python":platform.python_version(),"statsmodels":statsmodels.__version__,
 "PYTHONHASHSEED":os.environ.get("PYTHONHASHSEED"),
 "all_maxiter1000_converged":bool(df["maxiter1000_converged"].all()),
 "all_stationary":bool(df["maxiter1000_stationary"].all()),
 "max_abs_param_diff":float(max(df.filter(like="absdiff_").drop(columns=[c for c in df.filter(like="absdiff_").columns if c.endswith("llf")],errors="ignore").to_numpy().max(),0.0)),
 "replicate5_default_converged":bool(df.loc[df.replicate==5,"default_converged"].iloc[0]),
 "replicate5_maxiter1000_converged":bool(df.loc[df.replicate==5,"maxiter1000_converged"].iloc[0]),
}
(OUT/"CH11_U06_UCM_CONVERGENCE_DIAGNOSTIC_SUMMARY.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
print(df.to_string(index=False))
print(json.dumps(summary,indent=2,sort_keys=True))
