#!/usr/bin/env python3
"""Frozen TF-IDF + logistic baseline runner for Chapter 12. Requires sklearn 1.9.1."""
import argparse, csv, json, hashlib, os, sys
from pathlib import Path
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

EXPECTED_SKLEARN="1.9.1"
if sklearn.__version__ != EXPECTED_SKLEARN:
    raise SystemExit(f"HOLD: sklearn {sklearn.__version__}; expected {EXPECTED_SKLEARN}")

C_GRID=[0.1,1.0,10.0]

def load_rows(p):
    with open(p,encoding="utf-8",newline="") as f: return list(csv.DictReader(f))

def metrics(y,p):
    pr,rc,f1,sup=precision_recall_fscore_support(y,p,labels=[0,1,2],zero_division=0)
    return {
      "macro_f1":float(f1_score(y,p,average="macro")),
      "accuracy":float(accuracy_score(y,p)),
      "per_class":[{"label_id":i,"precision":float(pr[i]),"recall":float(rc[i]),"f1":float(f1[i]),"support":int(sup[i])} for i in range(3)],
      "confusion_matrix":confusion_matrix(y,p,labels=[0,1,2]).tolist()
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("record_ledger"); ap.add_argument("outdir")
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    rows=load_rows(a.record_ledger)
    by={s:[r for r in rows if r["split"]==s] for s in ["train","validation","test"]}
    if not all(by.values()): raise SystemExit("HOLD: empty split")
    vec=TfidfVectorizer(lowercase=True,strip_accents=None,analyzer="word",ngram_range=(1,2),min_df=2,max_df=1.0,
                        sublinear_tf=True,use_idf=True,smooth_idf=True,norm="l2",max_features=None)
    Xtr=vec.fit_transform([r["sentence"] for r in by["train"]])
    Xva=vec.transform([r["sentence"] for r in by["validation"]])
    Xte=vec.transform([r["sentence"] for r in by["test"]])
    ytr=np.array([int(r["label_id"]) for r in by["train"]])
    yva=np.array([int(r["label_id"]) for r in by["validation"]])
    yte=np.array([int(r["label_id"]) for r in by["test"]])
    cand=[]
    for C in C_GRID:
        clf=LogisticRegression(C=C,l1_ratio=0.0,solver="lbfgs",class_weight=None,fit_intercept=True,max_iter=1000,tol=1e-4)
        clf.fit(Xtr,ytr)
        pv=clf.predict(Xva); m=metrics(yva,pv)
        cand.append((m["macro_f1"],C,clf,m))
    cand.sort(key=lambda x:(-x[0],x[1]))
    _,bestC,clf,valm=cand[0]
    pt=clf.predict(Xte); testm=metrics(yte,pt)
    payload={"best_C":bestC,"validation":valm,"test":testm,"sklearn":sklearn.__version__,"vocab_size":len(vec.vocabulary_)}
    (out/"CH12_BASELINE_METRICS.json").write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    with open(out/"CH12_BASELINE_TEST_PREDICTIONS.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["record_id","y_true","y_pred"])
        for r,yp in zip(by["test"],pt): w.writerow([r["record_id"],r["label_id"],int(yp)])
    print(json.dumps(payload,sort_keys=True))
if __name__=="__main__": main()
