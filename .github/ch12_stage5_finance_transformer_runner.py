#!/usr/bin/env python3
"""Frozen DistilBERT transfer-classification runner for Chapter 12.
Canonical execution is blocked until the asset manifest and exact environment gate PASS.
"""
import argparse, csv, json, os, random, sys, hashlib
from pathlib import Path
import numpy as np
import torch
import sklearn
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

EXPECTED={"python":"3.14.7","torch":"2.14.0","sklearn":"1.9.1","transformers":"5.17.0"}
SEEDS=[12021,12022,12023]
LR=2e-5; WD=0.01; BATCH=16; MAX_EPOCHS=6; PATIENCE=2; IMPROVEMENT=1e-12; MAX_LENGTH=128

def basever(v): return v.split("+")[0]
def gate_runtime():
    py=".".join(map(str,sys.version_info[:3]))
    got={"python":py,"torch":basever(torch.__version__),"sklearn":sklearn.__version__,"transformers":transformers.__version__}
    if got!=EXPECTED: raise SystemExit(f"HOLD: runtime mismatch {got} expected {EXPECTED}")
    if os.environ.get("PYTHONHASHSEED")!="0": raise SystemExit("HOLD: PYTHONHASHSEED != 0")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def verify_assets(model_dir, manifest_path):
    m=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if m.get("status")!="PASS": raise SystemExit("HOLD: model asset manifest not PASS")
    for rec in m["files"]:
        p=Path(model_dir)/rec["relative_path"]
        if not p.is_file() or sha256_file(p)!=rec["sha256"]:
            raise SystemExit(f"HOLD: asset mismatch {rec['relative_path']}")

def load_rows(p):
    with open(p,encoding="utf-8",newline="") as f: return list(csv.DictReader(f))

def metric_payload(y,p):
    pr,rc,f1,sup=precision_recall_fscore_support(y,p,labels=[0,1,2],zero_division=0)
    return {"macro_f1":float(f1_score(y,p,average="macro")),"accuracy":float(accuracy_score(y,p)),
            "per_class":[{"label_id":i,"precision":float(pr[i]),"recall":float(rc[i]),"f1":float(f1[i]),"support":int(sup[i])} for i in range(3)],
            "confusion_matrix":confusion_matrix(y,p,labels=[0,1,2]).tolist()}

class DS(torch.utils.data.Dataset):
    def __init__(self, rows, tok):
        self.rows=rows; self.tok=tok
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        enc=self.tok(r["sentence"],truncation=True,max_length=MAX_LENGTH,padding=False)
        enc["labels"]=int(r["label_id"]); enc["record_id"]=r["record_id"]
        return enc

def evaluate(model, loader):
    model.eval(); ys=[]; ps=[]; ids=[]
    with torch.no_grad():
        for b in loader:
            rid=b.pop("record_id")
            y=b["labels"].cpu().tolist()
            o=model(**{k:v for k,v in b.items()})
            p=o.logits.argmax(-1).cpu().tolist()
            ys+=y; ps+=p; ids+=list(rid)
    return metric_payload(ys,ps), ids, ys, ps

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("record_ledger"); ap.add_argument("model_dir"); ap.add_argument("asset_manifest"); ap.add_argument("outdir")
    a=ap.parse_args(); gate_runtime(); verify_assets(a.model_dir,a.asset_manifest)
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    rows=load_rows(a.record_ledger)
    splits={s:[r for r in rows if r["split"]==s] for s in ["train","validation","test"]}
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True,use_fast=True)
    tok.truncation_side="right"
    collator=DataCollatorWithPadding(tok,return_tensors="pt")
    def collate(items):
        ids=[x.pop("record_id") for x in items]
        b=collator(items); b["record_id"]=ids; return b
    allres=[]
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        gen=torch.Generator().manual_seed(seed+100000)
        train_loader=torch.utils.data.DataLoader(DS(splits["train"],tok),batch_size=BATCH,shuffle=True,generator=gen,collate_fn=collate,num_workers=0)
        val_loader=torch.utils.data.DataLoader(DS(splits["validation"],tok),batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=0)
        test_loader=torch.utils.data.DataLoader(DS(splits["test"],tok),batch_size=BATCH,shuffle=False,collate_fn=collate,num_workers=0)
        model=AutoModelForSequenceClassification.from_pretrained(a.model_dir,local_files_only=True,num_labels=3)
        model.to("cpu")
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
        best=None; best_state=None; stale=0; history=[]
        for epoch in range(1,MAX_EPOCHS+1):
            model.train()
            for b in train_loader:
                b.pop("record_id")
                opt.zero_grad(set_to_none=True)
                loss=model(**b).loss
                if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
                loss.backward(); opt.step()
            vm,_,_,_=evaluate(model,val_loader)
            history.append({"epoch":epoch,"validation":vm})
            score=vm["macro_f1"]
            if best is None or score > best + IMPROVEMENT:
                best=score; stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_epoch=epoch
            else:
                stale+=1
                if stale>=PATIENCE: break
        if best_state is None: raise RuntimeError("no checkpoint")
        model.load_state_dict(best_state)
        tm,ids,ys,ps=evaluate(model,test_loader)
        with open(out/f"CH12_TRANSFORMER_TEST_PREDICTIONS_SEED_{seed}.csv","w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["record_id","y_true","y_pred"])
            for a1,b1,c1 in zip(ids,ys,ps): w.writerow([a1,b1,c1])
        rec={"seed":seed,"best_epoch":best_epoch,"best_validation_macro_f1":best,"test":tm,"history":history}
        (out/f"CH12_TRANSFORMER_METRICS_SEED_{seed}.json").write_text(json.dumps(rec,indent=2,sort_keys=True),encoding="utf-8")
        allres.append(rec)
    vals=[r["test"]["macro_f1"] for r in allres]
    agg={"per_seed":allres,"test_macro_f1_mean":float(np.mean(vals)),"test_macro_f1_sample_sd":float(np.std(vals,ddof=1)),
         "test_macro_f1_median":float(np.median(vals)),"note":"descriptive optimization-replicate variation; not inferential uncertainty"}
    (out/"CH12_TRANSFORMER_AGGREGATE.json").write_text(json.dumps(agg,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(agg,sort_keys=True))
if __name__=="__main__": main()
