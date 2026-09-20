#!/usr/bin/env python3
"""Frozen Chapter-12 finance data preparation gate. No model fit occurs here."""
import argparse, csv, hashlib, json, re, unicodedata, zipfile
from pathlib import Path

EXPECTED_ARCHIVE_SHA256="0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6"
MEMBER="FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
LABELS={"negative":0,"neutral":1,"positive":2}
SPLIT_SEED=12012
PROPORTIONS={"train":0.70,"validation":0.15,"test":0.15}
NEAR_DUP_CHAR5_JACCARD=0.90

def sha256_bytes(b): return hashlib.sha256(b).hexdigest()
def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20), b""): h.update(c)
    return h.hexdigest()

def norm_text(s):
    s=unicodedata.normalize("NFKC",s).strip().lower()
    s=re.sub(r"\s+"," ",s)
    return s

def grams5(s):
    if len(s)<5: return {s}
    return {s[i:i+5] for i in range(len(s)-4)}

def jac(a,b):
    u=a|b
    return 1.0 if not u else len(a&b)/len(u)

def horder(seed, gid):
    return hashlib.sha256(f"{seed}|{gid}".encode()).hexdigest()

def apportion(n):
    raw={k:n*v for k,v in PROPORTIONS.items()}
    base={k:int(raw[k]//1) for k in raw}
    remain=n-sum(base.values())
    order=sorted(raw,key=lambda k:(-(raw[k]-base[k]), ["train","validation","test"].index(k)))
    for k in order[:remain]: base[k]+=1
    return base

def deterministic_assign(groups):
    # groups: list of dict(group_id,label,record_ids,size)
    out={}
    for label in sorted({g["label"] for g in groups}):
        gl=[g for g in groups if g["label"]==label]
        target=apportion(sum(g["size"] for g in gl))
        counts={k:0 for k in PROPORTIONS}
        gl.sort(key=lambda g:(-g["size"],horder(SPLIT_SEED,g["group_id"])))
        for g in gl:
            # choose split with largest remaining deficit; ties train->validation->test
            deficits={k:target[k]-counts[k] for k in target}
            best=max(["train","validation","test"], key=lambda k:(deficits[k], -["train","validation","test"].index(k)))
            out[g["group_id"]]=best
            counts[best]+=g["size"]
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("outdir")
    a=ap.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    if sha256_file(a.archive)!=EXPECTED_ARCHIVE_SHA256:
        raise SystemExit("HOLD: dataset archive SHA256 mismatch")
    with zipfile.ZipFile(a.archive) as z:
        if MEMBER not in z.namelist():
            raise SystemExit("HOLD: expected all-agree member absent")
        raw=z.read(MEMBER)
        members=z.namelist()
    rows=[]
    for lineno,line in enumerate(raw.splitlines(keepends=True), start=1):
        line_nl=line.rstrip(b"\r\n")
        textline=line_nl.decode("iso-8859-1")
        if "@" not in textline: raise SystemExit(f"HOLD: malformed line {lineno}")
        sentence,label=textline.rsplit("@",1)
        if label not in LABELS: raise SystemExit(f"HOLD: unknown label {label!r} line {lineno}")
        rid=hashlib.sha256(MEMBER.encode()+b"\0"+str(lineno).encode()+b"\0"+line_nl).hexdigest()
        n=norm_text(sentence)
        gid=hashlib.sha256(n.encode("utf-8")).hexdigest()
        rows.append(dict(record_id=rid,source_line=lineno,sentence=sentence,label=label,label_id=LABELS[label],
                         normalized_text=n,exact_group_id=gid))
    if len(rows)!=2264: raise SystemExit(f"HOLD: expected 2264 rows, got {len(rows)}")

    # exact duplicate groups and conflicting-label gate
    by={}
    for r in rows: by.setdefault(r["exact_group_id"],[]).append(r)
    for gid,rr in by.items():
        if len({x["label"] for x in rr})!=1:
            raise SystemExit(f"HOLD: conflicting labels in exact duplicate group {gid}")

    # deterministic near-duplicate connected components on exact groups, within same label only.
    gids=list(by)
    parent={g:g for g in gids}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    cache={g:grams5(by[g][0]["normalized_text"]) for g in gids}
    for i,g1 in enumerate(gids):
        s1=by[g1][0]["normalized_text"]; lab1=by[g1][0]["label"]
        for g2 in gids[i+1:]:
            if by[g2][0]["label"]!=lab1: continue
            s2=by[g2][0]["normalized_text"]
            if min(len(s1),len(s2))/max(1,max(len(s1),len(s2))) < 0.80: continue
            if jac(cache[g1],cache[g2]) >= NEAR_DUP_CHAR5_JACCARD:
                union(g1,g2)

    # Merge exact groups into near-duplicate components.
    comp={}
    for gid,rr in by.items():
        root=find(gid)
        comp.setdefault(root,[]).extend(rr)
    groups=[]
    for root,rr in comp.items():
        labs={x["label"] for x in rr}
        if len(labs)!=1: raise SystemExit(f"HOLD: conflicting labels in near-duplicate component {root}")
        groups.append({"group_id":root,"label":next(iter(labs)),"record_ids":[x["record_id"] for x in rr],"size":len(rr)})

    assign=deterministic_assign(groups)
    for r in rows:
        r["group_id"]=find(r["exact_group_id"])
        r["split"]=assign[r["group_id"]]

    with open(out/"CH12_FINANCE_RECORD_LEDGER.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["record_id","source_line","sentence","label","label_id","normalized_text","exact_group_id","group_id","split"])
        w.writeheader(); w.writerows(rows)
    with open(out/"CH12_FINANCE_SPLIT_LEDGER.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["record_id","group_id","label","split"])
        w.writeheader()
        for r in rows: w.writerow({k:r[k] for k in ["record_id","group_id","label","split"]})
    stats={
      "archive_sha256":sha256_file(a.archive),
      "selected_member":MEMBER,
      "selected_member_sha256":sha256_bytes(raw),
      "n_records":len(rows),
      "label_counts":{lab:sum(r["label"]==lab for r in rows) for lab in LABELS},
      "split_counts":{s:sum(r["split"]==s for r in rows) for s in PROPORTIONS},
      "n_exact_groups":len(by),
      "n_near_duplicate_components":len(groups),
      "near_duplicate_rule":{"char_5gram_jaccard_threshold":NEAR_DUP_CHAR5_JACCARD,"length_ratio_min":0.80},
      "split_seed":SPLIT_SEED,
      "proportions":PROPORTIONS
    }
    (out/"CH12_FINANCE_DATA_PREP_QA.json").write_text(json.dumps(stats,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(stats,sort_keys=True))

if __name__=="__main__": main()
