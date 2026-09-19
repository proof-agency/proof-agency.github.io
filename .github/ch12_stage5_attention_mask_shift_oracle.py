#!/usr/bin/env python3
"""
Chapter 12 Stage-5 frozen attention/mask/shift oracle.
This script is canonical only under Python 3.14.7 + PyTorch 2.14.0 CPU.
It intentionally refuses a mismatched runtime.
"""
import os, sys, json, math, hashlib
from pathlib import Path

EXPECTED_TORCH = "2.14.0"
ATOL = 1e-12
RTOL = 1e-12

if os.environ.get("PYTHONHASHSEED") != "0":
    raise SystemExit("HOLD: PYTHONHASHSEED must be set to 0 before interpreter start")

import torch
import torch.nn.functional as F

if torch.__version__.split("+")[0] != EXPECTED_TORCH:
    raise SystemExit(f"HOLD: torch={torch.__version__}; expected {EXPECTED_TORCH}")

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(1201201)

dtype=torch.float64
device=torch.device("cpu")

# d_k=1 makes the analytic logits exact: [log(1),log(2),log(3),log(4)].
q=torch.ones((1,1,4,1), dtype=dtype, device=device)
k=torch.tensor([0.0, math.log(2.0), math.log(3.0), math.log(4.0)],
               dtype=dtype, device=device).reshape(1,1,4,1)
v=torch.tensor([10.0,20.0,30.0,40.0], dtype=dtype, device=device).reshape(1,1,4,1)

def causal_additive(n):
    m=torch.zeros((n,n), dtype=dtype, device=device)
    m[torch.triu(torch.ones((n,n), dtype=torch.bool, device=device), diagonal=1)] = float("-inf")
    return m

mask=causal_additive(4)
out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0,is_causal=False)
expected=torch.tensor([10.0, 50.0/3.0, 70.0/3.0, 30.0],
                      dtype=dtype, device=device).reshape(1,1,4,1)
torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)

# Scratch parity
scores=(q @ k.transpose(-2,-1)) + mask
weights=torch.softmax(scores, dim=-1)
scratch=weights @ v
torch.testing.assert_close(out, scratch, atol=ATOL, rtol=RTOL)
row_sums=weights.sum(-1)
torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=ATOL, rtol=RTOL)

# Future perturbation: change future K/V heavily; prefixes up to query t must not change.
k2=k.clone(); v2=v.clone()
k2[:,:,3,:] += 1000.0; v2[:,:,3,:] += 10000.0
out2=F.scaled_dot_product_attention(q,k2,v2,attn_mask=mask,dropout_p=0.0,is_causal=False)
torch.testing.assert_close(out[:,:,:3,:], out2[:,:,:3,:], atol=ATOL, rtol=RTOL)

# Padding-key oracle: block key index 1 for all queries, in addition to causality.
pad_mask=mask.clone()
pad_mask[:,1]=float("-inf")
out_pad=F.scaled_dot_product_attention(q,k,v,attn_mask=pad_mask,dropout_p=0.0,is_causal=False)
if not torch.isfinite(out_pad).all():
    raise AssertionError("padding oracle produced non-finite output")

# Functional-SDPA bool semantics: True means allowed.
bool_allowed=torch.isfinite(mask)
out_bool=F.scaled_dot_product_attention(q,k,v,attn_mask=bool_allowed,dropout_p=0.0,is_causal=False)
torch.testing.assert_close(out_bool, out, atol=ATOL, rtol=RTOL)

# BOS shift oracle
tokens=[11,12,13,14]
BOS=2
u=[BOS]+tokens[:-1]
y=tokens[:]
assert u == [2,11,12,13]
assert y == [11,12,13,14]
assert len(u)==len(y)

payload={
  "status":"PASS",
  "python":sys.version,
  "torch":torch.__version__,
  "device":"cpu",
  "dtype":"float64",
  "atol":ATOL,
  "rtol":RTOL,
  "expected_causal_output":[10.0,50.0/3.0,70.0/3.0,30.0],
  "causal_output":[float(x) for x in out.reshape(-1)],
  "bos_shift_input":u,
  "bos_shift_target":y,
  "tests":{
    "analytic_expected":True,
    "scratch_parity":True,
    "row_sums":True,
    "future_perturbation":True,
    "padding_mask":True,
    "functional_bool_polarity":True,
    "bos_shift":True
  }
}
Path("CH12_ATTENTION_MASK_SHIFT_ORACLE_RESULT.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
