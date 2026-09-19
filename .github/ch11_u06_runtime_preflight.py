from __future__ import annotations
import os, sys, importlib, json, platform
EXPECTED={
 'python':'3.14.7','numpy':'2.5.3','scipy':'1.18.1','pandas':'3.0.6','torch':'2.14.0',
 'statsmodels':'0.15.0','arch':'8.0.0','matplotlib':'3.11.2'
}
obs={'python':platform.python_version(),'PYTHONHASHSEED':os.environ.get('PYTHONHASHSEED')}
errors=[]
if obs['python']!=EXPECTED['python']: errors.append(f"python {obs['python']} != {EXPECTED['python']}")
if obs['PYTHONHASHSEED']!='0': errors.append('PYTHONHASHSEED must equal 0 before interpreter startup')
for name in ['numpy','scipy','pandas','torch','statsmodels','arch','matplotlib']:
    try:
        mod=importlib.import_module(name)
        v=getattr(mod,'__version__','UNKNOWN')
        if name=='torch' and '+' in v: v=v.split('+',1)[0]
        obs[name]=v
        if v!=EXPECTED[name]: errors.append(f"{name} {v} != {EXPECTED[name]}")
    except Exception as e:
        obs[name]=f"IMPORT_ERROR: {e}"
        errors.append(f"{name} import failed")
if 'torch' in obs and not str(obs['torch']).startswith('IMPORT_ERROR'):
    import torch
    obs['torch_device']='cpu'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    obs['torch_num_threads']=torch.get_num_threads()
    obs['torch_deterministic_algorithms']=torch.are_deterministic_algorithms_enabled()
    obs['torch_default_dtype']=str(torch.get_default_dtype())
    if obs['torch_num_threads']!=1: errors.append('torch_num_threads != 1')
    if not obs['torch_deterministic_algorithms']: errors.append('torch deterministic algorithms not enabled')
result={'expected':EXPECTED,'observed':obs,'status':'PASS' if not errors else 'HOLD','errors':errors}
print(json.dumps(result,indent=2,sort_keys=True))
with open('CH11_U06_RUNTIME_PREFLIGHT_RESULT.json','w',encoding='utf-8') as f:
    json.dump(result,f,indent=2,sort_keys=True)
sys.exit(0 if not errors else 2)
