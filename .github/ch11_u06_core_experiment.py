from __future__ import annotations
import hashlib, json, os, platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.structural import UnobservedComponents
from arch import arch_model
import statsmodels, arch, matplotlib, scipy

PROTOCOL_PATH = Path('.github/ch11_stage5_sequence_protocol_frozen.json')
EXPECTED_PROTOCOL_SHA = 'ed846434ed58c3c0f9b70e5095ae194db6bb953e095bdf04bd66185a958ad283'
OUT = Path('u06_results')
OUT.mkdir(exist_ok=True)

EXPECTED_RUNTIME = {
    'python':'3.14.7','numpy':'2.5.3','scipy':'1.18.1','pandas':'3.0.6','torch':'2.14.0',
    'statsmodels':'0.15.0','arch':'8.0.0','matplotlib':'3.11.2'
}

def torch_ver(v):
    return v.split('+',1)[0]

def runtime_gate():
    obs = {
        'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'pandas':pd.__version__,'torch':torch_ver(torch.__version__),'statsmodels':statsmodels.__version__,
        'arch':arch.__version__,'matplotlib':matplotlib.__version__,'PYTHONHASHSEED':os.environ.get('PYTHONHASHSEED')
    }
    errors=[f'{k}: {obs[k]} != {v}' for k,v in EXPECTED_RUNTIME.items() if obs[k]!=v]
    if obs['PYTHONHASHSEED']!='0': errors.append('PYTHONHASHSEED != 0')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float64)
    obs.update({
        'torch_num_threads':torch.get_num_threads(),
        'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'torch_default_dtype':str(torch.get_default_dtype()),
        'cuda_available':torch.cuda.is_available()
    })
    if torch.get_num_threads()!=1: errors.append('torch_num_threads != 1')
    if not torch.are_deterministic_algorithms_enabled(): errors.append('deterministic algorithms disabled')
    if torch.get_default_dtype()!=torch.float64: errors.append('torch default dtype != float64')
    if torch.cuda.is_available(): errors.append('CUDA unexpectedly available')
    result={'expected':EXPECTED_RUNTIME,'observed':obs,'status':'PASS' if not errors else 'HOLD','errors':errors}
    (OUT/'CH11_U06_RUNTIME_PREFLIGHT_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True))
    if errors: raise RuntimeError('; '.join(errors))

def protocol_gate():
    raw=PROTOCOL_PATH.read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    p=json.loads(raw)
    row={'path':str(PROTOCOL_PATH),'sha256':sha,'expected_sha256':EXPECTED_PROTOCOL_SHA,
         'protocol_id':p.get('protocol_id'),'sha_pass':sha==EXPECTED_PROTOCOL_SHA,
         'id_pass':p.get('protocol_id')=='CH11_STAGE5_V84_SEQUENCE_PROTOCOL_1'}
    pd.DataFrame([row]).to_csv(OUT/'CH11_U06_PROTOCOL_HASH_AND_INDEX_QA.csv',index=False)
    if not row['sha_pass'] or not row['id_pass']: raise RuntimeError('protocol mismatch')
    return p

def generate_dgp(seed,p):
    d=p['dgp']; n=int(d['generated_n']); burn=int(d['burn_in'])
    a,b,c=np.random.SeedSequence(seed).spawn(3)
    eta=np.random.default_rng(a).normal(0,float(d['sigma_eta']),n).astype(np.float64)
    eps=np.random.default_rng(b).normal(0,float(d['sigma_epsilon']),n).astype(np.float64)
    z=np.random.default_rng(c).normal(0,1,n).astype(np.float64)
    s=np.empty(n,np.float64); s[0]=float(d['mu'])
    mu,phi=float(d['mu']),float(d['phi'])
    for t in range(1,n):
        s[t]=mu+phi*(s[t-1]-mu)+eta[t]
    v=s+eps
    r=np.exp(s/2)*z
    return s[burn:],v[burn:],r[burn:]

def make_windows(z, origins, L=50):
    X=np.stack([z[o-L:o] for o in origins]).astype(np.float64)[...,None]
    y=np.array([z[o] for o in origins],dtype=np.float64)[:,None]
    return X,y

def ar_params(res,p):
    mp=dict(zip(res.param_names,np.asarray(res.params,float)))
    if 'const' not in mp: raise RuntimeError(f'ARIMA param mismatch {res.param_names}')
    return float(mp['const']),np.array([mp[f'ar.L{i}'] for i in range(1,p+1)],float),list(res.param_names)

def ar_forecast(window,const,phis,h):
    vals=list(map(float,window))
    for _ in range(h):
        pred=const+sum(phis[i]*(vals[-1-i]-const) for i in range(len(phis)))
        vals.append(float(pred))
    return vals[-1]

def select_ar(v):
    rows=[]
    for p in (1,2,5,10):
        res=ARIMA(v[:3600],order=(p,0,0),trend='c').fit()
        const,phis,names=ar_params(res,p)
        pred=np.array([ar_forecast(v[o-50:o],const,phis,1) for o in range(3600,4800)])
        target=v[3600:4800]
        rmse=float(np.sqrt(np.mean((pred-target)**2)))
        rows.append({'p':p,'validation_rmse_raw_v':rmse,'param_names':'|'.join(names)})
    rows=sorted(rows,key=lambda x:(x['validation_rmse_raw_v'],x['p']))
    return rows[0]['p'],rows

def fit_final_ar(v,p):
    res=ARIMA(v[:4800],order=(p,0,0),trend='c').fit()
    return ar_params(res,p)

def fit_ucm(v,mean,std):
    z=(v[:4800]-mean)/std
    res=UnobservedComponents(z,level=False,trend=False,irregular=True,autoregressive=1).fit(disp=False)
    mp=dict(zip(res.param_names,np.asarray(res.params,float)))
    req={'sigma2.irregular','sigma2.ar','ar.L1'}
    if not req.issubset(mp): raise RuntimeError(f'UCM param mismatch {res.param_names}')
    R,Q,phi=float(mp['sigma2.irregular']),float(mp['sigma2.ar']),float(mp['ar.L1'])
    if not (R>0 and Q>0 and abs(phi)<1): raise RuntimeError(f'UCM gate R={R} Q={Q} phi={phi}')
    return R,Q,phi,list(res.param_names),res

def kalman_forecast(window_std,R,Q,phi,h):
    a=0.0
    P=Q/(1-phi*phi)
    m=0.0
    for y in map(float,window_std):
        F=P+R
        K=P/F
        m=a+K*(y-a)
        C=(1-K)*P
        a=phi*m
        P=phi*phi*C+Q
    pred=m
    for _ in range(h):
        pred=phi*pred
    return float(pred)

class SmallRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.RNN(1,16,num_layers=1,nonlinearity='tanh',batch_first=True,bidirectional=False,dropout=0.0)
        self.readout=nn.Linear(16,1)
    def forward(self,x):
        out,_=self.rnn(x)
        return self.readout(out[:,-1,:])

def seeded_model(seed):
    torch.manual_seed(seed)
    return SmallRNN().to(dtype=torch.float64,device='cpu')

def loader(X,y,seed):
    ds=TensorDataset(torch.from_numpy(X),torch.from_numpy(y))
    g=torch.Generator(device='cpu'); g.manual_seed(seed)
    return DataLoader(ds,batch_size=64,shuffle=True,generator=g,num_workers=0,drop_last=False)

def predict(model,X,batch=256):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),batch):
            out.append(model(torch.from_numpy(X[i:i+batch])).cpu().numpy())
    return np.concatenate(out,axis=0)

def train_phase_a(v,mseed,sseed):
    mean=float(np.mean(v[:3600])); std=float(np.std(v[:3600],ddof=0)); z=(v-mean)/std
    Xtr,ytr=make_windows(z,range(50,3600))
    Xv,yv=make_windows(z,range(3600,4800))
    model=seeded_model(mseed)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0.0)
    lossfn=nn.MSELoss()
    dl=loader(Xtr,ytr,sseed)
    best=float('inf'); best_epoch=None; best_state=None; stale=0
    for epoch in range(1,201):
        model.train()
        for xb,yb in dl:
            opt.zero_grad(set_to_none=True)
            loss=lossfn(model(xb),yb)
            loss.backward()
            opt.step()
        vp=predict(model,Xv).reshape(-1)
        rmse=float(np.sqrt(np.mean((vp-yv.reshape(-1))**2)))
        if rmse < best-1e-6:
            best=rmse; best_epoch=epoch; best_state=deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
        if stale>=20: break
    if best_state is None: raise RuntimeError('RNN phase A failed')
    return best_epoch,best,mean,std

def train_phase_b(v,mseed,sseed,epochs):
    mean=float(np.mean(v[:4800])); std=float(np.std(v[:4800],ddof=0)); z=(v-mean)/std
    X,y=make_windows(z,range(50,4800))
    model=seeded_model(mseed)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0.0)
    lossfn=nn.MSELoss()
    dl=loader(X,y,sseed)
    for _ in range(epochs):
        model.train()
        for xb,yb in dl:
            opt.zero_grad(set_to_none=True)
            loss=lossfn(model(xb),yb)
            loss.backward()
            opt.step()
    return model,mean,std

def rnn_forecast(model,raw_window,mean,std,h):
    w=((np.asarray(raw_window,np.float64)-mean)/std).copy()
    model.eval()
    with torch.no_grad():
        for _ in range(h):
            pred=float(model(torch.from_numpy(w.reshape(1,50,1))).item())
            w=np.concatenate([w[1:],np.array([pred])])
    return float(w[-1]*std+mean)

def metric(pred,target):
    e=np.asarray(pred)-np.asarray(target)
    return float(np.sqrt(np.mean(e*e))),float(np.mean(np.abs(e)))

def garch_attempt(r,s,rep):
    row={'replicate':rep,'status':'PASS','error':''}
    try:
        res=arch_model(r[:4800],mean='Zero',vol='GARCH',p=1,o=0,q=1,dist='normal',rescale=False).fit(disp='off',show_warning=False)
        row['convergence_flag']=getattr(res,'convergence_flag',None)
        for k,v in res.params.items(): row[f'param_{k}']=float(v)
        f=res.forecast(horizon=1,reindex=False)
        row['last_pretest_variance_forecast']=float(f.variance.iloc[-1,0])
        row['last_pretest_latent_variance']=float(np.exp(s[4799]))
    except Exception as e:
        row['status']='EXTENSION-ONLY HOLD'
        row['error']=repr(e)
    return row

def main():
    runtime_gate()
    protocol=protocol_gate()
    assert protocol['indexing']['lookback']==50
    assert protocol['indexing']['horizons']==[1,5,20]
    assert len(range(50,3600))==3550
    assert len(range(3600,4800))==1200
    assert len(range(50,4800))==4750

    pred_rows=[]; result_rows=[]; seed_rows=[]; select_rows=[]; ucm_rows=[]; rnn_rows=[]; garch_rows=[]
    for rep,(dseed,mseed,sseed) in enumerate(zip(protocol['dgp']['dgp_seeds'],protocol['rnn']['model_seeds'],protocol['rnn']['shuffle_seeds']),start=1):
        print(f'REPLICATE {rep}/5 seed={dseed}',flush=True)
        s,v,r=generate_dgp(dseed,protocol)
        seed_rows.append({'replicate':rep,'dgp_seed':dseed,'rnn_model_seed':mseed,'rnn_shuffle_seed':sseed,'n_retained':len(v)})

        selected_p,ar_sel=select_ar(v)
        for x in ar_sel:
            select_rows.append({'replicate':rep,'family':'AR','selected':x['p']==selected_p,**x})
        ar_const,ar_phis,ar_names=fit_final_ar(v,selected_p)

        E_star,phaseA_rmse,train_mean,train_std=train_phase_a(v,mseed,sseed)
        select_rows.append({'replicate':rep,'family':'RNN','selected':True,'p':'','validation_rmse_raw_v':'','param_names':'','E_star':E_star,'validation_rmse_std':phaseA_rmse})
        model,pre_mean,pre_std=train_phase_b(v,mseed,sseed,E_star)
        rnn_rows.append({'replicate':rep,'model_seed':mseed,'shuffle_seed':sseed,'E_star':E_star,'phaseA_best_rmse_std':phaseA_rmse,
                         'phaseA_scaler_mean':train_mean,'phaseA_scaler_std_ddof0':train_std,'phaseB_scaler_mean':pre_mean,
                         'phaseB_scaler_std_ddof0':pre_std,'dtype':str(next(model.parameters()).dtype),
                         'device':str(next(model.parameters()).device),'threads':torch.get_num_threads(),
                         'deterministic':torch.are_deterministic_algorithms_enabled()})

        R,Q,phi,names,ures=fit_ucm(v,pre_mean,pre_std)
        ucm_rows.append({'replicate':rep,'param_names':'|'.join(names),'sigma2_irregular':R,'sigma2_ar':Q,'ar_L1':phi,
                         'stationary':abs(phi)<1,'fit_converged':bool(getattr(ures,'mle_retvals',{}).get('converged',True))})
        garch_rows.append(garch_attempt(r,s,rep))

        for h in (1,5,20):
            origins=range(4800,6001-h)
            targets=np.array([v[o+h-1] for o in origins],dtype=float)
            arpred=[]; sspred=[]; rnpred=[]
            for o in origins:
                w=v[o-50:o]
                if len(w)!=50: raise RuntimeError('window length gate')
                pa=ar_forecast(w,ar_const,ar_phis,h)
                ps=kalman_forecast((w-pre_mean)/pre_std,R,Q,phi,h)*pre_std+pre_mean
                pr=rnn_forecast(model,w,pre_mean,pre_std,h)
                arpred.append(pa); sspred.append(ps); rnpred.append(pr)
                target=v[o+h-1]
                for fam,pred in [('AR',pa),('STATE_SPACE',ps),('RNN',pr)]:
                    pred_rows.append({'replicate':rep,'dgp_seed':dseed,'model':fam,'horizon':h,'origin_t':o,'target_t':o+h,'forecast':pred,'target':target})
            for fam,preds in [('AR',arpred),('STATE_SPACE',sspred),('RNN',rnpred)]:
                rmse,mae=metric(preds,targets)
                result_rows.append({'replicate':rep,'dgp_seed':dseed,'model':fam,'horizon':h,'n_origins':len(targets),'RMSE':rmse,'MAE':mae,'provenance':'PROJECT-REPRODUCED'})

    pd.DataFrame(seed_rows).to_csv(OUT/'CH11_U06_SEED_RUN_LEDGER.csv',index=False)
    pd.DataFrame(select_rows).to_csv(OUT/'CH11_U06_MODEL_SELECTION_LEDGER.csv',index=False)
    pd.DataFrame(pred_rows).to_csv(OUT/'CH11_U06_CORE_PREDICTIONS.csv',index=False)
    raw=pd.DataFrame(result_rows)
    raw.to_csv(OUT/'CH11_U06_CORE_RESULTS_RAW.csv',index=False)
    pd.DataFrame(ucm_rows).to_csv(OUT/'CH11_U06_UCM_MAPPING_AND_STATIONARITY_QA.csv',index=False)
    pd.DataFrame(rnn_rows).to_csv(OUT/'CH11_U06_RNN_DETERMINISM_QA.csv',index=False)
    pd.DataFrame(garch_rows).to_csv(OUT/'CH11_U06_GARCH_EXTENSION_QA.csv',index=False)

    groups=[]
    for (model,h),g in raw.groupby(['model','horizon']):
        for metric_name in ('RMSE','MAE'):
            x=g[metric_name].to_numpy(float)
            groups.append({'model':model,'horizon':h,'metric':metric_name,'mean':float(np.mean(x)),
                           'sample_sd_ddof1':float(np.std(x,ddof=1)),'median':float(np.median(x)),'n_completed':len(x)})
    pd.DataFrame(groups).to_csv(OUT/'CH11_U06_CORE_RESULTS_AGGREGATE.csv',index=False)

    qa={'status':'PASS','replicates':5,'all_core_rows':len(raw),'prediction_rows':len(pred_rows),
        'origin_counts':{'k1':1200,'k5':1196,'k20':1181},'no_test_refit':True,
        'same_information_window':50,'s05_2_prose_drafted':False,'all_core_replicates_completed':len(raw)==45}
    (OUT/'CH11_U06_CORE_RUN_QA.json').write_text(json.dumps(qa,indent=2,sort_keys=True))
    if not qa['all_core_replicates_completed']: raise RuntimeError('core aggregate gate failed')
    print(json.dumps(qa,indent=2),flush=True)

if __name__=='__main__':
    main()
