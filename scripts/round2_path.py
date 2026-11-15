"""Diagnostic clock scan; never choose the forecasting clock using day1 truth."""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import round2 as r

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision('highest')
dest=r.OUT/'native_s42'
cfg=json.loads((dest/'config.json').read_text())
tr,va,te=r.load_data(cfg)
state=torch.load(dest/'final.pt',map_location='cpu',weights_only=False)
m,d=r.create(cfg,1000);m.load_state_dict(state['model']);m.cuda().eval();del state
out=r.OUT/'native_path';out.mkdir(exist_ok=True)
rows=[]
with torch.inference_mode():
    z=m.encoder(torch.tensor(tr.X,device='cuda'))
    zz={i:z[torch.tensor(tr.obs.Group.to_numpy()==i,device='cuda')] for i in [0,3]}
    zm={i:v.mean(0) for i,v in zz.items()}
    origin=m.encoder(torch.tensor(te.X[te.obs.Group==0],device='cuda')).mean(0)
    assert json.loads((dest/'eval_final_1000/selection.json').read_text())['selected']['mode']=='empirical'
    for k in range(7):
        alpha=k/6
        if k%2==0:
            path=dest/f'eval_final_1000/selected_day{k//2}_rep0.npz'
            pred=np.load(path)['prediction'];source=str(path.relative_to(r.ROOT))
        else:
            r.seed_all(1000)
            a=zz[0][torch.randint(len(zz[0]),(600,),device='cuda')]-zm[0]
            b=zz[3][torch.randint(len(zz[3]),(600,),device='cuda')]-zm[3]
            latent=origin+alpha*(zm[3]-zm[0])+((1-alpha)*a+alpha*b)/np.sqrt((1-alpha)**2+alpha**2)
            pred=d.ddim_sample_loop(m,(600,tr.n_vars),model_kwargs={'z_mod':latent}).cpu().numpy()
            path=out/f'alpha_{alpha:.6f}.npz'
            np.savez_compressed(path,prediction=pred,genes=np.array(tr.var_names,dtype=str))
            source=str(path.relative_to(r.ROOT))
        row=dict(alpha=alpha,nominal_day=3*alpha,source=source,
            T=float(pred[:,tr.var_names.get_loc('T')].astype(float).mean()),
            **r.metrics(te.X[te.obs.Group==1],pred,tr.X[tr.obs.Group==0]))
        rows.append(row)
        pd.DataFrame(rows).to_csv(out/'clock_scan.csv',index=False)
f=pd.DataFrame(rows)
true=[float(te.X[te.obs.Group==day,tr.var_names.get_loc('T')].astype(float).mean()) for day in range(4)]
fig,ax=plt.subplots(figsize=(6,4));ax.plot(range(4),true,'ko-',label='Observed')
ax.plot(f.nominal_day,f['T'],'o-',label='Generated: 7 diagnostic clock points')
ax.set(xlabel='Day (3 x interpolation fraction)',ylabel='T mean, original expression units',xticks=range(4));ax.legend()
fig.tight_layout();fig.savefig(out/'clock_scan.png',dpi=180);plt.close(fig)
r.write_json(dict(diagnostic_only=True,forecast_alpha_remains=1/3,points=7,
    max_T_on_sampled_grid=float(f['T'].max()),observed_day1_T=true[1],
    warning='A finite grid is not a proof about every continuous interpolation fraction'),out/'complete.json')
print((out/'complete.json').read_text())
