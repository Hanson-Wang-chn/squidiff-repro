"""Test whether a transient peak depends on inconsistent endpoint/middle sampling."""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import anndata as ad
from common import ROOT,setup,seed_all,build_model,write_json,record
from evaluate import metrics

@torch.inference_mode()
def main(args):
    setup();torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    p=ROOT/args.data_root/'ipsc';tr=ad.read_h5ad(p/'train.h5ad');te=ad.read_h5ad(p/'test.h5ad')
    s=torch.load(ROOT/'results/ipsc'/args.run/'latest.pt',map_location='cpu',weights_only=False)
    m,d=build_model(tr.n_vars,1000);m.load_state_dict(s['model']);m.cuda().eval()
    z=m.encoder(torch.tensor(tr.X,device='cuda'));g=tr.obs.Group.to_numpy();tg=te.obs.Group.to_numpy()
    means={i:z[torch.tensor(g==i,device='cuda')].mean(0) for i in [0,3]}
    z0=m.encoder(torch.tensor(te.X[tg==0],device='cuda'));z3=m.encoder(torch.tensor(te.X[tg==3],device='cuda'))
    direction=means[3]-means[0];j=tr.var_names.get_loc('T');rows=[]
    dest=ROOT/'results/ipsc'/args.run/('sampling_nulls' if not args.tag else 'sampling_nulls_'+args.tag);dest.mkdir(exist_ok=True)
    def draw(zz):return d.ddim_sample_loop(m,(len(zz),tr.n_vars),model_kwargs={'z_mod':zz}).cpu().numpy()
    def save_prediction(pred,schema,sigma,day,rep):
        label=f'{schema}_sigma{sigma:g}_day{day}_rep{rep}'
        np.savez_compressed(dest/f'{label}.npz',prediction=pred,genes=np.array(tr.var_names,dtype=str))
        marker_values={f'predicted_{name}':float(pred[:,tr.var_names.get_loc(name)].mean())
                       for name in ['NANOG','GATA6'] if name in tr.var_names}
        return {**marker_values,**metrics(te.X[tg==day],pred,tr.X[g==0])}
    for rep in range(args.repeats):
        for day,zz in [(0,z0),(3,z3)]:
            seed_all(2000+rep);pred=draw(zz)
            rows.append(dict(repeat=rep,schema='source_endpoint_reconstruction',sigma=0.,day=day,
                             predicted_T=float(pred[:,j].mean()),observed_T=float(te.X[tg==day,j].mean()),
                             **save_prediction(pred,'source_endpoint_reconstruction',0.,day,rep)))
        for sigma in [float(v) for v in args.sigmas.split(',')]:
            for day in [0,1,2,3]:
                seed_all(2000+rep)
                zz=z0.mean(0)+day/3*direction+sigma*torch.randn((600,60),device='cuda')
                pred=draw(zz)
                row=dict(repeat=rep,schema='matched_centroid_and_noise_all_days',sigma=sigma,day=day,
                         predicted_T=float(pred[:,j].mean()),observed_T=float(te.X[tg==day,j].mean()),
                         **save_prediction(pred,'matched_centroid_and_noise_all_days',sigma,day,rep))
                rows.append(row);print(json.dumps(row),flush=True)
                pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False)
    write_json({'args':vars(args),'no_shift_null':'day=0 with sigma>0; same noisy centroid sampler without a differentiation shift',
                'paired_noise':'identical seed at each day isolates the semantic shift'},dest/'complete.json')
    record(f'采样负对照 ipsc/{args.run} 完成：统一四天的潜变量抽样方式，分离方向移动与额外潜变量噪声。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='v1');p.add_argument('--data-root',default='data/processed');p.add_argument('--repeats',type=int,default=1);p.add_argument('--sigmas',default='0,0.5,0.7');p.add_argument('--tag',default='');main(p.parse_args())
