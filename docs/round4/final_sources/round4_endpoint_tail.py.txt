"""Training-endpoint capacity diagnostic, never an unseen-day prediction score."""
import json
import numpy as np
import pandas as pd
import torch
from common import ROOT,write_json
from evaluate import metrics
from round4 import OUT,VARIANTS,create,load_data,folder,sha
from round4_evaluate import bank

@torch.inference_mode()
def main():
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    out=OUT/'endpoint_tail';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Completed diagnostic already exists')
    rows=[]
    for name in ['native']+list(VARIANTS):
        dest=ROOT/'results/round2/native_s42' if name=='native' else OUT/f'{name}_s42'
        cfg=json.loads((dest/'config.json').read_text());cfg.update(encoder_norm=cfg.get('encoder_norm','batch'),block_skip=cfg.get('block_skip',False))
        state=torch.load(dest/'final.pt',map_location='cpu',weights_only=False)
        assert state['step']==100000
        m,d=create(cfg,1000);m.load_state_dict(state['model']);m.cuda().eval();del state
        tr,_,_=load_data(cfg);x=np.asarray(tr.X)
        assert sha(folder(cfg)/'train.h5ad')==cfg['train_data_sha256']
        mask=x[:,tr.var_names.get_loc('T')]>3
        truth=x[mask];groups=tr.obs.Group.to_numpy();labels=groups[mask]
        assert len(truth)>1
        z=m.encoder(torch.tensor(x,device='cuda'));target=z[torch.tensor(mask,device='cuda')]
        centers={g:z[torch.tensor(groups==g,device='cuda')].mean(0) for g in [0,3]}
        center=torch.stack([centers[int(g)] for g in labels])
        assert len(truth)<=600
        noise=bank(600,tr.n_vars,1000)['noise'][:len(truth)]
        for mode,condition in [('own_rna_encoded',target),('same_day_latent_center',center)]:
            pred=d.ddim_sample_loop(m,truth.shape,noise=noise.clone(),model_kwargs={'z_mod':condition},clip_denoised=False).cpu().numpy()
            row=dict(model=name,mode=mode,training_rows=len(truth),unique_training_ids=len(set(tr.obs_names[mask])),
                **metrics(truth,pred),paired_mse=float(np.mean((truth.astype(float)-pred)**2)))
            for gene in ['T','DKK1','DKK4']:
                j=tr.var_names.get_loc(gene)
                row[gene+'_observed']=float(truth[:,j].astype(float).mean());row[gene+'_predicted']=float(pred[:,j].astype(float).mean())
            rows.append(row)
            np.savez_compressed(out/f'{name}_{mode}.npz',prediction=pred,truth=truth,genes=np.array(tr.var_names,dtype=str),cell_ids=np.array(tr.obs_names[mask],dtype=str))
            pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            print('TAIL',name,mode,row['T_observed'],row['T_predicted'],row['r2'],flush=True)
        del m,d,z
    write_json(dict(rows=len(rows),models=['native']+list(VARIANTS),selection='Training endpoint T > 3; threshold and comparison frozen before new middle-day evaluation',
        shared_xT_within_model=True,updates=0,
        interpretation='RNA-conditioned training capacity control; neither generation of an unseen state nor prediction of its population expansion'),out/'complete.json')

if __name__=='__main__':main()
