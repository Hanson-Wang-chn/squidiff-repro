"""Call the original released interpolation method directly on the final source model."""
import argparse
import hashlib
import numpy as np
import pandas as pd
import anndata as ad
import torch
from common import ROOT,setup,seed_all,build_model,write_json
from evaluate import metrics
from sample_squidiff import sampler


@torch.inference_mode()
def main(args):
    setup();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    tr=ad.read_h5ad(ROOT/args.data_root/args.dataset/'train.h5ad')
    te=ad.read_h5ad(ROOT/args.data_root/args.dataset/'test.h5ad')
    dest=ROOT/'results'/args.dataset/args.run
    state=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
    assert state['step']==100000
    m,d=build_model(tr.n_vars,1000);m.load_state_dict(state['model']);m.cuda().eval()
    s=sampler.__new__(sampler);s.model=m;s.sample_fn=d.ddim_sample_loop
    z=m.encoder(torch.tensor(tr.X,device='cuda'));g=tr.obs.Group.to_numpy()
    zm={k:z[torch.tensor(g==k,device='cuda')].mean(0) for k in np.unique(g)}
    means={k:tr.X[g==k].astype(float).mean(0) for k in np.unique(g)}
    out=dest/'official_api';out.mkdir(exist_ok=True);rows=[]
    if args.dataset=='ipsc':
        origin=m.encoder(torch.tensor(te.X[te.obs.Group==0],device='cuda'))
        labels={0:'day0_no_shift_null',1:'day1',2:'day2',3:'day3_from_day0'}
        cases=[(labels[day],origin,zm[3]-zm[0],day/3,True,te.X[te.obs.Group==day],
                (1-day/3)*means[0]+day/3*means[3]) for day in [0,1,2,3]]
    else:
        cases=[('AB',zm[0],zm[1]+zm[2]-zm[0],1.,False,te.X,means[1]+means[2]-means[0])]
    existing=out/'metrics.csv'
    if existing.exists():rows=pd.read_csv(existing).to_dict('records')
    for target,origin,direction,scale,add_noise,truth,base in cases:
        if any(r['target']==target for r in rows) and (out/f'{target}.npz').exists():continue
        seed_all(1000)
        pred=s.interp_with_direction(z_sem_origin=origin,gene_size=tr.n_vars,direction=direction,
                                    scale=scale,add_noise_term=add_noise).cpu().numpy()
        row=dict(dataset=args.dataset,run=args.run,target=target,checkpoint_step=state['step'],
                 generated_cells=len(pred),**metrics(truth,pred,tr.X[g==0]))
        exact_mse=float(np.mean((truth.astype(float).mean(0)-base)**2))
        row['relative_mse_gain']=1-row['mean_mse']/exact_mse
        if args.dataset=='ipsc':
            j=tr.var_names.get_loc('T');row.update(predicted_T_mean=float(pred[:,j].mean()),true_T_mean=float(truth[:,j].mean()))
        rows.append(row);np.savez_compressed(out/f'{target}.npz',prediction=pred,genes=np.array(tr.var_names,dtype=str))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    source=ROOT/'upstream/Squidiff/sample_squidiff.py'
    write_json(dict(args=vars(args),api='Unmodified sample_squidiff.sampler.interp_with_direction',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        includes_no_shift_null_and_forward_day3=args.dataset=='ipsc',
        purpose='Confirm that local latent algebra and Gaussian sampling do not explain the qualitative failure; NumPy is used exactly as in the official method.'),out/'complete.json')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['ipsc','norman'])
    p.add_argument('--run',default='source_fp32_s42');p.add_argument('--data-root',required=True)
    main(p.parse_args())
