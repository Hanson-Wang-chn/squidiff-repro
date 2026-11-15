"""Cell-disjoint endpoint diagnostic for the original author's BVO model.

The input RNA of each evaluated cell is provided to the encoder. This tests
conditional generation, not prediction without the target transcriptome.
"""
import json
from pathlib import Path
import anndata as ad
import numpy as np
import torch
from common import ROOT,build_model,dense,seed_all,write_json
from evaluate import metrics


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    a=ad.read_h5ad('/home/whs/squidiff-ckpt/VO_trained_adata.h5ad')
    full=ad.read_h5ad(ROOT/'data/cache/ipsc_full.h5ad',backed='r')
    genes=full.var_names.get_indexer(a.var_names)
    assert (genes>=0).all()
    # Backed dense h5py matrices allow only one fancy index per operation.
    ids=a.obs_names[a.obs.day==0]
    order=full.obs_names.get_indexer(ids)
    sorted_order=np.argsort(order)
    xx=np.asarray(full.X[order[sorted_order],:])[:,genes][np.argsort(sorted_order)]
    assert np.array_equal(xx,dense(a[a.obs.day==0]))
    pool=np.flatnonzero((full.obs.Group.to_numpy()==0)&~full.obs_names.isin(a.obs_names))
    rng=np.random.default_rng(20261115)
    pick=np.sort(rng.choice(pool,300,replace=False))
    truth=np.asarray(full.X[pick,:])[:,genes].astype(np.float32)
    unseen_ids=full.obs_names[pick].to_numpy(dtype=str)
    assert set(unseen_ids).isdisjoint(a.obs_names)
    full.file.close()
    seed_all(1000)
    m,d=build_model(a.n_vars,1000)
    m.load_state_dict(torch.load('/home/whs/squidiff-ckpt/VO_diff_model.pt',
        map_location='cpu',weights_only=True),strict=True)
    m.cuda().eval()
    z=m.encoder(torch.tensor(truth,device='cuda'))
    pred=d.ddim_sample_loop(m,truth.shape,model_kwargs={'z_mod':z},clip_denoised=False).cpu().numpy()
    train=dense(a[a.obs.day==0])
    mean=np.repeat(train.mean(0)[None,:],len(pred),axis=0)
    scores={label:dict(**metrics(truth,p),paired_mse=float(np.mean((truth-p)**2)))
            for label,p in [('encoded_unseen_ipsc',pred),('training_ipsc_mean',mean)]}
    dest=ROOT/'results/round3/author/unseen_ipsc';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'prediction.npz',prediction=pred,truth=truth,
        truth_ids=unseen_ids,genes=a.var_names.to_numpy(dtype=str))
    write_json(dict(scores=scores,provided_training_id_overlap=0,cells=len(pred),
        author_training_ipsc_raw_alignment_max_error=0.,sampling_steps=1000,
        scope='Cell-disjoint day0 conditional generation; target RNA supplied to encoder; no donor holdout'),
        dest/'complete.json')
    print(json.dumps(scores),flush=True)


if __name__=='__main__':main()
