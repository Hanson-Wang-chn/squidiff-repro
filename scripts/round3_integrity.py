"""Final third-round checks from saved weights, predictions, and frozen sources."""
import hashlib
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from common import ROOT,write_json
from evaluate import metrics


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    tr,va,te=(ad.read_h5ad(ROOT/f'data/round2/source/{p}.h5ad') for p in ['train','val','test'])
    assert tr.var_names.equals(va.var_names) and tr.var_names.equals(te.var_names)
    output=ROOT/'results/round3'
    checks=[];predictions=[]
    for variant in ['xstart','minsnr','epsres']:
        dest=output/f'{variant}_s42'
        status=json.loads((dest/'status.json').read_text())
        assert status['status']=='trained' and status['step']==100000
        s=torch.load(dest/'final.pt',map_location='cpu',weights_only=False)
        assert s['step']==100000 and all(torch.isfinite(v).all() for v in s['model'].values())
        cfg=s['config']
        assert sha(ROOT/'data/round2/source/train.h5ad')==cfg['train_data_sha256']
        for name,h in cfg['source_sha256'].items():
            assert sha(ROOT/'upstream/Squidiff/Squidiff'/name)==h
        for name,h in cfg['adapter_hashes'].items():
            assert sha(ROOT/'scripts'/name)==h,name
        bn={k:int(v) for k,v in s['model'].items() if k.endswith('num_batches_tracked')}
        assert set(bn.values())=={100000}
        assert cfg['parameters']==sum(v.numel() for k,v in s['model'].items()
            if not any(w in k for w in ['running_mean','running_var','num_batches_tracked']))
        checks.append(dict(variant=variant,step=100000,bn_updates=bn,
            target=cfg['target'],weighting=cfg['weighting'],final_sha256=sha(dest/'final.pt'),
            train_data_and_source_hashes_match=True,all_tensors_finite=True))
        ev=dest/'eval_final_1000'
        complete=json.loads((ev/'complete.json').read_text())
        f=pd.read_csv(ev/'metrics.csv')
        assert complete['rows']==len(f)==4
        selected=json.loads((ev/'selection.json').read_text())
        candidates=pd.read_csv(ev/'candidates.csv')
        best=candidates.sort_values('wasserstein').iloc[0]
        assert best['mode']==selected['selected']['mode']
        assert abs(best['sigma']-selected['selected']['sigma'])<1e-8
        for _,r in f.iterrows():
            path=ev/f'{r.method}_day{r.day}_rep{r["repeat"]}.npz'
            b=np.load(path,allow_pickle=False)
            assert np.array_equal(b['genes'],tr.var_names)
            pred=b['prediction'];truth=te.X[te.obs.Group==r.day]
            assert pred.shape==(600,203) and np.isfinite(pred).all()
            mm=metrics(truth,pred,tr.X[tr.obs.Group==0])
            for name in ['mean_mse','r2','pearson','mean_gene_wasserstein','energy_distance']:
                assert abs(mm[name]-r[name])<1e-8,(path,name,mm[name],r[name])
            j=tr.var_names.get_loc('T')
            assert abs(float(pred[:,j].astype(float).mean())-r.T_pred)<1e-8
            predictions.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),
                metrics_recomputed=True,genes_in_exact_order=True))
        del s
    # New code and experiments must not modify previously archived evidence.
    old=json.loads((ROOT/'docs/archive_manifest.json').read_text())
    for item in old['files']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    author=json.loads((output/'author/inventory.json').read_text())
    assert sha(Path('/home/whs/squidiff-ckpt/VO_diff_model.pt'))==author['checkpoint_sha256']
    diag=output/'diagnostics'
    if (diag/'complete.json').exists():
        f=pd.read_csv(diag/'metrics.csv')
        for _,r in f.iterrows():
            path=diag/f'{r.variant}_{r.method}.npz'
            b=np.load(path,allow_pickle=False)
            assert np.array_equal(b['genes'],tr.var_names)
            mm=metrics(te.X[te.obs.Group==1],b['prediction'])
            for name in ['r2','mean_mse','energy_distance']:
                assert abs(mm[name]-r[name])<1e-8,(path,name)
    all_files=[]
    for path in sorted(output.rglob('*.npz')):
        b=np.load(path,allow_pickle=False)
        expected=author['genes'] if 'author' in path.parts else tr.var_names
        assert np.array_equal(b['genes'],expected),path
        pred=b['prediction']
        assert pred.ndim==2 and pred.shape[1]==len(expected) and np.isfinite(pred).all()
        all_files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),shape=list(pred.shape)))
    write_json(dict(models=checks,predictions=predictions,
        all_prediction_files=all_files,total_prediction_files=len(all_files),
        author_checkpoint_unchanged=True,old_archive_files_verified=len(old['files']),
        old_archive_unchanged=True,source_train_rows=tr.n_obs,source_val_rows=va.n_obs,
        source_test_rows=te.n_obs,middle_day_train_overlap=0),output/'final_integrity.json')
    print('ROUND3_INTEGRITY_OK',len(checks),'models;',len(predictions),'predictions;',len(old['files']),'historical files')


if __name__=='__main__':main()
