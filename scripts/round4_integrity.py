"""Independent saved-artifact checks; never regenerate or change predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import ROOT,write_json
from round4 import OUT,VARIANTS,load_data,folder,sha
from round4_evaluate import scores,CONTROLS

def main():
    torch.set_num_threads(2)
    log=ROOT/'logs/round4'
    training=json.loads((log/'training_complete.json').read_text())
    assert training['all_success'] and training['updates']==400000
    queue=json.loads((log/'queue.json').read_text())
    assert sha(OUT/'frozen_protocol.json')==queue['frozen_protocol_sha256']
    frozen=json.loads((OUT/'frozen_protocol.json').read_text())
    for panel,key in [('source','source_hashes'),('fullsource','full_hashes')]:
        for part,h in frozen[key].items():assert sha(folder({'panel':panel})/f'{part}.h5ad')==h
    evaluation=json.loads((log/'evaluation_complete.json').read_text())
    assert evaluation['all_success'] and all(r['exit_code']==0 for r in evaluation['jobs'])
    models=[];predictions=[];tables=[]
    reference,_,_=load_data({'panel':'source'})
    for name in VARIANTS:
        dest=OUT/f'{name}_s42'
        status=json.loads((dest/'status.json').read_text())
        assert status['status']=='trained' and status['step']==status['target']==100000
        cfg=json.loads((dest/'config.json').read_text())
        assert cfg['seed']==42 and cfg['parameters']==52142343
        assert sha(folder(cfg)/'train.h5ad')==cfg['train_data_sha256']
        for p,h in cfg['source_sha256'].items():assert sha(ROOT/'upstream/Squidiff/Squidiff'/p)==h
        for p,h in cfg['adapter_sha256'].items():assert sha(ROOT/'scripts'/p)==h
        s=torch.load(dest/'final.pt',map_location='cpu',weights_only=False)
        assert s['step']==100000 and all(torch.isfinite(v).all() for v in s['model'].values())
        counters={k:int(v) for k,v in s['model'].items() if k.endswith('num_batches_tracked')}
        if name=='layernorm':assert counters=={}
        else:assert len(counters)==2 and set(counters.values())=={100000}
        latest=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
        assert latest['step']==100000
        assert all(torch.equal(v,latest['model'][k]) for k,v in s['model'].items())
        steps={int(v['step']) for v in latest['optimizer']['state'].values()}
        assert steps=={100000}
        assert all(torch.isfinite(v).all() for v in latest['ema'].values())
        tr,va,te=load_data(cfg)
        assert set(tr.obs.Group)=={0,3}
        assert set(tr.obs_names).isdisjoint(va.obs_names)
        assert set(tr.obs_names).isdisjoint(te.obs_names[te.obs.Group.isin([1,2])])
        if name=='fullsource':assert set(tr.obs_names).isdisjoint(te.obs_names) and tr.n_obs==16580
        models.append(dict(model=name,step=100000,optimizer_steps=list(steps),bn_counters=counters,
            train_rows=tr.n_obs,unique_cells=len(set(tr.obs_names)),all_weights_finite=True,
            final_matches_latest=True,final_sha256=sha(dest/'final.pt'),
            train_sha256=cfg['train_data_sha256'],seconds=status['seconds']))
        del s,latest
    expected=list(VARIANTS)+['native','bncal','ema','emacal','finalclip','paper_cellnoise']
    for name in expected:
        base=OUT/f'{name}_s42' if name in VARIANTS else OUT/'inference_controls'/name
        out=base/'eval_final_1000'
        complete=json.loads((out/'complete.json').read_text())
        assert complete['args']['sample_steps']==1000 and complete['args']['cells']==600
        cfg=complete['config'];tr,va,te=load_data(cfg)
        assert sha(folder(cfg)/'train.h5ad')==complete['train_data_sha256']
        weight=(OUT/f'{name}_s42/final.pt' if name in VARIANTS else
                ROOT/'results/round2/native_s42'/('latest.pt' if name in ['ema','emacal'] else 'final.pt'))
        assert sha(weight)==complete['weight_sha256']
        f=pd.read_csv(out/'metrics.csv');assert len(f)==complete['rows']==10
        selection=json.loads((out/'selection.json').read_text())
        candidates=pd.read_csv(out/'candidates.csv')
        best=candidates.sort_values('wasserstein').iloc[0]
        assert best['mode']==selection['selected']['mode']
        assert abs(best['sigma']-selection['selected']['sigma'])<1e-8
        assert selection['intermediate_RNA_used'] is False
        for _,r in f.iterrows():
            p=out/f'{r.method}_day{int(r.day)}.npz'
            b=np.load(p,allow_pickle=False);pred=b['prediction']
            assert np.array_equal(b['genes'],tr.var_names) and pred.shape==(600,203)
            assert np.isfinite(pred).all()
            mm=scores(np.asarray(te.X[te.obs.Group==r.day]),pred,tr,reference)
            for key in ['pearson','r2','notebook_reverse_r2','mean_mse','mean_gene_wasserstein',
                        'energy_distance','T_pred','T_above3','transient_tp','transient_fp']:
                assert abs(mm[key]-r[key])<1e-8,(name,r.method,r.day,key,mm[key],r[key])
            predictions.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),shape=list(pred.shape),metrics_recomputed=True))
        tables.append(dict(model=name,rows=len(f),candidate_count=len(candidates),
            train_test_overlap=len(set(tr.obs_names)&set(te.obs_names)),
            validation_train_overlap=0,intermediate_training_overlap=0))
    tails=[]
    td=OUT/'endpoint_tail';tail_complete=json.loads((td/'complete.json').read_text())
    tf=pd.read_csv(td/'metrics.csv');assert len(tf)==tail_complete['rows']==10
    for _,r in tf.iterrows():
        p=td/f'{r.model}_{r["mode"]}.npz';b=np.load(p,allow_pickle=False)
        assert np.array_equal(b['genes'],reference.var_names)
        pred=b['prediction'];truth=b['truth']
        assert pred.shape==truth.shape and pred.shape[1]==203 and np.isfinite(pred).all()
        assert np.all(truth[:,reference.var_names.get_loc('T')]>3)
        cfg=json.loads(((ROOT/'results/round2/native_s42' if r.model=='native' else OUT/f'{r.model}_s42')/'config.json').read_text())
        tr,_,_=load_data(cfg);mask=np.asarray(tr.X)[:,tr.var_names.get_loc('T')]>3
        assert np.array_equal(tr.X[mask],truth)
        assert np.array_equal(np.array(tr.obs_names[mask],dtype=str),b['cell_ids'])
        from evaluate import metrics
        mm=metrics(truth,pred)
        for key in ['r2','mean_mse','energy_distance']:assert abs(mm[key]-r[key])<1e-8
        j=reference.var_names.get_loc('T')
        assert abs(float(pred[:,j].astype(float).mean())-r.T_predicted)<1e-8
        tails.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),shape=list(pred.shape),metrics_recomputed=True))
    assert {str(p.relative_to(ROOT)) for p in OUT.rglob('*.npz')}=={r['path'] for r in predictions+tails}
    archive=json.loads((ROOT/'docs/archive_manifest.json').read_text())
    for item in archive['files']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    retired=json.loads((log/'checkpoint_retirement.json').read_text())
    assert retired['status']=='complete'
    for item in retired['records']:
        assert not (ROOT/item['path']).exists()
        assert sha(ROOT/item['final_weight'])==item['final_sha256']
    assert json.loads((OUT/'native_replay_compatibility.json').read_text())['all_exact']
    write_json(dict(models=models,predictions=predictions,evaluations=tables,
        total_updates=400000,total_predictions=len(predictions),
        endpoint_tail_predictions=tails,supplementary_predictions=len(tails),
        historical_archive_files_verified=len(archive['files']),historical_archive_unchanged=True,
        retired_optimizer_snapshots=len(retired['records']),retained_old_final_hashes_match=True,
        native_full_fixed_replay_exact=True),OUT/'final_integrity.json')
    print('ROUND4_INTEGRITY_OK',len(models),'trained models',len(predictions),'predictions',flush=True)

if __name__=='__main__':main()
