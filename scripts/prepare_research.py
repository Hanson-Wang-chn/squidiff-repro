"""Build controlled protocol variants without overwriting v1 data."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from common import ROOT, write_json, record

def full_ipsc():
    path=ROOT/'data/cache/ipsc_full.h5ad'
    if path.exists():return ad.read_h5ad(path)
    meta=pd.read_csv(ROOT/'data/raw/ipsc_cell_metadata_cols.tsv',sep='\t').set_index('cell_name',drop=False)
    header=pd.read_csv(ROOT/'data/raw/ipsc_log_normalised_counts.csv',nrows=0).columns
    frame=pd.read_csv(ROOT/'data/raw/ipsc_log_normalised_counts.csv',index_col=0,
                      dtype={c:np.float32 for c in header[1:]})
    a=ad.AnnData(frame.T.astype(np.float32));a.obs=meta.loc[a.obs_names].copy()
    a.var['original_gene_id']=a.var_names.astype(str)
    a.var['gene_name']=[s.split('_',1)[1] if '_' in s else s for s in a.var_names]
    a.var_names=a.var.gene_name.to_numpy();a.var_names_make_unique()
    a.obs['Group']=a.obs.day.str.replace('day','',regex=False).astype(int)
    path.parent.mkdir(parents=True,exist_ok=True)
    a.write_h5ad(path,compression='gzip')
    return a

def save(root,ds,tr,va,te,info):
    dest=ROOT/'data/research'/root/ds;dest.mkdir(parents=True,exist_ok=True)
    assert tr.var_names.equals(va.var_names) and tr.var_names.equals(te.var_names)
    assert set(tr.obs_names).isdisjoint(va.obs_names)
    if not info.get('intentional_source_overlap',False):
        assert set(tr.obs_names).isdisjoint(te.obs_names),'Unexpected train/test cell overlap'
    for name,x in [('train',tr),('val',va),('test',te)]:
        x.uns.clear()
        for k in list(x.layers):del x.layers[k]
        x.write_h5ad(dest/f'{name}.h5ad',compression='gzip')
        info[name]={'shape':list(x.shape),'unique_cells':len(set(x.obs_names)),
                    'groups':{str(k):int(v) for k,v in x.obs.Group.value_counts().items()}}
    info['genes']=list(tr.var_names);info['train_test_cell_overlap']=len(set(tr.obs_names)&set(te.obs_names))
    info['data_root']=str(dest.parent.relative_to(ROOT));write_json(info,dest/'manifest.json')
    record(f'研究数据 {root}/{ds} 准备完成：train={tr.shape}, test={te.shape}，训练/测试重合细胞ID={info["train_test_cell_overlap"]}。')

def make_ipsc(a):
    # Verify all retained v1 values directly against the original public matrix.
    errors={}
    for part in ['train','val','test']:
        old=ad.read_h5ad(ROOT/f'data/processed/ipsc/{part}.h5ad')
        direct=a[old.obs_names,old.var_names].X
        err=float(np.max(np.abs(direct-old.X)));assert err==0.
        errors[part]=err
    write_json({'raw_csv_to_processed_max_error':errors,'cells':a.n_obs,'genes':a.n_vars},ROOT/'results/research/ipsc_raw_alignment.json')
    oldtr=ad.read_h5ad(ROOT/'data/processed/ipsc/train.h5ad')
    oldva=ad.read_h5ad(ROOT/'data/processed/ipsc/val.h5ad')
    oldte=ad.read_h5ad(ROOT/'data/processed/ipsc/test.h5ad')
    genes=oldtr.var_names
    excluded=set(oldva.obs_names)|set(oldte.obs_names)
    full_train=a[a.obs.Group.isin([0,3])&~a.obs_names.isin(excluded),genes].copy()
    save('full_endpoints','ipsc',full_train,oldva.copy(),oldte.copy(),
         {'question':'Larger endpoint-only training pool; fixed v1 genes and evaluation cells',
          'scope':'All remaining day0/day3 cells; no day1/day2 model training',
          'features':'Exactly frozen v1 203-gene table','donor_holdout':False})
    rng=np.random.default_rng(42);allchosen=[];discovery_ids=[]
    for d in range(4):
        chosen=rng.choice(a.obs_names[a.obs.Group==d].to_numpy(),2000,replace=False)
        allchosen.extend(chosen)
        discovery_ids.extend(chosen[:1200])
    # Positive control uses the previously excluded middle-day discovery cells,
    # keeping the v1 feature table and evaluation cells exactly fixed.
    discovered=a[discovery_ids].copy()
    mid=discovered[discovered.obs.Group.isin([1,2]),genes].copy()
    tr=ad.concat([oldtr,mid],join='inner',merge='first')
    save('all_days','ipsc',tr,oldva.copy(),oldte.copy(),
         {'seed':42,'question':'Capacity positive control; intermediate training cells deliberately supplied',
          'scope':'Not endpoint-only prediction; same v1 genes and exact evaluation cells'})
    # Re-run the released preprocessing literally: replacement sampling, legacy
    # NumPy RNG and the raw metadata order. No original author seed was released.
    rng=np.random.RandomState(42);train_ids=[];val_ids=[];test_ids=[]
    for day in a.obs.day.unique():
        pool=np.flatnonzero(a.obs.day.to_numpy()==day)
        chosen=pool[rng.choice(len(pool),2000,replace=True)]
        order=np.arange(2000);rng.shuffle(order);chosen=chosen[order]
        train_ids.extend(chosen[:1200]);val_ids.extend(chosen[1200:1400]);test_ids.extend(chosen[1400:])
    disc=a[train_ids].copy();sc.pp.highly_variable_genes(disc,n_top_genes=200)
    mask=disc.var.highly_variable.to_numpy()|disc.var_names.isin(['NANOG','T','GATA6'])
    gg=disc.var_names[mask]
    tr=disc[disc.obs.Group.isin([0,3]),gg].copy();te=a[test_ids,gg].copy()
    # The source never uses validation for checkpoint selection. To monitor this
    # run, use additional endpoints excluded from both its train and test IDs.
    unused=a[~a.obs_names.isin(set(tr.obs_names)|set(te.obs_names))]
    ids=[]
    for g in [0,3]:ids.extend(unused.obs_names[unused.obs.Group==g][:200])
    va=a[ids,gg].copy()
    save('source','ipsc',tr,va,te,{'seed':42,'sampling':'released notebook legacy RNG + replacement',
        'intentional_source_overlap':True,
        'checkpoint':'fixed final 100000 steps; validation is monitoring only',
        'original_seed_unknown':True,'hvg_fit':'all four discovery-day partitions as in released notebook'})
    # Fresh strict endpoint-feature protocol: use cells not used anywhere in v1
    # discovery/train/validation/test. No middle-day expression enters feature fitting.
    oldused=set(allchosen);rng=np.random.default_rng(20261113)
    ids_tr=[];ids_va=[];ids_te=[]
    for g in range(4):
        pool=a.obs_names[(a.obs.Group==g)&(~a.obs_names.isin(oldused))].to_numpy()
        chosen=rng.choice(pool,2000,replace=False)
        if g in [0,3]:ids_tr.extend(chosen[:1200]);ids_va.extend(chosen[1200:1400])
        ids_te.extend(chosen[1400:])
    tr=a[ids_tr].copy();sc.pp.highly_variable_genes(tr,n_top_genes=200)
    gg=tr.var_names[tr.var.highly_variable.to_numpy()|tr.var_names.isin(['NANOG','T','GATA6'])]
    save('fresh_endpoints','ipsc',a[ids_tr,gg].copy(),a[ids_va,gg].copy(),a[ids_te,gg].copy(),
        {'seed':20261113,'hvg_fit':'endpoint training cells only','all_cells_disjoint_from_v1_8000':True,
         'note':'Fresh cell draw, not a donor-held-out test; no intermediate expression used for features'})

def make_norman():
    a=ad.read_h5ad(ROOT/'data/raw/norman/perturb_processed.h5ad')
    groups={'ctrl':0,'PTPN12+ctrl':1,'ZBTB25+ctrl':2,'PTPN12+ZBTB25':3}
    a=a[a.obs.condition.isin(groups)].copy();a.obs['Group']=a.obs.condition.astype(str).map(groups).astype(int)
    markers=['ALAS2','HBA1','HBA2','HIST1H1C','GYPB','SLC25A37','LGALS1']
    oldtr=ad.read_h5ad(ROOT/'data/processed/norman/train.h5ad')
    oldva=ad.read_h5ad(ROOT/'data/processed/norman/val.h5ad')
    oldte=ad.read_h5ad(ROOT/'data/processed/norman/test.h5ad')
    for n in [200,1000]:
        sc.pp.highly_variable_genes(a,n_top_genes=n)
        gg=a.var_names[a.var.highly_variable.to_numpy()|a.var.gene_name.isin(markers)]
        # Source-like feature selection, with target condition included for HVGs.
        # Preserve v1 training/validation cells to isolate the feature-table effect.
        root='source_features' if n==200 else 'source_1000'
        tr=a[oldtr.obs_names,gg].copy();va=a[oldva.obs_names,gg].copy();te=a[oldte.obs_names,gg].copy()
        for x in [tr,va,te]:x.X=x.X.toarray() if hasattr(x.X,'toarray') else x.X
        save(root,'norman',tr,va,te,{'hvg_n_top':n,'hvg_fit':'all four conditions per released prep notebook',
            'scope':'transductive feature selection; same v1 cells isolates feature table',
            'note':'n_top=200 matches saved 203-dimensional model; current source n_top=1000 gives 1000 dimensions'})
        # Both input-size variants use the same original raw cell order.
        # Source training uses all 7890 cells with training-only monitoring.
        full_root='source_full' if n==200 else 'source_1000_full'
        save_full_norman(full_root,a,gg,va,te,n)
    # Positive control: expose part of the combination during training and evaluate
    # a disjoint part, isolating reconstruction from zero-shot interaction inference.
    gg=oldtr.var_names;rng=np.random.default_rng(42);ids=rng.permutation(oldte.n_obs)
    n=int(.8*len(ids));combo=oldte[ids[:n]].copy()
    tr=ad.concat([oldtr,combo],join='inner',merge='first');te=oldte[ids[n:]].copy()
    save('combo_seen','norman',tr,oldva.copy(),te,{'seed':42,'scope':'Positive control; AB no longer unseen',
        'question':'Can the same model represent/reconstruct the combination after examples are supplied?'})

def save_full_norman(root,a,genes,val,test,n_top):
    trfull=a[a.obs.Group.isin([0,1,2]),genes].copy()
    trfull.X=trfull.X.toarray() if hasattr(trfull.X,'toarray') else trfull.X
    dest=ROOT/'data/research'/root/'norman';dest.mkdir(parents=True,exist_ok=True)
    trfull.uns.clear();trfull.layers.clear()
    trfull.write_h5ad(dest/'train.h5ad',compression='gzip')
    val.write_h5ad(dest/'val.h5ad',compression='gzip');test.write_h5ad(dest/'test.h5ad',compression='gzip')
    write_json({'train_cells':trfull.n_obs,'test_cells':test.n_obs,'genes':list(genes),
                'hvg_fit':f'all four conditions; top{n_top} plus seven markers',
                'cell_order':'Original raw AnnData order in both 203/1000-gene variants',
                'validation':'training monitor only; no independent validation in source'},dest/'manifest.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['ipsc','norman','both'],default='both');args=p.parse_args()
    if args.dataset in ['norman','both']:make_norman()
    if args.dataset in ['ipsc','both']:make_ipsc(full_ipsc())
