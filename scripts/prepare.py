"""Deterministic public-data processing, preserving original cell identifiers."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from common import ROOT, write_json, record, dense

def hvg(a, n, markers, symbol_col=None):
    sc.pp.highly_variable_genes(a, n_top_genes=n, flavor='seurat')
    symbols = a.var[symbol_col].astype(str) if symbol_col else a.var_names
    mask = a.var.highly_variable.to_numpy() | np.asarray(pd.Index(symbols).isin(markers))
    return a.var_names[mask]

def save_split(name, a, train_ids, val_ids, test_ids, genes, info):
    dest = ROOT / 'data/processed' / name
    dest.mkdir(parents=True, exist_ok=True)
    sets = [set(a.obs_names[i]) for i in [train_ids, val_ids, test_ids]]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for key, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        b = a[ids, genes].copy()
        b.X = dense(b)
        b.uns.clear()
        b.write_h5ad(dest / f'{key}.h5ad', compression='gzip')
        info[key] = {'shape': list(b.shape), 'groups': {str(k): int(v) for k,v in b.obs.Group.value_counts().items()}}
    info['genes'] = list(genes)
    info['disjoint_cell_ids'] = True
    write_json(info, dest / 'manifest.json')
    record(f'{name} 数据准备完成：' + json.dumps({k: info[k] for k in ['train','val','test']}, ensure_ascii=False))

def norman(seed):
    a = ad.read_h5ad(ROOT / 'data/raw/norman/perturb_processed.h5ad')
    groups = {'ctrl':0, 'PTPN12+ctrl':1, 'ZBTB25+ctrl':2, 'PTPN12+ZBTB25':3}
    a = a[a.obs.condition.isin(groups)].copy()
    a.obs['Group'] = a.obs.condition.astype(str).map(groups).astype(int)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for g in [0,1,2]:
        ids = rng.permutation(np.flatnonzero(a.obs.Group.to_numpy()==g))
        nv = max(1, int(len(ids)*.1))
        val.extend(ids[:nv]); train.extend(ids[nv:])
    test = np.flatnonzero(a.obs.Group.to_numpy()==3)
    markers = ['ALAS2','HBA1','HBA2','HIST1H1C','GYPB','SLC25A37','LGALS1']
    # 203-dimensional notebook model: 200 HVGs union curated marker genes.
    # Fit feature selection on training cells only to avoid target-condition leakage.
    genes = hvg(a[train].copy(), 200, markers, 'gene_name')
    save_split('norman', a, np.array(train), np.array(val), test, genes,
               {'seed':seed, 'source':'GEARS Dataverse 6154020 (GSE133344)',
                'hvg_fit':'training conditions only', 'hvg_n_top':200, 'markers':markers,
                'note':'Public prep notebook currently requests 1000 HVGs but saved model uses 203; this run uses 200 HVGs union its seven curated markers.'})

def ipsc(seed):
    meta = pd.read_csv(ROOT / 'data/raw/ipsc_cell_metadata_cols.tsv', sep='\t')
    meta = meta.set_index('cell_name', drop=False)
    rng = np.random.default_rng(seed)
    ids, splits = [], []
    for day in ['day0','day1','day2','day3']:
        available = meta.index[meta.day==day].to_numpy()
        chosen = rng.choice(available, 2000, replace=False)
        ids.extend(chosen)
        splits.extend(['train']*1200+['val']*200+['test']*600)
    # Read only the sampled cells from the 5.4 GB gene-by-cell CSV.
    chosen_set = set(ids)
    frame = pd.read_csv(ROOT / 'data/raw/ipsc_log_normalised_counts.csv', index_col=0,
                        usecols=lambda c: c.startswith('Unnamed:') or c in chosen_set, dtype=None)
    assert frame.shape[1] == 8000, frame.shape
    frame = frame.loc[:, ids]
    a = ad.AnnData(frame.T.astype(np.float32))
    a.obs = meta.loc[a.obs_names, ['cell_name','day']].copy()
    a.obs['Group'] = a.obs.day.str.replace('day','',regex=False).astype(int)
    a.obs['split'] = splits
    a.var_names = pd.Index([x.split('_', 1)[1] if '_' in x else x for x in a.var_names])
    a.var_names_make_unique()
    # Matches notebook gene discovery across its 4-day discovery partition;
    # days 1/2 discovery cells never enter model fitting, validation or evaluation.
    discovery = a.obs.split=='train'
    genes = hvg(a[discovery].copy(), 200, ['NANOG','T','GATA6'])
    train = np.flatnonzero(discovery & a.obs.Group.isin([0,3]))
    val = np.flatnonzero((a.obs.split=='val') & a.obs.Group.isin([0,3]))
    test = np.flatnonzero(a.obs.split=='test')
    save_split('ipsc', a, train, val, test, genes,
               {'seed':seed, 'source':'Zenodo 3625024; author Figshare mirror 27948633',
                'hvg_fit':'separate 4800-cell day0-3 discovery partition (author protocol)',
                'sampling':'2000 unique cells/day without replacement; fixes notebook duplicate-cell leakage',
                'model_training_days':[0,3], 'test_days':[0,1,2,3],
                'gene_discovery_intermediate_cells_excluded_from_model':2400})

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['norman','ipsc']);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    globals()[args.dataset](args.seed)
