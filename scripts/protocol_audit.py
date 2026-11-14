"""Persist protocol facts and cell/feature provenance for the scientific audit."""
import hashlib
import json
from pathlib import Path
import anndata as ad
from common import ROOT, write_json


def main():
    records = []
    roots = ['data/processed', 'data/research/source',
             'data/research/source_full', 'data/research/all_days',
             'data/research/fresh_endpoints', 'data/research/source_1000_full',
             'data/research/full_endpoints']
    for root in roots:
        for ds in ['ipsc', 'norman']:
            path = ROOT / root / ds
            if not (path / 'train.h5ad').exists():
                continue
            parts = {k: ad.read_h5ad(path / f'{k}.h5ad') for k in ['train', 'val', 'test']}
            tr, va, te = [parts[k] for k in ['train', 'val', 'test']]
            assert tr.var_names.equals(va.var_names) and tr.var_names.equals(te.var_names)
            overlap = set(tr.obs_names) & set(te.obs_names)
            allowed = root == 'data/research/source' and ds == 'ipsc'
            assert not overlap or allowed, (root, ds, len(overlap))
            mid_overlap = set(tr.obs_names) & set(te.obs_names[te.obs.Group.isin([1, 2])]) if ds == 'ipsc' else set()
            assert not mid_overlap, (root, ds, 'intermediate test cells in training')
            row = dict(data_root=root, dataset=ds,
                       train_test_overlap=len(overlap), overlap_intentional=allowed,
                       train_validation_overlap=len(set(tr.obs_names) & set(va.obs_names)),
                       feature_order_identical=True, partitions={})
            for k, a in parts.items():
                row['partitions'][k] = dict(shape=list(a.shape), unique_cells=len(set(a.obs_names)),
                    groups={str(g): int(n) for g, n in a.obs.Group.value_counts().items()},
                    sha256=hashlib.sha256((path / f'{k}.h5ad').read_bytes()).hexdigest())
            records.append(row)
    notebook = ROOT / 'upstream/Squidiff_reproducibility/preprocess_fig2_data.ipynb'
    n = json.loads(notebook.read_text())
    execution = {str(i): dict(execution_count=n['cells'][i].get('execution_count'),
                             source=''.join(n['cells'][i]['source'])) for i in [6, 7, 8]}
    out = dict(data=records, ipsc_preprocessing_execution=execution,
        interpretation={
            'cluster_filter': 'Released notebook executes new_adata=adata after the cluster filter; the filter is overwritten before sampling.',
            'source_ipsc_overlap': 'Replacement sampling creates endpoint train/test duplicates. Reproduced explicitly only in source runs; intermediate test IDs remain absent from model fitting.',
            'source_norman_features': 'Released prep currently requests 1000 HVGs, whereas evaluation loads 203-input weights. Both 200+marker and 1000-gene variants are tested.',
            'r2_argument_order': 'sample_squidiff.sampler.cal_metric passes prediction first to r2_score. Results retain both notebook_reverse_r2 and conventional truth-reference r2.',
            'source_latent_sampling': 'iPSC endpoint curves encode individual real endpoint cells; middle curves use mean(z_day0)+fraction*delta+0.7*N(0,I). Matched-sampler negative controls test this distinction.',
            'source_norman_literal': 'A 60-dimensional control centroid is passed as z_sem_origin; interp_with_direction takes its scalar mean, yielding a different expression than the intended vector equation.',
            'checkpoint_limitation': 'The exact trained fig2/Norman checkpoints, processed feature lists and original preprocessing seed were not located in the accessible public artifacts; this is a public-protocol reimplementation, not an exact-weight rerun.'})
    write_json(out, ROOT / 'results/research/protocol_audit.json')
    print(f'Audited {len(records)} dataset variants; no unexpected train/test cell overlaps.')


if __name__ == '__main__':
    main()
