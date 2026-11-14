"""Feature-selection sensitivity with frozen source train/test cell populations."""
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.metrics import r2_score
from common import ROOT, write_json


def main():
    a = ad.read_h5ad(ROOT / 'data/cache/ipsc_full.h5ad')
    source = ROOT / 'data/research/source/ipsc'
    tr = ad.read_h5ad(source / 'train.h5ad')
    te = ad.read_h5ad(source / 'test.h5ad')
    full_tr, full_te = a[tr.obs_names].X.astype(float), a[te.obs_names].X.astype(float)
    means = {d: full_tr[tr.obs.Group == d].mean(0) for d in [0, 3]}
    truth = {d: full_te[te.obs.Group == d].mean(0) for d in [1, 2]}
    ref = set(tr.var_names)
    v1 = set(ad.read_h5ad(ROOT / 'data/processed/ipsc/train.h5ad').var_names)
    rows, features = [], {}
    for seed in range(42, 52):
        rng = np.random.RandomState(seed)
        train_ids = []
        for day in a.obs.day.unique():
            pool = np.flatnonzero(a.obs.day.to_numpy() == day)
            chosen = pool[rng.choice(len(pool), 2000, replace=True)]
            order = np.arange(2000)
            rng.shuffle(order)
            train_ids.extend(chosen[order][:1200])
        disc = a[train_ids].copy()
        sc.pp.highly_variable_genes(disc, n_top_genes=200)
        mask = disc.var.highly_variable.to_numpy() | disc.var_names.isin(['NANOG', 'T', 'GATA6'])
        genes = disc.var_names[mask]
        features[str(seed)] = list(genes)
        if seed == 42:
            assert set(genes) == ref
        for day in [1, 2]:
            baseline = (1 - day / 3) * means[0][mask] + day / 3 * means[3][mask]
            observed = truth[day][mask]
            error = (baseline - observed) ** 2
            j = list(genes).index('T')
            rows.append(dict(hvg_sampling_seed=seed, target=f'day{day}', genes=len(genes),
                             overlap_source42=len(set(genes) & ref), overlap_v1=len(set(genes) & v1),
                             baseline_r2=r2_score(observed, baseline), baseline_mse=float(error.mean()),
                             T_share_of_baseline_squared_error=float(error[j] / error.sum()),
                             expression_variance_across_genes=float(np.var(observed))))
    out = ROOT / 'results/research/hvg_sensitivity'
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'fixed_cells_variable_features.csv', index=False)
    write_json(dict(features=features, scope='No model training or test-based feature selection. Only feature-discovery cell draw changes; evaluation/training population means remain fixed.'), out / 'method.json')
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
