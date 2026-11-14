"""Cell bootstrap with fixed trained models and fixed training-mean baselines.

This is conditional sampling uncertainty, not biological replication or uncertainty
over training datasets. Repeated target cell IDs form one bootstrap cluster.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from common import ROOT, write_json


def main(args):
    np.random.seed(42)
    path = ROOT / 'results' / args.dataset / args.run / 'research_eval' / args.tag
    tr = ad.read_h5ad(ROOT / args.data_root / args.dataset / 'train.h5ad')
    te = ad.read_h5ad(ROOT / args.data_root / args.dataset / 'test.h5ad')
    g = tr.obs.Group.to_numpy()
    means = {k: tr.X[g == k].astype(np.float64).mean(0) for k in np.unique(g)}
    rows = []
    metrics = pd.read_csv(path / 'metrics.csv')
    for _, case in metrics.iterrows():
        if case['mode'] in ['oracle', 'target_centroid']:
            continue
        target = case.target
        selected = te if args.dataset == 'norman' else te[te.obs.Group == int(target[-1])]
        truth = selected.X.astype(np.float64)
        unique, inverse, counts = np.unique(selected.obs_names, return_inverse=True, return_counts=True)
        first = np.unique(inverse, return_index=True)[1]
        values = truth[first]
        pred = np.load(path / f'{case["mode"]}_{target}_rep{int(case["repeat"])}.npz')['prediction'].astype(np.float64)
        if args.dataset == 'norman':
            base = means[1] + means[2] - means[0]
        else:
            frac = int(target[-1]) / 3
            base = (1 - frac) * means[0] + frac * means[3]
        rng = np.random.default_rng(12345)
        weights = rng.multinomial(len(unique), np.ones(len(unique)) / len(unique), args.repeats) * counts[None, :]
        a = weights @ values / weights.sum(1, keepdims=True)
        weights_p = rng.multinomial(len(pred), np.ones(len(pred)) / len(pred), args.repeats)
        b = weights_p @ pred / len(pred)
        err = np.mean((a - b) ** 2, axis=1)
        linear = np.mean((a - base) ** 2, axis=1)
        stats = dict(relative_mse_gain=1 - err / linear,
                     r2=1 - err / np.var(a, axis=1))
        if args.dataset == 'norman':
            stats['interaction_r2'] = 1 - err / np.var(a - base, axis=1)
        else:
            j = tr.var_names.get_loc('T')
            stats['T_amplitude_ratio'] = b[:, j] / a[:, j]
        row = dict(dataset=args.dataset, run=args.run, mode=case['mode'], target=target,
                   repeat=int(case['repeat']), target_cells=len(truth), unique_target_cells=len(unique),
                   generated_cells=len(pred), bootstrap_repeats=args.repeats,
                   fraction_bootstrap_mse_gain_positive=float(np.mean(stats['relative_mse_gain'] > 0)))
        for name, values_ in stats.items():
            for q, value in zip(['p025', 'p50', 'p975'], np.quantile(values_, [.025, .5, .975])):
                row[f'{name}_{q}'] = float(value)
        rows.append(row)
    out = ROOT / 'results/research/conditional_uncertainty' / args.dataset / args.run
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f'{args.tag}.csv', index=False)
    write_json(dict(args=vars(args), assumptions=[
        'Training cells, model weights and exact training-mean baseline are fixed.',
        'Targets resampled by unique original cell ID to account for replacement duplicates.',
        'Generated cells are independently resampled; targets and generated cells are unpaired.',
        'Intervals describe cell sampling only, not donors, training-set uncertainty or biological replication.'
    ]), out / 'method.json')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', choices=['ipsc', 'norman'])
    p.add_argument('--run', required=True)
    p.add_argument('--data-root', default='data/processed')
    p.add_argument('--tag', default='raw_controls')
    p.add_argument('--repeats', type=int, default=1000)
    main(p.parse_args())
