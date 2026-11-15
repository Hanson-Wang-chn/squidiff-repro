"""Independent, read-only reanalysis for report.md (2026-11-15).

Reads frozen predictions, data and final weights. Never trains a model or changes
an existing result. New outputs go only to results/diagnosis_20261115.
CPU dependencies: numpy, pandas, scipy, anndata, sklearn, torch, matplotlib;
--hvg additionally needs scanpy. No import of common.py or its logging side effects.
"""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/diagnosis_20261115'
warnings.filterwarnings('ignore', message='Observation names are not unique')


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def read_data(root):
    return tuple(ad.read_h5ad(ROOT / root / 'ipsc' / f'{p}.h5ad')
                 for p in ['train', 'test'])


def summary_metrics(y, p):
    return dict(r2=float(r2_score(y, p)), reverse_r2=float(r2_score(p, y)),
                pearson=float(np.corrcoef(y, p)[0, 1]),
                mse=float(np.mean((y - p) ** 2)))


def predictions():
    """Recompute mean metrics, error concentration and thresholded T detection."""
    rows, genes, integrity, baseline_rows = [], [], [], []
    data = {}
    paths = sorted((ROOT / 'results/ipsc').glob('*/research_eval/*/metrics.csv'))
    for path in paths:
        if not (path.parent / 'complete.json').exists():
            continue
        config = json.loads((path.parents[2] / 'config.json').read_text())
        root = config.get('data_root', 'data/processed')
        if root not in data:
            data[root] = read_data(root)
        tr, te = data[root]
        means = {d: tr.X[tr.obs.Group == d].astype(float).mean(0) for d in [0, 3]}
        env = np.maximum(means[0], means[3])
        j = tr.var_names.get_loc('T')
        for _, old in pd.read_csv(path).iterrows():
            if old.target not in ['day1', 'day2']:
                continue
            npz = path.parent / f'{old["mode"]}_{old.target}_rep{int(old["repeat"])}.npz'
            archive = np.load(npz, allow_pickle=False)
            assert np.array_equal(archive['genes'], tr.var_names.to_numpy())
            pred = archive['prediction'].astype(float)
            assert np.isfinite(pred).all()
            day = int(old.target[-1])
            truth = te.X[te.obs.Group == day].astype(float)
            y, p = truth.mean(0), pred.mean(0)
            base = (1 - day / 3) * means[0] + day / 3 * means[3]
            met = summary_metrics(y, p)
            errors = (y - p) ** 2
            transient = y - env >= .5
            keep = np.arange(len(y)) != j
            err = max(abs(met['r2'] - old.r2), abs(met['mse'] - old.mean_mse))
            assert err < 1e-10, (str(npz), err)
            integrity.append(dict(path=str(npz.relative_to(ROOT)), max_metric_error=err,
                                  sha256=hashlib.sha256(npz.read_bytes()).hexdigest()))
            row = dict(run=config['run'], tag=path.parent.name, step=int(old.step),
                       mode=old['mode'], day=day, data_root=root, **met,
                       baseline_r2=float(r2_score(y, base)),
                       mse_gain=1 - met['mse'] / np.mean((y - base) ** 2),
                       T_true=y[j], T_pred=p[j], T_error_share=errors[j] / errors.sum(),
                       transient_count=int(transient.sum()),
                       transient_error_share=errors[transient].sum() / errors.sum(),
                       r2_without_T=float(r2_score(y[keep], p[keep])),
                       baseline_r2_without_T=float(r2_score(y[keep], base[keep])),
                       T_true_p50=float(np.median(truth[:, j])),
                       T_pred_p50=float(np.median(pred[:, j])),
                       T_pred_p95=float(np.quantile(pred[:, j], .95)))
            for cutoff in [0, .1, 1, 3]:
                row[f'T_true_above_{cutoff}'] = float(np.mean(truth[:, j] > cutoff))
                row[f'T_pred_above_{cutoff}'] = float(np.mean(pred[:, j] > cutoff))
            rows.append(row)
            if config['run'] == 'source_fp32_s42' and path.parent.name == 'endpoint_validated_noise' and day == 1:
                for k, gene in enumerate(tr.var_names):
                    genes.append(dict(gene=gene, observed=y[k], predicted=p[k], baseline=base[k],
                                      endpoint_max=env[k], observed_excess=y[k] - env[k],
                                      predicted_excess=p[k] - env[k], squared_error=errors[k],
                                      error_share=errors[k] / errors.sum()))
    pd.DataFrame(rows).to_csv(OUT / 'recomputed_metrics.csv', index=False)
    pd.DataFrame(genes).sort_values('squared_error', ascending=False).to_csv(
        OUT / 'day1_error_genes.csv', index=False)
    write_json('prediction_integrity.json', dict(count=len(integrity),
               max_metric_error=max(r['max_metric_error'] for r in integrity), files=integrity))
    support = []
    for root, (tr, te) in data.items():
        j = tr.var_names.get_loc('T')
        for partition, a in [('train', tr), ('test', te)]:
            for day in sorted(a.obs.Group.unique()):
                v = a.X[a.obs.Group == day, j].astype(float)
                support.append(dict(data_root=root, partition=partition, day=int(day), cells=len(v),
                                    mean=float(v.mean()), median=float(np.median(v)),
                                    maximum=float(v.max()), above_zero=int(np.sum(v > 0)),
                                    above_3=int(np.sum(v > 3))))
        m0 = tr.X[tr.obs.Group == 0].astype(float).mean(0)
        m3 = tr.X[tr.obs.Group == 3].astype(float).mean(0)
        delta = m3 - m0
        for day in [1, 2]:
            y = te.X[te.obs.Group == day].astype(float).mean(0)
            # Evaluation-only oracle: the most favorable scalar clock on the line.
            alpha = float(np.dot(y - m0, delta) / np.dot(delta, delta))
            nominal = m0 + day / 3 * delta
            optimal = m0 + np.clip(alpha, 0, 1) * delta
            baseline_rows.append(dict(data_root=root, day=day, nominal_alpha=day/3,
                oracle_alpha=alpha, nominal_r2=float(r2_score(y, nominal)),
                oracle_segment_r2=float(r2_score(y, optimal)),
                fraction_nominal_sse_after_oracle_clock=float(np.sum((y-optimal)**2)/np.sum((y-nominal)**2)),
                truth_T=y[j], largest_T_on_endpoint_expression_segment=max(m0[j], m3[j]),
                oracle_uses_target=True))
    pd.DataFrame(support).to_csv(OUT / 'T_support.csv', index=False)
    pd.DataFrame(baseline_rows).to_csv(OUT / 'expression_clock_oracle.csv', index=False)


def latent():
    """CPU encoder geometry and an exact latent-scale reparameterization check."""
    import torch
    sys.path.insert(0, str(ROOT / 'upstream/Squidiff'))
    from Squidiff.MLPModel import MLPModel
    torch.set_num_threads(4)
    rows, spectra = [], []
    for run in ['source_fp32_s42', 'source_fp32_s43', 'source_fp32_s44',
                'all_days_fp32_s42', 'full_endpoints_fp32_s42']:
        config = json.loads((ROOT / 'results/ipsc' / run / 'config.json').read_text())
        tr, te = read_data(config['data_root'])
        model = MLPModel(tr.n_vars, tr.n_vars, num_layers=3, use_encoder=True).eval()
        path = ROOT / 'results/ipsc' / run / 'step_100000.pt'
        state = torch.load(path, weights_only=False, map_location='cpu')
        model.load_state_dict(state['model'])
        del state
        def encode(x):
            with torch.inference_mode():
                return torch.cat([model.encoder(part) for part in
                                  torch.tensor(x, dtype=torch.float32).split(512)]).numpy().astype(float)
        z, zt = encode(tr.X), encode(te.X)
        mu = {d: z[tr.obs.Group == d].mean(0) for d in [0, 3]}
        delta = mu[3] - mu[0]
        origin = zt[te.obs.Group == 0].mean(0)
        res = np.concatenate([z[tr.obs.Group == d] - mu[d] for d in [0, 3]])
        covariance = res.T @ res / (len(res) - 2)
        eigen = np.linalg.eigvalsh(covariance)
        rms = float(np.sqrt(np.trace(covariance) / len(delta)))
        for k, val in enumerate(eigen):
            spectra.append(dict(run=run, eigenvalue_rank=k, eigenvalue=val))
        for day in [1, 2]:
            zz = zt[te.obs.Group == day]
            actual = zz.mean(0)
            ideal_alpha = float(np.dot(actual - origin, delta) / np.dot(delta, delta))
            distance_line = float(np.linalg.norm(actual - origin - ideal_alpha * delta))
            distance_nominal = float(np.linalg.norm(actual - origin - day/3 * delta))
            rows.append(dict(run=run, day=day, model_parameters=sum(p.numel() for p in model.parameters()),
                endpoint_distance=float(np.linalg.norm(delta)), within_rms=rms,
                covariance_min_eigenvalue=float(eigen[0]), covariance_max_eigenvalue=float(eigen[-1]),
                covariance_condition_number=float(eigen[-1]/eigen[0]),
                covariance_effective_dimension=float(eigen.sum()**2/np.sum(eigen**2)),
                sigma07_over_rms=.7/rms, noise07_rms_radius=.7*np.sqrt(len(delta)),
                noise07_radius_over_day1_shift=.7*np.sqrt(len(delta))/(np.linalg.norm(delta)/3),
                target_centroid_oracle_alpha=ideal_alpha,
                target_centroid_distance_to_line=distance_line,
                target_centroid_distance_to_nominal=distance_nominal,
                target_centroid_distance_over_endpoint_distance=distance_nominal/np.linalg.norm(delta),
                target_latents_are_diagnostic_only=True))
        if run == 'source_fp32_s42':
            torch.manual_seed(20261115)
            xt = torch.randn(16, tr.n_vars)
            t = torch.arange(16) * 60
            x = torch.tensor(tr.X[:16], dtype=torch.float32)
            with torch.inference_mode():
                original_z = model.encoder(x)
                before = model(xt, t, z_mod=original_z).clone()
                # The only entrance from encoder z to the denoiser is these affine maps.
                scale = 2.
                model.encoder.fc3.weight.mul_(scale)
                model.encoder.fc3.bias.mul_(scale)
                for block in model.mlp_blocks:
                    block.zsem_dense.weight.div_(scale)
                scaled_z = model.encoder(x)
                after = model(xt, t, z_mod=scaled_z)
                write_json('latent_scale_invariance.json', dict(scale=scale,
                    max_encoder_scale_error=float((scaled_z-scale*original_z).abs().max()),
                    max_noise_prediction_error=float((before-after).abs().max()),
                    fixed_sigma_after_scaling_equivalent_original_sigma=.7/scale,
                    scope='Algebraic/one-forward-pass check; no retraining or full new DDIM trajectory'))
        print('Encoded', run, flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'latent_geometry.csv', index=False)
    pd.DataFrame(spectra).to_csv(OUT / 'latent_covariance_eigenvalues.csv', index=False)


def hvg():
    """Sensitivity to correctly declaring log2 for HVG only; expression unchanged."""
    import scanpy as sc
    a = ad.read_h5ad(ROOT / 'data/cache/ipsc_full.h5ad')
    tr, te = read_data('data/research/source')
    rng = np.random.RandomState(42)
    ids = []
    for day in a.obs.day.unique():
        pool = np.flatnonzero(a.obs.day.to_numpy() == day)
        chosen = pool[rng.choice(len(pool), 2000, replace=True)]
        order = np.arange(2000)
        rng.shuffle(order)
        ids.extend(chosen[order][:1200])
    discovery = a[ids].copy()
    refs, rows = {}, []
    for name, base, dtype in [('source_assumes_ln', None, np.float32),
                              ('source_float64', None, np.float64),
                              ('declare_log2', 2, np.float32)]:
        b = discovery.copy()
        b.X = b.X.astype(dtype)
        if base is not None:
            b.uns['log1p'] = {'base': base}
        sc.pp.highly_variable_genes(b, n_top_genes=200)
        keep = b.var.highly_variable.to_numpy() | b.var_names.isin(['NANOG', 'T', 'GATA6'])
        selected = b.var_names[keep]
        refs[name] = list(selected)
        if name == 'source_assumes_ln':
            assert list(selected) == list(tr.var_names), 'Source HVG recreation differs'
        for day in [1, 2]:
            m0 = np.asarray(a[tr.obs_names[tr.obs.Group == 0], selected].X, dtype=float).mean(0)
            m3 = np.asarray(a[tr.obs_names[tr.obs.Group == 3], selected].X, dtype=float).mean(0)
            y = np.asarray(a[te.obs_names[te.obs.Group == day], selected].X, dtype=float).mean(0)
            rows.append(dict(hvg_variant=name, day=day, genes=len(selected),
                        overlap_source=len(set(selected)&set(tr.var_names)),
                        baseline_r2=float(r2_score(y,(1-day/3)*m0+day/3*m3))))
    write_json('hvg_log_base_genes.json', dict(features=refs,
        scope='HVG-only sensitivity, no retraining or choice of best test gene panel; original X retained.'))
    pd.DataFrame(rows).to_csv(OUT / 'hvg_log_base.csv', index=False)


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = pd.read_csv(OUT / 'recomputed_metrics.csv')
    source = f[f.run.str.fullmatch(r'source_fp32_s(?:42|43|44)') & (f.step==100000)]
    curve = pd.read_csv(ROOT/'results/research/sampling_curve_comparison.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ax = axes[0,0]
    ax.plot(range(4), curve.groupby('day').observed_T.first(), 'o-', color='black', label='Observed')
    for col, label in [('mixed_sampler_T','Released mixed sampler'), ('matched_sampler_T','Matched sampler')]:
        p = curve.groupby('day')[col]
        ax.plot(range(4),p.mean(),'o-',label=label)
        ax.fill_between(range(4),p.min(),p.max(),alpha=.15)
    p=source[source.tag=='endpoint_validated_noise'].groupby('day').T_pred.mean()
    ax.plot([1,2],p,'s--',label='Endpoint-calibrated noise')
    ax.set(title='A  T amplitude',xlabel='Day',ylabel='Mean expression (public data scale)',xticks=range(4))
    ax.legend(fontsize=8)
    ax=axes[0,1]
    p=f[(f.run=='source_fp32_s42')&(f.step==100000)&(f.day==1)]
    keys=[('raw_controls','centroid_sigma0.7'),('endpoint_validated_noise',None),('raw_controls','oracle')]
    labels=['Fixed sigma 0.7','Calibrated sigma','Encode true day1\n(diagnostic)']
    vals=[]
    for tag,mode in keys:
        q=p[p.tag==tag]
        if mode is not None:q=q[q['mode']==mode]
        vals.append(q.iloc[0])
    x=np.arange(3)
    ax.bar(x-.18,[v['T_pred_above_0'] for v in vals],.36,label='Expression > 0')
    ax.bar(x+.18,[v['T_pred_above_3'] for v in vals],.36,label='Expression > 3')
    ax.axhline(vals[0]['T_true_above_3'],color='k',ls='--',label='Observed fraction > 3')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1.08),ylabel='Fraction of cells',title='B  Detection versus substantive expression')
    ax.legend(fontsize=8)
    ax=axes[1,0]
    g=pd.read_csv(OUT/'day1_error_genes.csv').head(12)
    ax.bar(g.gene,g.error_share*100)
    ax.tick_params(axis='x',rotation=55)
    ax.set(title='C  Day1 errors are concentrated (calibrated seed42)',ylabel='Share of squared mean error (%)')
    ax=axes[1,1]
    s=f[(f.step==100000)&(f.day==1)&f.run.isin(['all_days_fp32_s42'])&(f.tag=='raw_controls')]
    s=s[s['mode'].isin(['centroid_sigma0.7','target_centroid','oracle'])].set_index('mode')
    modes=['centroid_sigma0.7','target_centroid','oracle']
    ax.bar(range(3),s.loc[modes].T_pred,color=['C0','C2','C2'])
    ax.axhline(s.T_true.iloc[0],color='k',ls='--',label='Observed')
    ax.set(xticks=range(3),xticklabels=['Endpoint line','True day1 center\n(diagnostic)','True day1 cells\n(diagnostic)'],
           ylabel='Mean T expression',title='D  All-days-trained model: path versus capacity')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/'diagnosis.png',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--latent',action='store_true')
    parser.add_argument('--hvg',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    predictions()
    if args.latent:latent()
    if args.hvg:hvg()
    plots()
    write_json('method.json', dict(date='2026-11-15', args=vars(args),
        scope='Independent CPU reanalysis of frozen data/predictions; optional endpoint encoder geometry. No training.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        warning='All test-based diagnostics are explanatory, not newly validated forecasting improvements.'))
    print('Written',OUT)


if __name__=='__main__':
    main()
