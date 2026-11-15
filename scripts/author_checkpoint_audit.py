"""Audit supplied Figshare assets and run the author's BVO checkpoint unchanged.

These are training-endpoint diagnostics, never held-out temporal prediction.
No author's asset is modified, extracted again, or deleted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
import torch

from common import ROOT, build_model, dense, seed_all, write_json
from evaluate import metrics

OUT = ROOT / 'results/round3/author'


def digest(path, kind='sha256'):
    h = hashlib.new(kind)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def inventory(folder):
    OUT.mkdir(parents=True, exist_ok=True)
    files = json.loads((ROOT/'docs/figshare_v3.json').read_text())['files']
    expected = {f['name']: f for f in files}
    records = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name in expected:
            md5 = digest(path, 'md5')
            assert md5 == expected[path.name]['computed_md5'], path
            records.append(dict(path=str(path), bytes=path.stat().st_size, md5=md5,
                                figshare_match=True))
    with zipfile.ZipFile(folder/'27948633.zip') as z:
        members = [dict(name=f.filename, bytes=f.file_size, crc32=f'{f.CRC:08x}')
                   for f in z.infolist()]
    assert not any('fig2_model' in f['name'] for f in members)
    weight = folder/'VO_diff_model.pt'
    state = torch.load(weight, map_location='cpu', weights_only=True)
    a = ad.read_h5ad(folder/'VO_trained_adata.h5ad')
    assert a.n_vars == state['input_layer.weight'].shape[1] == 596
    assert state['output_layer.weight'].shape[0] == 596
    # All BVO control endpoints are already in the author's training matrix.
    full = ad.read_h5ad(folder/'adata_norm.h5ad', backed='r')
    bvo = a.obs.day.to_numpy() == 1
    assert a.obs_names[bvo].isin(full.obs_names).all()
    direct = full[a.obs_names[bvo], a.var_names].X
    direct = direct.toarray() if hasattr(direct, 'toarray') else np.asarray(direct)
    error = float(np.max(np.abs(direct - dense(a)[bvo])))
    assert error < 1e-6, error
    control = full.obs[(full.obs.group == 'Control') & (full.obs.Major_type != 'HSC-like')]
    assert set(control.index) == set(a.obs_names[bvo])
    full.file.close()
    info = dict(files=records, archive_members=members,
        checkpoint_sha256=digest(weight), shape=list(a.shape), genes=a.var_names.tolist(),
        obs_counts=a.obs.Major_type.value_counts().to_dict(),
        bn_updates={k:int(v) for k,v in state.items() if k.endswith('num_batches_tracked')},
        control_alignment_max_error=error, all_non_hsc_control_cells_in_training=True,
        fig2_checkpoint_present=False,
        scope='BVO day -1 / day11 training endpoints; no real intermediate timepoints supplied')
    write_json(info, OUT/'inventory.json')
    print(json.dumps({k:v for k,v in info.items() if k not in ['genes','archive_members','files']}), flush=True)
    return a


@torch.inference_mode()
def evaluate(folder, cells, steps, repeats):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(42)
    a = ad.read_h5ad(folder/'VO_trained_adata.h5ad')
    x = dense(a)
    g = a.obs.Major_type.to_numpy().astype(int)
    m, d = build_model(a.n_vars, steps)
    m.load_state_dict(torch.load(folder/'VO_diff_model.pt',map_location='cpu',weights_only=True),strict=True)
    m.cuda().eval()
    z = m.encoder(torch.tensor(x, device='cuda'))
    groups = {i: z[torch.tensor(g == i, device='cuda')] for i in range(4)}
    means = {i: v.mean(0) for i,v in groups.items()}
    dest = OUT/f'eval_{steps}'
    dest.mkdir(exist_ok=True)
    rows, gene_rows = [], []

    def draw(zz, noise):
        result=[]
        for start in range(0, len(zz), 400):
            zz1 = zz[start:start+400]
            result.append(d.ddim_sample_loop(m,(len(zz1),a.n_vars),
                noise=noise[start:start+len(zz1)],model_kwargs={'z_mod':zz1},
                clip_denoised=False).cpu().numpy())
        return np.concatenate(result)

    for rep in range(repeats):
        seed_all(1000+rep)
        # Same x_T and source sample for every mode; no best-of sampling.
        ids = {i: torch.randperm(len(v),device='cuda')[:cells] for i,v in groups.items()}
        noise = torch.randn(cells,a.n_vars,device='cuda')
        standard = torch.randn(cells,60,device='cuda')
        for group in range(4):
            source = groups[group][ids[group]]
            truth = x[g == group][ids[group].cpu().numpy()]
            targets = {
                'encoded': source,
                'center': means[group].expand(cells,-1),
                'gaussian_0.2': means[group]+.2*standard,
                # Encode a different class: preserve valid latent magnitudes.
                'wrong_class': groups[(group+1)%4][ids[(group+1)%4]],
            }
            for mode,zz in targets.items():
                pred=draw(zz,noise)
                row=dict(group=group,mode=mode,repeat=rep,sampling_steps=steps,
                    scope='training endpoints, conditional generation', **metrics(truth,pred))
                # Only encoded pairs have a genuine input/output cell pairing.
                if mode == 'encoded': row['paired_cell_mse']=float(np.mean((truth-pred)**2))
                rows.append(row)
                np.savez_compressed(dest/f'{mode}_group{group}_rep{rep}.npz',
                    prediction=pred, truth_ids=a.obs_names[g==group][ids[group].cpu().numpy()].to_numpy(dtype=str),
                    genes=a.var_names.to_numpy(dtype=str))
                for j,name in enumerate(a.var_names):
                    gene_rows.append(dict(group=group,mode=mode,repeat=rep,gene=name,
                        observed=float(truth[:,j].mean()),predicted=float(pred[:,j].mean())))
                pd.DataFrame(rows).to_csv(dest/'metrics.csv', index=False)
                print(json.dumps(row),flush=True)
    pd.DataFrame(gene_rows).to_csv(dest/'gene_means.csv',index=False)
    # BVO interpolation has no corresponding measured intermediate truth.
    # Keep only descriptive curves, including the alpha=1 endpoint as a control.
    curves=[]
    for group in [1,2,3]:
        for alpha in [0., 1/3, 2/3, 1., 1.5]:
            seed_all(7000)
            noise=torch.randn(cells,a.n_vars,device='cuda')
            zz=means[0]+alpha*(means[group]-means[0])+.2*torch.randn(cells,60,device='cuda')
            pred=draw(zz,noise)
            np.savez_compressed(dest/f'curve_group{group}_alpha{alpha:.4f}.npz',
                prediction=pred,genes=a.var_names.to_numpy(dtype=str))
            for marker in ['NANOG','T','GATA6','PECAM1','CLDN5','PDGFRB','ACTA2','COL1A1','LUM']:
                if marker in a.var_names:
                    j=a.var_names.get_loc(marker)
                    curves.append(dict(group=group,alpha=alpha,gene=marker,mean=float(pred[:,j].mean()),
                        scope='unvalidated interpolation; real intermediate cells unavailable'))
    pd.DataFrame(curves).to_csv(dest/'descriptive_curves.csv',index=False)
    write_json(dict(cells=cells,steps=steps,repeats=repeats,rows=len(rows),
        checkpoint_sha256=digest(folder/'VO_diff_model.pt'),
        condition='original author epsilon model and original reverse equations',
        temporal_generalization_tested=False),dest/'complete.json')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['inventory','evaluate'])
    p.add_argument('--folder',type=Path,default=Path('/home/whs/squidiff-ckpt'))
    p.add_argument('--cells',type=int,default=300)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--repeats',type=int,default=1)
    args=p.parse_args()
    if args.action=='inventory': inventory(args.folder)
    else: evaluate(args.folder,args.cells,args.steps,args.repeats)
