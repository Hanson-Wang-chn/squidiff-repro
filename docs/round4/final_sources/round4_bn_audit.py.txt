"""Endpoint-only BatchNorm audit. No intermediate RNA or target normalization."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import ROOT, build_model, write_json


@torch.no_grad()
def recalibrate(model, x):
    """Exact training-pool moments, sequentially, under fixed inference weights."""
    model.eval()
    e = model.encoder
    h = e.fc1(x)
    e.bn1.running_mean.copy_(h.mean(0))
    e.bn1.running_var.copy_(h.var(0, unbiased=True))
    h = e.fc2(torch.relu(e.bn1(h)))
    e.bn2.running_mean.copy_(h.mean(0))
    e.bn2.running_var.copy_(h.var(0, unbiased=True))
    return model


@torch.no_grad()
def main():
    import round2 as r
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    folder = ROOT/'results/round4/bn_audit'
    folder.mkdir(parents=True, exist_ok=True)
    tr, va, _ = r.load_data(dict(panel='source'))
    x = torch.tensor(np.asarray(tr.X), device='cuda')
    native = ROOT/'results/round2/native_s42'
    state = torch.load(native/'latest.pt', map_location='cpu', weights_only=False)
    cfg = json.loads((native/'config.json').read_text())
    rows = []
    geometry = []
    for label, weights in [('raw',state['model']),('ema',state['ema'])]:
        m, d = build_model(tr.n_vars)
        m.load_state_dict(weights)
        m.cuda().eval()
        before = m.encoder(x)
        rms = before.std(0).square().mean().sqrt()
        # Same fixed cell appears with mixed-day or single-day companion cells.
        saved = copy.deepcopy(m.state_dict())
        e = m.encoder
        e.bn1.train(); e.bn2.train()
        ids0=np.flatnonzero(tr.obs.Group.to_numpy()==0)
        ids3=np.flatnonzero(tr.obs.Group.to_numpy()==3)
        assert len(ids0)==len(ids3)
        order=np.stack([ids0,ids3],axis=1).reshape(-1)
        zchunks=torch.empty_like(before)
        zchunks[order]=torch.cat([e(chunk) for chunk in x[order].split(64)])
        m.load_state_dict(saved); m.eval()
        geometry.append(dict(weights=label,condition='mixed_train_batches',
            latent_rmse_over_eval_rms=float((zchunks-before).square().mean().sqrt()/rms)))
        allidx = []
        outputs = []
        for group in [0,3]:
            ids = np.flatnonzero(tr.obs.Group.to_numpy()==group)
            e.bn1.train();e.bn2.train()
            for indices in np.array_split(ids, max(1,len(ids)//64)):
                outputs.append(e(x[indices]));allidx.extend(indices)
        homogeneous = torch.empty_like(before)
        homogeneous[allidx] = torch.cat(outputs)
        m.load_state_dict(saved);m.eval()
        geometry.append(dict(weights=label,condition='single_day_batches',
            latent_rmse_over_eval_rms=float((homogeneous-before).square().mean().sqrt()/rms)))
        for mode in ['original_buffers','training_recalibrated']:
            if mode == 'training_recalibrated':recalibrate(m,x)
            z = m.encoder(x)
            row = dict(weights=label,buffers=mode,
                endpoint_validation_loss=r.validation(m,d,va,cfg),
                latent_change_over_raw_rms=float((z-before).square().mean().sqrt()/rms),
                bn1_mean_change=float((m.encoder.bn1.running_mean-saved['encoder.bn1.running_mean'].cuda()).square().mean().sqrt()),
                bn2_mean_change=float((m.encoder.bn2.running_mean-saved['encoder.bn2.running_mean'].cuda()).square().mean().sqrt()))
            rows.append(row);print(row,flush=True)
        del m, d
    pd.DataFrame(rows).to_csv(folder/'endpoint_losses.csv',index=False)
    pd.DataFrame(geometry).to_csv(folder/'batch_dependence.csv',index=False)
    write_json(dict(rows=rows,geometry=geometry,scope='All fitting and auditing uses endpoint RNA only; train-batch mode is a diagnostic, not a prediction protocol'),folder/'complete.json')

if __name__=='__main__':main()
