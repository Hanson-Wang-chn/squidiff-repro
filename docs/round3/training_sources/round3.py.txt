"""Third-round experiments using the already audited native training controller.

Only the factory and output destination differ from round2. Training retains the
author's run_loop/run_step/forward_backward and default AdamW dispatch.
"""
import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import round2 as r
from common import ROOT, build_model, write_json
from Squidiff.script_util import model_and_diffusion_defaults, create_model_and_diffusion
from diffusion_adapter import correct_reverse_target, min_snr_weight

OUT = ROOT/'results/round3'
VARIANTS = {
    'xstart': dict(target='xstart',weighting='uniform',model_variant='source'),
    'minsnr': dict(target='epsilon',weighting='min_snr_gamma5',model_variant='source'),
    'epsres': dict(target='epsilon',weighting='uniform',model_variant='epsilon_residual'),
}


def create(config, steps=1000):
    if config['target'] == 'xstart':
        cfg = model_and_diffusion_defaults()
        cfg.update(gene_size=config['genes'],output_dim=config['genes'],num_layers=3,
            use_encoder=True,predict_xstart=True,timestep_respacing=f'ddim{steps}')
        m,d = create_model_and_diffusion(**cfg)
        correct_reverse_target(d)
    else:
        m,d = build_model(config['genes'],steps,model_variant=config['model_variant'])
    if config['weighting'] == 'min_snr_gamma5':
        base = d.training_losses
        def training_losses(model,x_start,t,model_kwargs=None,noise=None):
            terms=base(model,x_start,t,model_kwargs=model_kwargs,noise=noise)
            terms['loss']=terms['loss']*min_snr_weight(d,t,gamma=5.)
            return terms
        d.training_losses=training_losses
    return m,d


def configure():
    OUT.mkdir(parents=True,exist_ok=True)
    # Evaluation computes the same common-panel diagnostics as round2.
    audit=json.loads((ROOT/'results/round2/data_audit.json').read_text())
    write_json(audit,OUT/'data_audit.json')
    parts={name:ad.read_h5ad(ROOT/f'data/round2/source/{name}.h5ad')
           for name in ['train','val','test']}
    tr,va,te=(parts[name] for name in ['train','val','test'])
    assert tr.var_names.is_unique
    assert tr.var_names.equals(va.var_names) and tr.var_names.equals(te.var_names)
    assert set(tr.obs.Group)=={0,3} and set(va.obs.Group)=={0,3}
    assert set(tr.obs_names).isdisjoint(va.obs_names)
    assert set(tr.obs_names).isdisjoint(te.obs_names[te.obs.Group.isin([1,2])])
    write_json(dict(genes=tr.var_names.tolist(),rows={k:v.n_obs for k,v in parts.items()},
        hashes={name:hashlib.sha256((ROOT/f'data/round2/source/{name}.h5ad').read_bytes()).hexdigest()
                for name in parts},
        train_test_endpoint_overlap=len(set(tr.obs_names)&set(te.obs_names)),
        middle_day_training_overlap=0,validation_overlap=0,
        hvg_scope='Historical source panel discovered across all four days; diagnostic evaluation only'),
        OUT/'protocol.json')
    hashes={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
            for name in ['round3.py','diffusion_adapter.py','round2.py','common.py']}
    r.OUT=OUT
    r.VARIANTS={name:dict(panel='source',transform='identity',width=2048,latent_noise=0.,wd=0.,
        adapter_hashes=hashes,**values) for name,values in VARIANTS.items()}
    r.create=create


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['train','evaluate'])
    p.add_argument('--variant',choices=list(VARIANTS),required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--checkpoint',default='final.pt')
    p.add_argument('--sample-steps',type=int,default=1000)
    p.add_argument('--cells',type=int,default=600)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--full',action='store_true')
    p.add_argument('--signed',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--criterion',choices=['wasserstein','mse'],default='wasserstein')
    p.add_argument('--reuse-candidates',action='store_true')
    p.add_argument('--selected-only',action='store_true')
    args=p.parse_args()
    configure()
    if args.action=='train': r.train(args)
    else: r.evaluate(args)
