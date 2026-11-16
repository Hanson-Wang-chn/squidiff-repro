"""Local execution helpers; network and diffusion are the author's implementation."""
import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream/Squidiff'))
from Squidiff.script_util import model_and_diffusion_defaults, create_model_and_diffusion
from Squidiff.MLPModel import MLPModel

class EpsilonResidualMLPModel(MLPModel):
    """Explicit training-stability extension; source architecture stays the default."""
    def forward(self,x,timesteps=None,**kwargs):
        return x+super().forward(x,timesteps,**kwargs)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for these experiments')

def build_model(genes, sampling_steps=None, model_variant='source'):
    cfg = model_and_diffusion_defaults()
    cfg.update(gene_size=genes, output_dim=genes, num_layers=3, use_encoder=True)
    if sampling_steps:
        cfg['timestep_respacing'] = f'ddim{sampling_steps}'
    model,diffusion=create_model_and_diffusion(**cfg)
    if model_variant=='epsilon_residual':
        model.__class__=EpsilonResidualMLPModel
        with torch.no_grad():
            model.output_layer.weight.zero_();model.output_layer.bias.zero_()
    elif model_variant!='source':raise ValueError(model_variant)
    return model,diffusion

def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)

def record(message):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    (ROOT/'logs').mkdir(exist_ok=True)
    with open(ROOT/'logs/events.jsonl','a') as f:
        f.write(json.dumps({'time':now,'message':message},ensure_ascii=False)+'\n')
    short=message
    if message.startswith('研究评估 ') and ' 完成：' in message:
        short=message.split(' 完成：')[0]+' 完成；详细指标见对应 research_eval 目录。'
    with open(ROOT / 'README.md', 'a') as f:
        f.write(f'\n- {now}：{short}\n')
    print(f'[{now}] {message}', flush=True)

def dense(a):
    return np.asarray(a.X.toarray() if hasattr(a.X, 'toarray') else a.X, dtype=np.float32)
