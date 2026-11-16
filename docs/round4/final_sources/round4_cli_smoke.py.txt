"""Exercise the repaired CLI through two real native optimizer updates."""
import json
from unittest.mock import patch
import numpy as np
import torch
from common import ROOT,write_json
import train_source_safe as safe

def main():
    torch.set_num_threads(2)
    args,check=safe.parse(['--data_path',str(ROOT/'data/round2/source/train.h5ad'),
        '--logger_path',str(ROOT/'logs/round4/safe_cli_smoke'),
        '--resume_checkpoint',str(ROOT/'logs/round4/smoke_output_unused'),
        '--gene_size','203','--output_dim','203','--batch_size','8',
        '--lr_anneal_steps','2','--log_interval','1','--save_interval','10000',
        '--use_drug_structure','False','--use_encoder','True','--class_cond','False'])
    assert not check and args['use_drug_structure'] is False
    saves=[];losses=[]
    def observe_save(loop):saves.append(loop.step)
    def observe_plot(values,config):losses.extend(float(v.detach().cpu()) for v in values)
    # Exercise training normally; replace only disk-heavy checkpoint writing.
    with patch.object(safe.TrainLoop,'save',observe_save),patch.object(safe,'plot_loss',observe_plot):safe.run(args)
    assert len(losses)==2 and np.isfinite(losses).all() and saves==[0,2]
    write_json(dict(real_optimizer_updates=2,losses=losses,checkpoint_calls=saves,
        chemistry_dependency_required=False,drug_false_parsed_correctly=True,
        scope='Engineering smoke test only; no biological evaluation and no checkpoints saved'),ROOT/'results/round4/cli_smoke.json')
    print('SAFE_CLI_SMOKE_OK',losses,flush=True)

if __name__=='__main__':main()
