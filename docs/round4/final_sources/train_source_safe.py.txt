"""Repair bool('False') CLI parsing and validate the native expression task.

The default expression-only path needs no chemistry package. Native network,
TrainLoop, CPU shuffled loader, optimizer and diffusion remain unmodified.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'upstream/Squidiff'))
from Squidiff.script_util import model_and_diffusion_defaults,create_model_and_diffusion,args_to_dict
from Squidiff import logger,dist_util
from Squidiff.train_util import TrainLoop,plot_loss
from Squidiff.resample import create_named_schedule_sampler


def boolean(value):
    if isinstance(value,bool):return value
    if value.lower() in ['true','1','yes']:return True
    if value.lower() in ['false','0','no']:return False
    raise argparse.ArgumentTypeError('Expected true/false, 1/0, or yes/no')


def parse(argv=None):
    defaults=model_and_diffusion_defaults()
    defaults.update(data_path='',control_data_path='',schedule_sampler='uniform',lr=1e-4,
        weight_decay=0.,lr_anneal_steps=100000,batch_size=64,microbatch=-1,
        ema_rate='0.9999',log_interval=10000,save_interval=10000,resume_checkpoint='',
        use_fp16=False,fp16_scale_growth=1e-3,gene_size=100,output_dim=100,
        num_layers=3,class_cond=False,use_encoder=True,diffusion_steps=1000,
        logger_path='',use_drug_structure=False,comb_num=1,use_ddim=True)
    parser=argparse.ArgumentParser(description=__doc__)
    for key,value in defaults.items():
        parser.add_argument('--'+key,default=value,type=boolean if isinstance(value,bool) else type(value))
    parser.add_argument('--check-only',action='store_true')
    args=vars(parser.parse_args(argv));check=args.pop('check_only')
    if not args['data_path'] or not args['logger_path']:raise ValueError('data_path and logger_path are required')
    if args['gene_size']!=args['output_dim']:raise ValueError('gene_size must equal output_dim for epsilon prediction')
    if args['learn_sigma']:raise ValueError('Pinned native diffusion training does not implement learned variance')
    if args['predict_xstart']:raise ValueError('Pinned reverse sampler assumes epsilon; use the matched x0 adapter in round3.py')
    if args['use_fp16']:raise ValueError('Pinned MLP FP16 conversion is incomplete; use FP32')
    if args['use_drug_structure'] and not args['control_data_path']:raise ValueError('Drug conditioning requires control_data_path')
    if args['lr_anneal_steps']<=0:raise ValueError('Specify a positive number of optimization updates')
    if not check and not args['resume_checkpoint']:raise ValueError('Specify resume_checkpoint as the native output directory; the pinned entry does not automatically resume')
    import anndata as ad
    a=ad.read_h5ad(args['data_path'],backed='r')
    try:
        if a.n_vars!=args['gene_size']:raise ValueError('Dataset gene count differs from the model')
        if 'Group' not in a.obs:raise ValueError('Dataset must contain obs.Group')
    finally:a.file.close()
    return args,check


def run(args):
    import anndata as ad
    from torch.utils.data import DataLoader
    from round2 import Cells
    logger.configure(dir=args['logger_path'])
    m,d=create_model_and_diffusion(**args_to_dict(args,model_and_diffusion_defaults().keys()))
    m.to(dist_util.dev())
    if args['use_drug_structure']:
        from Squidiff.scrna_datasets import prepared_data
        data=prepared_data(data_dir=args['data_path'],control_data_dir=args['control_data_path'],
            batch_size=args['batch_size'],use_drug_structure=True,comb_num=args['comb_num'])
    else:
        from common import dense
        expression=ad.read_h5ad(args['data_path']);expression.X=dense(expression)
        data=DataLoader(Cells(expression,{'transform':'identity'}),
            batch_size=args['batch_size'],shuffle=True)
    loop=TrainLoop(model=m,diffusion=d,data=data,batch_size=args['batch_size'],microbatch=args['microbatch'],
        lr=args['lr'],ema_rate=args['ema_rate'],log_interval=args['log_interval'],save_interval=args['save_interval'],
        resume_checkpoint=args['resume_checkpoint'],use_fp16=False,fp16_scale_growth=args['fp16_scale_growth'],
        schedule_sampler=create_named_schedule_sampler(args['schedule_sampler'],d),weight_decay=args['weight_decay'],
        lr_anneal_steps=args['lr_anneal_steps'],use_drug_structure=args['use_drug_structure'],comb_num=args['comb_num'])
    loop.run_loop();plot_loss(loop.loss_list,args)


if __name__=='__main__':
    args,check=parse();print(json.dumps(args,indent=2),flush=True)
    if not check:run(args)
