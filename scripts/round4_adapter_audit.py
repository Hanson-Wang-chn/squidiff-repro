"""Analytic tests for reverse-floor ablation and safe CLI parsing."""
import json
import numpy as np
import torch
from common import ROOT,build_model,write_json
from round4_evaluate import omit_step_floor
from train_source_safe import parse

def main():
    torch.set_num_threads(2)
    m,d=build_model(4)
    _,dd=build_model(4);omit_step_floor(dd)
    x=torch.randn(3,4);t=torch.tensor([25,500,950])
    class Known(torch.nn.Module):
        def __init__(self,y):super().__init__();self.y=y
        def forward(self,x,t,**kwargs):
            aa=torch.tensor(d.alphas_cumprod,dtype=x.dtype)[t.long(),None]
            return (x-aa.sqrt()*self.y)/(1-aa).sqrt()
    positive=torch.ones_like(x)*2
    a=d.p_mean_variance(Known(positive),x,t)
    b=dd.p_mean_variance(Known(positive),x,t)
    error=max(float((a[k]-b[k]).abs().max()) for k in a)
    assert error==0
    negative=torch.ones_like(x)*-2
    orig=d.p_mean_variance(Known(negative),x,t)['pred_xstart']
    unclip=dd.p_mean_variance(Known(negative),x,t)['pred_xstart']
    assert float(orig.abs().max())==0 and float((unclip-negative).abs().max())<1e-4
    base=['--data_path',str(ROOT/'data/round2/source/train.h5ad'),
        '--logger_path',str(ROOT/'logs/round4/cli_probe'),'--gene_size','203','--output_dim','203','--check-only']
    args,check=parse(base+['--use_drug_structure','False','--use_encoder','True','--class_cond','False'])
    assert check and args['use_drug_structure'] is False and args['use_encoder'] is True and args['class_cond'] is False
    rejected=[]
    for extra in [['--output_dim','202'],['--predict_xstart','True'],['--learn_sigma','True']]:
        try:parse(base+extra)
        except ValueError as e:rejected.append(str(e))
        else:raise AssertionError(extra)
    write_json(dict(nonnegative_x0_native_max_error=error,negative_x0_recovery_max_error=float((unclip-negative).abs().max()),
        false_string_parsed_correctly=True,unsupported_configurations_rejected=rejected),ROOT/'results/round4/adapter_audit.json')
    print('ADAPTER_AUDIT_OK',error,rejected)

if __name__=='__main__':main()
