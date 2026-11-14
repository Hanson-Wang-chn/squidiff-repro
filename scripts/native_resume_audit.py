"""Verify exact checkpoint continuation with the released CPU loader and NumPy timestep sampler."""
import json
import anndata as ad
import torch
from torch.utils.data import DataLoader, TensorDataset
from common import ROOT, setup, seed_all, build_model, atomic_save, write_json
from train import rng_state, restore_rng
from Squidiff.resample import UniformSampler


def main():
    setup(); seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    a=ad.read_h5ad(ROOT/'data/research/source/ipsc/train.h5ad')
    dataset=TensorDataset(torch.from_numpy(a.X))
    def loader(): return DataLoader(dataset,batch_size=64,shuffle=True)
    dl=loader()
    m,d=build_model(a.n_vars);m.cuda().train();sampler=UniformSampler(d)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=0.)
    def step(model,optimizer,data,index):
        x=next(iter(data))[0].cuda();t,_=sampler.sample(len(x),'cuda')
        optimizer.zero_grad(set_to_none=True)
        loss=d.training_losses(model,x,t,model_kwargs={'drug_dose':None,'control_feature':None})['loss'].mean()
        loss.backward()
        for p in optimizer.param_groups:p['lr']=1e-4*(1-max(0,index-1)/100000)
        optimizer.step()
        return float(loss)
    for i in range(7):step(m,opt,dl,i)
    path=ROOT/'logs/native_equivalence/native_loader_resume.pt'
    atomic_save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state()),path)
    expected=[step(m,opt,dl,i) for i in range(7,12)]
    restored,_=build_model(a.n_vars);restored.cuda().train()
    restored_opt=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=0.)
    restored_dl=loader()
    s=torch.load(path,map_location='cpu',weights_only=False)
    restored.load_state_dict(s['model']);restored_opt.load_state_dict(s['optimizer']);restore_rng(s['rng'])
    actual=[step(restored,restored_opt,restored_dl,i) for i in range(7,12)]
    error=max(float((m.state_dict()[k].float()-v.float()).abs().max()) for k,v in restored.state_dict().items())
    out=dict(uninterrupted_losses=expected,resumed_losses=actual,max_state_error=error,
             exact_match=expected==actual and error==0.,
             loader='Fresh shuffled CPU DataLoader iterator each step',timestep_sampler='Original NumPy UniformSampler',
             optimizer='Source non-fused AdamW; source LR timing; IEEE FP32')
    write_json(out,ROOT/'results/research/native_loader_resume_audit.json')
    path.unlink()
    print(json.dumps(out,indent=2))
    assert out['exact_match']


if __name__=='__main__':main()
