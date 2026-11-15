"""Final artifact checks, including native update counts and frozen old archive."""
import hashlib
import json
from pathlib import Path
import anndata as ad
import numpy as np
import torch
import round2 as r

torch.set_num_threads(2)
checks=[]
for variant in r.VARIANTS:
    dest=r.OUT/f'{variant}_s42'
    status=json.loads((dest/'status.json').read_text())
    cfg=json.loads((dest/'config.json').read_text())
    assert status['status']=='trained' and status['step']==100000,(variant,status)
    checkpoint=torch.load(dest/'final.pt',map_location='cpu',weights_only=False)
    assert checkpoint['step']==100000
    assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
    bn={k:int(v) for k,v in checkpoint['model'].items() if k.endswith('num_batches_tracked')}
    assert set(bn.values())=={100000},(variant,bn)
    train_path=r.ROOT/'data/round2'/cfg['panel']/'train.h5ad'
    assert hashlib.sha256(train_path.read_bytes()).hexdigest()==cfg['train_data_sha256']
    for name,sha in cfg['source_sha256'].items():
        assert hashlib.sha256((r.ROOT/'upstream/Squidiff/Squidiff'/name).read_bytes()).hexdigest()==sha
    checks.append(dict(variant=variant,step=checkpoint['step'],batch_norm_updates=bn,
        all_weights_finite=True,data_and_upstream_hashes_match=True,
        final_sha256=hashlib.sha256((dest/'final.pt').read_bytes()).hexdigest()))
    del checkpoint

raw=ad.read_h5ad(r.ROOT/'data/cache/ipsc_full.h5ad')
data=[]
for panel in ['source','log2hvg']:
    for part in ['train','val','test']:
        a=ad.read_h5ad(r.ROOT/'data/round2'/panel/f'{part}.h5ad')
        exact=np.asarray(raw[a.obs_names,a.var_names].X,dtype=np.float32)
        assert np.array_equal(exact,a.X),(panel,part)
        data.append(dict(panel=panel,partition=part,max_raw_cache_error=0.))
del raw

manifest=json.loads((r.ROOT/'docs/archive_manifest.json').read_text())
archive=[]
for item in manifest['files']:
    p=r.ROOT/item['path']
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual==item['sha256'],str(p)
    archive.append(str(item['path']))
r.write_json(dict(models=checks,data=data,old_archive_files_verified=len(archive),
    all_old_archive_hashes_unchanged=True),r.OUT/'final_checks.json')
print('FINAL_INTEGRITY_OK',len(checks),'models;',len(data),'data partitions;',len(archive),'old archive files')
