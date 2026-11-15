"""Small executable probes of released notebook interfaces; no training needed."""
import json
from pathlib import Path
import anndata as ad
from common import ROOT,write_json


def cells(name):
    n=json.loads((ROOT/'upstream/Squidiff_reproducibility'/name).read_text())
    return {i:''.join(c.get('source',[])) for i,c in enumerate(n['cells'])}


def main():
    b=cells('fig4_VO_reproducibility.ipynb')
    assert "data = sc.read_h5ad" in b[54]
    assert "data[0].X" in b[56] and "shape = (6838," in b[57]
    a=ad.read_h5ad('/home/whs/squidiff-ckpt/VO_trained_adata.h5ad')
    first=a[0]
    assert first.shape==(1,596)
    assert len(first[first.obs.Major_type==0])==0
    prep=cells('preprocess_fig2_data.ipynb')
    next_=cells('prep_diff_data.ipynb')
    assert "val_adata.write('val_adata_raw.h5ad')" in prep[8]
    assert "'diff_val_raw.h5ad'" in next_[1]
    assert "test_adata.write('test_adata_raw.h5ad')" in prep[8]
    assert "'diff_test_raw.h5ad'" in next_[1]
    fig2=cells('squidiff_on_differentiation.ipynb')
    assert 'fig2_model.pt' in fig2[1] and 'for i in range(6)' in fig2[8]
    write_json(dict(
        bvo_interface=dict(load_cell=54,indexing_cell=56,sample_cell=57,
            complete_shape=list(a.shape),literal_data0_shape=list(first.shape),
            first_cell_major_type=int(first.obs.Major_type.iloc[0]),
            literal_data0_ipsc_cells=0,
            consequence='Encoder receives one cell; semantic vector broadcasts during 6838-cell generation; later iPSC subset is empty',
            repair='Use whole AnnData consistently, as author_checkpoint_audit.py does'),
        fig2_filename_mismatch=dict(written=['diff_train_raw.h5ad','val_adata_raw.h5ad','test_adata_raw.h5ad'],
            next_notebook_reads=['datasets/diff_train_raw.h5ad','datasets/diff_val_raw.h5ad','datasets/diff_test_raw.h5ad'],
            repair='Local data preparation uses explicit stable paths and validates cell/gene identity'),
        fig2_replication=dict(model_load_cell=1,repeat_cell=8,sampling_repeats=6,
            model_retraining_inside_repeat=False,
            consequence='Six draws from one loaded model are not six independent model-training runs'),
        fig2_checkpoint_path='../Squidiff_project/saved_model/fig2_model.pt'),
        ROOT/'results/round3/notebook_audit.json')
    print('NOTEBOOK_INTERFACE_AUDIT_OK')


if __name__=='__main__':main()
