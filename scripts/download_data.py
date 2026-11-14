"""Resume public downloads and verify recorded MD5 digests (Figshare supplies iPSC MD5)."""
import hashlib
import json
import os
import zipfile
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[
    ('ipsc_log_normalised_counts.csv','https://ndownloader.figshare.com/files/59494394','98ab0c56756049b24179fef7f29a69a5'),
    ('ipsc_cell_metadata_cols.tsv','https://ndownloader.figshare.com/files/59494382','c2e1fd0b1b268047bdcbdbb2f6dab0cb'),
    ('norman.tar.gz','https://dataverse.harvard.edu/api/access/datafile/6154020','cdc41d6050e619c37fd9dd44d440e2b4'),
]

def md5(path):
    h=hashlib.md5()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    dest=ROOT/'data/raw';dest.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for name,url,digest in SOURCES:
        path=dest/name
        if not path.exists() or md5(path)!=digest:
            part=path.with_suffix(path.suffix+'.partial')
            if path.exists():os.replace(path,part)
            offset=part.stat().st_size if part.exists() else 0
            headers={'Range':f'bytes={offset}-'} if offset else {}
            with requests.get(url,headers=headers,stream=True,timeout=(30,300)) as r:
                r.raise_for_status()
                append=offset and r.status_code==206
                with open(part,'ab' if append else 'wb') as f:
                    for chunk in r.iter_content(8*1024*1024):f.write(chunk)
            actual=md5(part)
            if actual!=digest:raise RuntimeError(f'{name}: expected {digest}, received {actual}')
            os.replace(part,path)
        print(f'verified {name} {path.stat().st_size} bytes {digest}',flush=True)
        manifest.append(dict(name=name,url=url,md5=digest,bytes=path.stat().st_size))
    # GEARS names the public file .tar.gz, but it is a ZIP archive.
    if not (dest/'norman/perturb_processed.h5ad').exists():
        with zipfile.ZipFile(dest/'norman.tar.gz') as z:
            for info in z.infolist():
                target=(dest/info.filename).resolve()
                if dest.resolve() not in target.parents:raise RuntimeError('Unsafe archive path')
            z.extractall(dest)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
