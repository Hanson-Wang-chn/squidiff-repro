"""Share exact duplicate checkpoint bytes while retaining all historical paths.

Only completed, immutable 100k runs in round2/round3 are eligible. Every pair is
SHA256-checked before replacement by a hard link; training saves are atomic.
"""
import hashlib
import json
import os
from pathlib import Path
from common import ROOT,write_json


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    log=ROOT/'logs/round3/exact_checkpoint_dedup.json'
    records=json.loads(log.read_text()) if log.exists() else []
    for root in [ROOT/'results/round2',ROOT/'results/round3']:
        for folder in root.glob('*_s42'):
            p=folder/'status.json'
            if not p.exists():continue
            s=json.loads(p.read_text())
            if s.get('status')!='trained' or s.get('step')!=100000:continue
            for source,target in [(folder/'final.pt',folder/'step_100000.pt'),
                                  (folder/'best.pt',folder/'step_010000.pt')]:
                if not source.exists() or not target.exists() or os.path.samefile(source,target):continue
                if source.stat().st_size!=target.stat().st_size:continue
                h=digest(source)
                if digest(target)!=h:continue
                size=target.stat().st_size
                assert target.stat().st_nlink==1,target
                temp=target.with_suffix('.dedup.tmp')
                os.link(source,temp)
                os.replace(temp,target)
                assert os.path.samefile(source,target) and digest(target)==h
                records.append(dict(source=str(source.relative_to(ROOT)),target=str(target.relative_to(ROOT)),
                    bytes_reclaimed=size,sha256=h,all_file_names_and_bytes_preserved=True))
                write_json(records,log)
    print('EXACT_CHECKPOINT_DEDUP',len(records),'pairs,',sum(r['bytes_reclaimed'] for r in records),'bytes reclaimed')


if __name__=='__main__':main()
