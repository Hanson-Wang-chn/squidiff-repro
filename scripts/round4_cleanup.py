"""Retire completed old optimizer snapshots, keeping their inference weights."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def main():
    log = ROOT/'logs/round4/checkpoint_retirement.json'
    if log.exists():
        raise RuntimeError('Retirement already recorded; do not overwrite its evidence')
    log.parent.mkdir(parents=True, exist_ok=True)
    before = shutil.disk_usage(ROOT).free
    records = []
    paths = sorted((ROOT/'results/norman').glob('*/latest.pt'))
    paths += sorted(p for p in (ROOT/'results/ipsc').glob('*/latest.pt')
                    if p.parent.name not in ['v1', 'source_fp32_s42'])
    for p in paths:
        state = json.loads((p.parent/'status.json').read_text())
        assert state['status'] == 'trained' and state['step'] == 100000, p
        final = p.parent/'step_100000.pt'
        assert final.exists() and final.stat().st_size > 100_000_000, final
        row = dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                   sha256=sha(p), final_weight=str(final.relative_to(ROOT)),
                   final_sha256=sha(final), training_step=100000)
        records.append(row)
        log.write_text(json.dumps(dict(status='in_progress',records=records), indent=2))
        p.unlink()
        print('RETIRED', row['path'], flush=True)
    after = shutil.disk_usage(ROOT).free
    log.write_text(json.dumps(dict(status='complete',time=time.time(),
        authorization='User explicitly allowed old checkpoint deletion for round4',
        retained='All final/best inference weights, data, predictions and logs; current round2/3 resume states',
        consequence='Retired historical runs cannot resume optimizer/RNG state; inference remains available',
        free_before=before,free_after=after,bytes_unlinked=sum(r['bytes'] for r in records),
        records=records),indent=2))
    print('FREE_GIB', after/1024**3, flush=True)

if __name__ == '__main__': main()
