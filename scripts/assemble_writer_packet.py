#!/usr/bin/env python3
"""Select/copy an immutable writer evidence snapshot; no training or API calls."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from datetime import datetime, timezone
from writer_handoff_catalog import ROOT, ROOTS, OUT, PACKAGE, select_file
from inventory_writer_data import category


def csvwrite(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stage():
    audit=OUT/'audit';(PACKAGE/'inventory').mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((audit/'all_files.csv').open()))
    index={(r['root'],r['relative_path']):r for r in rows}
    # Include current plotting/helper files, including files created after the scan.
    extras=list((ROOT/'runs/paper-2x2-rl-20260919').rglob('*'))
    extras += [p for p in (ROOT/'scripts').glob('*.py') if any(x in p.name for x in ['writer','credit_case'])]
    for p in extras:
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(ROOT);st=p.stat()
        index[('shared',str(rel))]=dict(root='shared',relative_path=str(rel),category=category(rel),suite='/'.join(rel.parts[:2]),bytes=st.st_size)
    selected=[];excluded=defaultdict(lambda:dict(files=0,bytes=0));sections=defaultdict(lambda:dict(files=0,bytes=0))
    for r in index.values():
        section,reason=select_file(r)
        if not section:
            excluded[reason]['files']+=1;excluded[reason]['bytes']+=int(r['bytes']);continue
        src=ROOTS[r['root']]/r['relative_path']
        if not src.is_file():raise FileNotFoundError(src)
        size=src.stat().st_size
        selected.append(dict(source=str(src),destination=str(Path('evidence')/section/r['root']/r['relative_path']),
                             section=section,category=r['category'],bytes=size,reason=reason))
        sections[section]['files']+=1;sections[section]['bytes']+=size
    assert sum(r['bytes'] for r in selected)<8_500_000_000
    csvwrite(audit/'selection_plan.csv',selected)
    summary=dict(selected_files=len(selected),selected_bytes=sum(r['bytes'] for r in selected),sections=sections,exclusions=excluded)
    (audit/'selection_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f"Selected {len(selected)} files, {summary['selected_bytes']/1e9:.3f} GB",flush=True)
    def copy(row):
        src=Path(row['source']);dst=PACKAGE/row['destination'];dst.parent.mkdir(parents=True,exist_ok=True)
        before=src.stat();h=hashlib.sha256()
        with src.open('rb') as a,dst.open('wb') as b:
            while chunk:=a.read(1024*1024):h.update(chunk);b.write(chunk)
        after=src.stat()
        if before.st_mtime_ns!=after.st_mtime_ns or before.st_size!=after.st_size:raise RuntimeError(f'Source changed during copy: {src}')
        return dict(**row,sha256=h.hexdigest())
    manifest=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,row in enumerate(pool.map(copy,selected),1):
            manifest.append(row)
            if i%10000==0:print(f'Copied {i}/{len(selected)}',flush=True)
    csvwrite(PACKAGE/'inventory/source_manifest.csv',manifest)
    for name in ['storage_by_category.csv','storage_by_suite.csv','scan.json','selection_summary.json']:
        shutil.copy2(audit/name,PACKAGE/'inventory'/name)
    # The full index is useful for discovery but need not be expanded on disk.
    import gzip
    with (audit/'all_files.csv').open('rb') as a,gzip.open(PACKAGE/'inventory/all_project_files.csv.gz','wb',compresslevel=6) as b:shutil.copyfileobj(a,b)
    suite_rows=[]
    for r in csv.DictReader((audit/'storage_by_suite.csv').open()):
        if r['suite'].startswith(('runs/','analysis/')):
            path=ROOTS[r['root']]/r['suite']
            suite_rows.append(dict(**r,source=str(path),kind='directory' if path.is_dir() else 'standalone artifact'))
    csvwrite(PACKAGE/'inventory/all_experiment_locations.csv',suite_rows)
    print('Evidence snapshot and source manifest complete.',flush=True)


def archive():
    files=sorted(p for p in PACKAGE.rglob('*') if p.is_file())
    size=sum(p.stat().st_size for p in files)
    assert size<10_000_000_000,size
    dest=OUT/'VPO-RM-writer-packet.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for p in files:z.write(p,Path('writer_packet')/p.relative_to(PACKAGE))
    with zipfile.ZipFile(dest) as z:
        assert len(z.infolist())==len(files)
        bad=z.testzip()
        if bad:raise RuntimeError(f'ZIP CRC failure: {bad}')
    digest=hashlib.sha256()
    with dest.open('rb') as f:
        while b:=f.read(4*1024*1024):digest.update(b)
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=len(files),uncompressed_bytes=size,
                 zip_bytes=dest.stat().st_size,sha256=digest.hexdigest(),zip_crc_verified=True,uploaded=False)
    assert receipt['zip_bytes']<10_000_000_000
    (OUT/'packet_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (OUT/'VPO-RM-writer-packet.zip.sha256').write_text(digest.hexdigest()+'  '+dest.name+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    archive() if args.archive else stage()
