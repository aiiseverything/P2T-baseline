#!/usr/bin/env python3
"""Finish the verified final-scope archive, replace Drive files, verify hashes."""
from pathlib import Path
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/paper-writer-handoff-20260919'
state=json.loads((OUT/'audit/final_archive_process.json').read_text());pid=state['pid']
if pid:
    proc=Path('/proc')/str(pid)
    try:initial=(proc/'stat').read_text().split()[21]
    except FileNotFoundError:initial=None
    print(f'Waiting for final archive PID {pid}',flush=True)
    while proc.exists():
        try:
            fields=(proc/'stat').read_text().split()
            if fields[21]!=initial or fields[2]=='Z':break
        except FileNotFoundError:break
        time.sleep(3)
receipt=json.loads((OUT/'packet_receipt.json').read_text())
assert receipt['zip_crc_verified'] and receipt['zip_bytes']==(OUT/'VPO-RM-writer-packet.zip').stat().st_size
assert receipt['zip_bytes']<10_000_000_000 and receipt['uncompressed_bytes']<10_000_000_000
with zipfile.ZipFile(OUT/'VPO-RM-writer-packet.zip') as z:
    names=z.namelist()
    assert not any('/historical/' in n for n in names)
    assert not any(any(p.startswith(('llama31-rl-canonical-20260918','llama31-sft-aligned-20260917','sft-llama31-8b-instruct-clean2k5e2-20260917')) for p in Path(n).parts) for n in names)
    for n in names:
        if any(p in Path(n).parts for p in ['llama31-evals-20260918','llama31-alpacaeval-20260918']):
            assert 'base' in Path(n).parts and any(p in Path(n).parts for p in ['results','generations']),n
    assert 'writer_packet/EXPERIMENT_SCOPE.md' in names
    assert 'writer_packet/table_review/reconciled_main.csv' in names
    assert len(names)==receipt['files']
assert (OUT/'drive_upload/VPO-RM-writer-packet.zip').samefile(OUT/'VPO-RM-writer-packet.zip')
(OUT/'audit/final_archive_scope_verified.json').write_text(json.dumps(dict(
    verified_utc=datetime.now(timezone.utc).isoformat(),excluded_llama_instruct_sft_files=0,
    original_instruct_baseline_retained=True,zip_sha256=receipt['sha256'],files=receipt['files']),indent=2)+'\n')
print(f'Final scope verified; uploading {receipt["zip_bytes"]/1e9:.3f} GB',flush=True)
command=['rclone','copy',str(OUT/'drive_upload'),'vpo-sft-drive:',
         '--config','/root/.config/rclone-vpo-sft/rclone.conf','--drive-root-folder-id',state['folder_id'],
         '--checksum','--transfers','3','--checkers','3','--tpslimit','2','--drive-chunk-size','64M',
         '--low-level-retries','10','--retries','3','--stats','30s','--log-level','INFO',
         '--log-file',str(OUT/'audit/writer_drive_upload_final.log')]
process=subprocess.Popen(command,cwd=ROOT)
(OUT/'audit/upload_process.json').write_text(json.dumps(dict(pid=process.pid,folder_id=state['folder_id']),indent=2)+'\n')
if process.wait()!=0:raise RuntimeError('Final packet upload failed; inspect writer_drive_upload_final.log')
subprocess.run([sys.executable,str(ROOT/'scripts/finalize_writer_drive.py')],check=True,cwd=ROOT)
print('FINAL WRITER PACKET UPLOADED AND VERIFIED',flush=True)
