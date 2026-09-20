#!/usr/bin/env python3
"""Wait for the known upload process, then verify Drive hashes and record completion."""
from pathlib import Path
import json
import subprocess
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/paper-writer-handoff-20260919'
state=json.loads((OUT/'audit/upload_process.json').read_text())
pid=state['pid'];proc=Path('/proc')/str(pid)
initial=(proc/'stat').read_text().split()[21] if proc.exists() else None
print(f'Waiting for upload PID {pid}',flush=True)
while proc.exists():
    try:
        stat=(proc/'stat').read_text().split()
        if stat[21]!=initial or stat[2]=='Z':break
    except FileNotFoundError:break
    time.sleep(5)

common=['--config','/root/.config/rclone-vpo-sft/rclone.conf','--drive-root-folder-id',state['folder_id'],
        '--tpslimit','1','--low-level-retries','10','--retries','3']
# Refresh the standalone reading guide after transfer, before final hash checks.
packet=json.loads((OUT/'packet_receipt.json').read_text())
guide=OUT/'drive_upload/REVIEW_zh.md'
if guide.exists():
    text=guide.read_text().replace('新版写手ZIP正在打包上传，完成后以packet_receipt.json和Drive的UPLOAD_COMPLETE.json为准。',
        f'新版写手ZIP为 {packet["zip_bytes"]/1e9:.3f} GB，展开 {packet["uncompressed_bytes"]/1e9:.3f} GB；上传与校验状态以Drive的UPLOAD_COMPLETE.json为准。')
    guide.write_text(text);(OUT/'REVIEW_zh.md').write_text(text)
    subprocess.run(['rclone','copyto',str(guide),'vpo-sft-drive:REVIEW_zh.md','--checksum']+common,check=True,cwd=ROOT)
cmd=['rclone','check',str(OUT/'drive_upload'),'vpo-sft-drive:','--one-way','--checkers','2',
     '--combined',str(OUT/'audit/writer_drive_check.txt'),'--log-level','INFO',
     '--log-file',str(OUT/'audit/writer_drive_check.log')]+common
subprocess.run(cmd,check=True,cwd=ROOT)
checks=(OUT/'audit/writer_drive_check.txt').read_text().splitlines()
files=[p for p in (OUT/'drive_upload').rglob('*') if p.is_file()]
assert len(checks)==len(files) and all(s.startswith('= ') for s in checks),checks
assert {s[2:] for s in checks}=={str(p.relative_to(OUT/'drive_upload')) for p in files}
receipt=json.loads((OUT/'packet_receipt.json').read_text())
receipt.update(uploaded=True,verified_utc=datetime.now(timezone.utc).isoformat(),
               destination_url='https://drive.google.com/drive/folders/'+state['folder_id'],
               drive_files_verified=len(files),drive_hash='MD5',drive_differences=0,
               table_status='Original supplied tables preserved; metric/seed differences and uniform-protocol tables included')
(OUT/'packet_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
marker=OUT/'audit/UPLOAD_COMPLETE.json';marker.write_text(json.dumps(receipt,indent=2)+'\n')
subprocess.run(['rclone','copyto',str(marker),'vpo-sft-drive:UPLOAD_COMPLETE.json','--checksum']+common,check=True,cwd=ROOT)
(OUT/'audit/writer_drive_verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
